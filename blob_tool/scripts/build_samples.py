"""Build test recordings with ground-truth speaker turns from AMI and ClassBank.

Usage: uv run python scripts/build_samples.py [--all] [--duration 30] [--force]

  default  one recording per dataset/corpus, cut to its busiest <duration> s window → samples/sample/
  --all    every recording under samples/raw/, full length                         → samples/all/

Raw data lives in samples/raw/<dataset>/:
  ami/        downloaded automatically (ES2002a corner camera, mixed headset audio, RTTM). Other
              meetings dropped in as <id>.Corner.avi + <id>.Mix-Headset.wav + <id>.rttm are picked up too.
  classbank/  manual download (free TalkBank login). Unzip a corpus' transcripts and some lesson videos
              into classbank/<corpus>/ (e.g. classbank/Curtis/). Each video pairs with the .cha of the same name.

Output: samples/<mode>/<dataset>/[<corpus>/]<recording>/<recording>.mp4 + .rttm (+ .transcript.tsv for
ClassBank: start, end, speaker, role, text), and samples/<mode>/SOURCES.md with sources and licenses.

Sources (licenses verified 2026-10):
  * AMI Meeting Corpus — © AMI Consortium, CC BY 4.0: https://groups.inf.ed.ac.uk/ami/corpus/license.shtml
  * AMI speaker-turn RTTM — pyannote/AMI-diarization-setup, Apache-2.0
  * ClassBank (https://talkbank.org/class/) — CC BY-NC-SA 4.0; any use must cite the corpus reference

Recordings are fetched from their original hosts and kept in the gitignored samples/ folder; they are
not redistributed by this repository.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
RAW = SAMPLES / "raw"
UA = {"User-Agent": "Mozilla/5.0 (blob-tool sample fetcher)"}
VIDEO_EXT = {".mp4", ".m4v", ".mov", ".avi", ".mpg", ".mpeg"}
BULLET = re.compile(r"\x15(\d+)_(\d+)\x15")

AMI_ID = "ES2002a"
AMI_BASE = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{}"
AMI_RTTM = "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/only_words/rttms/train/{}.rttm"
AMI_CITE = ('Carletta et al., "The AMI Meeting Corpus: A Pre-announcement", MLMI 2005; RTTM "only_words" setup: '
            "Landini et al., arXiv:2012.14952")
AMI_LICENSE = "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (RTTM tooling Apache-2.0)"

# corpus folder name -> (page, DOI, required reference)
CORPORA = {
    "TIMSS-Math-USA": ("https://talkbank.org/class/access/TIMSS-Math/USA.html", "doi:10.21415/T5Z30S",
                       "Stigler, Gallimore & Hiebert (2000), Educational Psychologist 35, 87-100"),
    "Curtis": ("https://talkbank.org/class/access/Curtis.html", "doi:10.21415/T5V890",
               "Lehrer & Curtis (2000), Teaching Children Mathematics 6(5), 324"),
}
CB_LICENSE = "[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)"


@dataclass
class Utt:
    speaker: str
    start: float
    end: float
    text: str = ""


@dataclass
class Recording:
    dataset: str          # folder under samples/<mode>/
    group: str            # corpus folder ("" for AMI)
    name: str
    video: Path
    audio: Path | None    # separate audio track to mux in (AMI)
    utts: list[Utt]
    roles: dict[str, str] | None  # ClassBank only → .transcript.tsv
    desc: str
    source: str
    license: str
    cite: str


def download(url: str, dst: Path) -> Path:
    if dst.exists():
        print(f"  ✓ {dst.name} (cached)")
        return dst
    print(f"  ↓ {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dst)
    return dst


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


def duration_of(video: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)],
                         check=True, capture_output=True, text=True).stdout
    return float(json.loads(out)["format"]["duration"])


def read_rttm(path: Path) -> list[Utt]:
    utts = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            start = float(p[3])
            utts.append(Utt(p[7], start, start + float(p[4])))
    return utts


def clean(text: str) -> str:
    text = BULLET.sub("", text)
    text = re.sub(r"\[[^\]]*\]|&=\S+|\(\.+\)|[<>]|\+[./!?\"]+", " ", text)  # CHAT codes, pauses, overlaps
    return " ".join(text.split())


def parse_cha(path: Path) -> tuple[list[Utt], dict[str, str]]:
    # join continuation lines (leading tab) onto the tier they belong to
    tiers: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("\t") and tiers:
            tiers[-1] += " " + line.strip()
        elif line:
            tiers.append(line)

    roles: dict[str, str] = {}
    utts: list[Utt] = []
    for tier in tiers:
        if tier.startswith("@Participants:"):
            for entry in tier.split(":", 1)[1].split(","):
                words = entry.split()
                if words:
                    roles[words[0]] = words[-1] if len(words) > 1 else ""
        elif tier.startswith("*"):
            spk, _, body = tier[1:].partition(":")
            m = BULLET.search(body)
            if m and int(m.group(2)) > int(m.group(1)):
                utts.append(Utt(spk.strip(), int(m.group(1)) / 1000, int(m.group(2)) / 1000, clean(body)))
    return utts, roles


def busiest_window(utts: list[Utt], length: float, total: float) -> float:
    """Start of the window with the most speaker changes (ties: more distinct speakers)."""
    best, best_score = 0.0, (-1, -1)
    for u in utts:
        start = min(u.start, max(total - length, 0.0))
        inside = [x.speaker for x in utts if x.start >= start and x.end <= start + length]
        score = (sum(a != b for a, b in zip(inside, inside[1:])), len(set(inside)))
        if score > best_score:
            best, best_score = start, score
    return best


def clip(utts: list[Utt], start: float, length: float) -> list[Utt]:
    out = []
    for u in utts:
        s, e = max(u.start - start, 0.0), min(u.end - start, length)
        if e > s:
            out.append(Utt(u.speaker, s, e, u.text))
    return out


def ami_recordings() -> list[Recording]:
    raw = RAW / "ami"
    raw.mkdir(parents=True, exist_ok=True)
    base = AMI_BASE.format(AMI_ID)
    download(f"{base}/video/{AMI_ID}.Corner.avi", raw / f"{AMI_ID}.Corner.avi")
    download(f"{base}/audio/{AMI_ID}.Mix-Headset.wav", raw / f"{AMI_ID}.Mix-Headset.wav")
    download(AMI_RTTM.format(AMI_ID), raw / f"{AMI_ID}.rttm")

    recs = []
    for video in sorted(raw.glob("*.Corner.avi")):
        mid = video.name.split(".")[0]
        audio, rttm = raw / f"{mid}.Mix-Headset.wav", raw / f"{mid}.rttm"
        if audio.exists() and rttm.exists():
            recs.append(Recording("ami", "", mid, video, audio, read_rttm(rttm), None,
                                  f"AMI meeting {mid}, corner camera + mixed headset audio",
                                  AMI_BASE.format(mid) + "/", AMI_LICENSE, AMI_CITE))
    return recs


def classbank_recordings() -> list[Recording]:
    root = RAW / "classbank"
    if not root.exists():
        return []
    videos = {v.stem: v for v in root.rglob("*") if v.suffix.lower() in VIDEO_EXT}
    recs = []
    for cha in sorted(root.rglob("*.cha")):
        video = videos.get(cha.stem)
        if video is None:
            continue
        utts, roles = parse_cha(cha)
        if not utts:
            print(f"  ! {cha.name}: no timed utterances, skipped")
            continue
        corpus = cha.relative_to(root).parts[0]
        page, doi, ref = CORPORA.get(corpus, ("https://talkbank.org/class/", "see corpus page", "see corpus page"))
        recs.append(Recording("classbank", corpus, cha.stem, video, None, utts, roles,
                              f"ClassBank {corpus} lesson {cha.stem}", page, CB_LICENSE, f"{ref}; {doi}"))
    return recs


def build(rec: Recording, out_root: Path, length: float | None, force: bool) -> str:
    folder = out_root / rec.dataset / rec.group / rec.name
    folder.mkdir(parents=True, exist_ok=True)
    total = duration_of(rec.video)
    if length and length < total:
        start, span = busiest_window(rec.utts, length, total), length
        cut = ["-ss", f"{start:.3f}", "-t", f"{span:.3f}"]
        mods = f"cut to {start:.0f}–{start + span:.0f} s of {total:.0f} s (busiest window), turns shifted to the clip"
    else:
        start, span, cut, mods = 0.0, total, [], "full length"

    out = folder / f"{rec.name}.mp4"
    if force or not out.exists():
        print(f"  → {out.relative_to(SAMPLES)} ({start:.0f}–{start + span:.0f} s of {total:.0f} s)")
        inputs = [*cut, "-i", str(rec.video)] + ([*cut, "-i", str(rec.audio)] if rec.audio else [])
        maps = ["-map", "0:v:0", "-map", "1:a:0"] if rec.audio else []
        ffmpeg(*inputs, *maps, "-vf", "scale=-2:'min(1080,ih)'", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", str(out))
    else:
        print(f"  ✓ {out.relative_to(SAMPLES)} (cached)")

    turns = clip(rec.utts, start, span)
    (folder / f"{rec.name}.rttm").write_text("".join(
        f"SPEAKER {rec.name} 1 {u.start:.3f} {u.end - u.start:.3f} <NA> <NA> {u.speaker} <NA> <NA>\n" for u in turns))
    files = ".mp4, .rttm"
    if rec.roles is not None:
        (folder / f"{rec.name}.transcript.tsv").write_text("start\tend\tspeaker\trole\ttext\n" + "".join(
            f"{u.start:.3f}\t{u.end:.3f}\t{u.speaker}\t{rec.roles.get(u.speaker, '')}\t{u.text}\n" for u in turns))
        files += ", .transcript.tsv"

    speakers = sorted({u.speaker for u in turns})
    return (f"| `{folder.relative_to(out_root)}/` ({files}) | {rec.desc}; {len(turns)} turns, "
            f"{len(speakers)} speakers ({', '.join(speakers)}) | {rec.source} | {rec.license} | {rec.cite} | "
            f"{mods}; re-encoded to H.264/AAC ≤1080p |")


def main() -> None:
    args = sys.argv[1:]
    full, force = "--all" in args, "--force" in args
    length = None if full else float(args[args.index("--duration") + 1]) if "--duration" in args else 30.0
    out_root = SAMPLES / ("all" if full else "sample")

    print("AMI meeting corpus:")
    recs = ami_recordings()
    print("ClassBank classroom lessons:")
    classbank = classbank_recordings()
    if not classbank:
        print(f"  (none: put a corpus' transcripts and lesson videos under {RAW / 'classbank'}/<corpus>/)")
    recs += classbank
    if not full:
        picked: dict[tuple[str, str], Recording] = {}
        for r in recs:
            picked.setdefault((r.dataset, r.group), r)
        recs = list(picked.values())

    rows = [build(r, out_root, length, force) for r in recs]
    (out_root / "SOURCES.md").write_text(
        f"# Sources: samples/{out_root.name}/\n\n"
        "Downloaded from the original hosts (ClassBank by hand, with a TalkBank login) for local testing only. "
        "This includes real classroom footage of children: keep it local, never commit or redistribute it.\n\n"
        "| Folder | Content | Source | License | Cite | Modifications |\n|---|---|---|---|---|---|\n"
        + "\n".join(rows) + "\n\n"
        "Not included: OUC-CGE classroom group engagement dataset (Lu et al., Scientific Data 12, 644, 2025; "
        "OSF DOI 10.17605/OSF.IO/BRD2C; adult participants) for larger-scale evaluation.\n")
    print(f"\n{len(rows)} recording(s) ready in {out_root}")


if __name__ == "__main__":
    main()

"""Download openly licensed test clips into blob_tool/samples/.

Usage: uv run python scripts/fetch_samples.py [--force]

Sources (licenses verified 2026-10):
  * Pexels classroom clips — Pexels License (free use and modification, no attribution required,
    no redistribution of unmodified media): https://www.pexels.com/license/
  * AMI Meeting Corpus ES2002a — © AMI Consortium, CC BY 4.0:
    https://groups.inf.ed.ac.uk/ami/corpus/license.shtml
  * Speaker-turn RTTM — pyannote/AMI-diarization-setup, Apache-2.0:
    https://github.com/pyannote/AMI-diarization-setup

Clips are fetched from the original hosts and kept in the gitignored samples/ folder; they are
not redistributed by this repository.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
UA = {"User-Agent": "Mozilla/5.0 (blob-tool sample fetcher)"}

PEXELS = [
    # (pexels video id, output name, description)
    (8617053, "classroom_group_listening.mp4", "A group of students listening to their teacher (Yan Krukau)"),
    (8342233, "classroom_teacher_teaching.mp4", "Video of a teacher teaching students"),
    (8616777, "classroom_vertical_phone.mp4", "A man teaching his students — vertical 1080x1920, like a phone recording"),
]

AMI_ID = "ES2002a"
AMI_BASE = f"https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{AMI_ID}"
AMI_RTTM = f"https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/only_words/rttms/train/{AMI_ID}.rttm"
AMI_START, AMI_DURATION = 50.0, 180.0  # skip the silent set-up, keep 3 minutes of conversation


def download(url: str, dst: Path, force: bool) -> Path:
    if dst.exists() and not force:
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


def fetch_pexels(force: bool) -> list[str]:
    lines = []
    for vid, name, desc in PEXELS:
        raw = download(f"https://www.pexels.com/download/video/{vid}/", SAMPLES / "raw" / f"pexels_{vid}.mp4", force)
        out = SAMPLES / name
        if force or not out.exists():
            # 20 s, ≤1080p keeps processing quick while still being realistic
            ffmpeg("-i", str(raw), "-t", "20", "-vf", "scale=-2:'min(1080,ih)'",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an", str(out))
        lines.append(f"| `{name}` | {desc} | https://www.pexels.com/video/{vid}/ | "
                     "[Pexels License](https://www.pexels.com/license/) | trimmed to 20 s, scaled to ≤1080p, audio removed |")
    return lines


def fetch_ami(force: bool) -> list[str]:
    raw = SAMPLES / "raw"
    video = download(f"{AMI_BASE}/video/{AMI_ID}.Corner.avi", raw / f"{AMI_ID}.Corner.avi", force)
    audio = download(f"{AMI_BASE}/audio/{AMI_ID}.Mix-Headset.wav", raw / f"{AMI_ID}.Mix-Headset.wav", force)
    rttm = download(AMI_RTTM, raw / f"{AMI_ID}.rttm", force)

    out = SAMPLES / f"ami_{AMI_ID}_meeting.mp4"
    if force or not out.exists():
        ffmpeg("-ss", str(AMI_START), "-t", str(AMI_DURATION), "-i", str(video),
               "-ss", str(AMI_START), "-t", str(AMI_DURATION), "-i", str(audio),
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", str(out))

    # ground-truth speaker turns, shifted to the trimmed clip's timeline
    shifted = []
    for line in rttm.read_text().splitlines():
        p = line.split()
        start, dur = float(p[3]) - AMI_START, float(p[4])
        if start + dur <= 0 or start >= AMI_DURATION:
            continue
        p[3] = f"{max(start, 0):.3f}"
        p[4] = f"{min(dur + min(start, 0), AMI_DURATION - max(start, 0)):.3f}"
        shifted.append(" ".join(p))
    (SAMPLES / f"ami_{AMI_ID}_meeting.rttm").write_text("\n".join(shifted) + "\n")
    speakers = sorted({ln.split()[7] for ln in shifted})
    return [
        f"| `ami_{AMI_ID}_meeting.mp4` | AMI meeting {AMI_ID}, corner camera + mixed headset audio, "
        f"{len(speakers)} speakers. © AMI Consortium; cite Carletta et al., \"The AMI Meeting Corpus: "
        f"A Pre-announcement\", MLMI 2005 | {AMI_BASE}/ | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) "
        f"| trimmed to {AMI_START:.0f}–{AMI_START + AMI_DURATION:.0f} s, muxed and re-encoded to H.264/AAC |",
        f"| `ami_{AMI_ID}_meeting.rttm` | ground-truth speaker turns, \"only_words\" setup; cite Landini et al., "
        f"arXiv:2012.14952 | https://github.com/pyannote/AMI-diarization-setup | Apache-2.0 "
        f"(annotations CC BY 4.0) | timestamps shifted and clipped to the trimmed window |",
    ]


def main() -> None:
    force = "--force" in sys.argv
    (SAMPLES / "raw").mkdir(parents=True, exist_ok=True)
    print("Pexels classroom clips:")
    rows = fetch_pexels(force)
    print("AMI meeting (multi-speaker audio + video):")
    rows += fetch_ami(force)
    (SAMPLES / "SOURCES.md").write_text(
        "# Sample data sources\n\n"
        "These files are downloaded from their original hosts for local testing only. Do not commit or "
        "redistribute them (the Pexels License forbids redistributing unmodified media).\n\n"
        "| File | Content | Source | License | Modifications |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
        "Not auto-downloaded: OUC-CGE classroom group engagement dataset (Lu et al., Scientific Data 12, 644, 2025; "
        "OSF DOI 10.17605/OSF.IO/BRD2C; adult participants) for larger-scale evaluation.\n"
    )
    print(f"\nSamples ready in {SAMPLES}")


if __name__ == "__main__":
    main()

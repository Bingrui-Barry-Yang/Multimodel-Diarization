"""LLM comparison runs with a fake Gemini client (no network, no key)."""

from types import SimpleNamespace

import pytest

from blob_tool import llm


class FakeClient:
    """Records everything one client sees; answers with the bytes it was given."""

    instances: list["FakeClient"] = []

    def __init__(self):
        self.uploads, self.calls, self.deleted = [], [], []
        FakeClient.instances.append(self)
        self.files = SimpleNamespace(upload=self._upload, get=self._get, delete=self._delete)
        self.models = SimpleNamespace(generate_content=self._generate)

    def _upload(self, file, config):
        assert not isinstance(file, (str, bytes)) and hasattr(file, "read")  # stream, not a path
        self.uploads.append((file.read(), config))
        return SimpleNamespace(name=f"files/{len(self.uploads)}", state=SimpleNamespace(name="PROCESSING"))

    def _get(self, name):
        return SimpleNamespace(name=name, state=SimpleNamespace(name="ACTIVE"))

    def _delete(self, name):
        self.deleted.append(name)

    def _generate(self, model, contents, config):
        self.calls.append((model, contents, config))
        return SimpleNamespace(text=f"saw {self.uploads[-1][0].decode()}")


@pytest.fixture
def job(tmp_path):
    for variant, name in llm.VARIANTS.items():
        (tmp_path / name).write_bytes(variant.encode())
    j = SimpleNamespace(id="job1", dir=tmp_path, status="done")
    return j


@pytest.fixture
def manager(job, monkeypatch):
    FakeClient.instances = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    return llm.LlmManager({job.id: job}, client_factory=FakeClient, start_worker=False)


def test_each_variant_is_an_isolated_request(job, manager):
    run = manager.submit(job, "Who speaks?", list(llm.VARIANTS), llm.MODELS[0], confirm_original=True)
    manager.process(manager.queue.get_nowait())

    saved = manager.get(job, run.id)
    assert saved["status"] == "done"
    assert {v: a["text"] for v, a in saved["answers"].items()} == {v: f"saw {v}" for v in llm.VARIANTS}
    # one fresh client per variant, each with exactly one upload, one call and one delete
    assert len(FakeClient.instances) == len(llm.VARIANTS)
    for c in FakeClient.instances:
        assert len(c.uploads) == 1 and len(c.calls) == 1 and c.deleted == ["files/1"]
        assert c.uploads[0][1]["display_name"] == "video.mp4"
        model, contents, config = c.calls[0]
        assert contents[1] == "Who speaks?" and len(contents) == 2 and config["temperature"] == 0.0
    assert manager.list(job)[0]["id"] == run.id


def test_original_needs_explicit_confirmation(job, manager):
    with pytest.raises(ValueError, match="confirm_original"):
        manager.submit(job, "Who speaks?", ["blobs", "original"], llm.MODELS[0], confirm_original=False)
    manager.submit(job, "Who speaks?", ["blobs"], llm.MODELS[0], confirm_original=False)
    for v in ("blurred", "faceblur"):  # blurred versions still hold original pixels
        with pytest.raises(ValueError, match="confirm_original"):
            manager.submit(job, "Who speaks?", ["pose", v], llm.MODELS[0], confirm_original=False)


def test_rejects_missing_variants_and_bad_input(job, manager):
    (job.dir / "pose.mp4").unlink()
    with pytest.raises(ValueError, match="not available"):
        manager.submit(job, "q", ["pose"], llm.MODELS[0], False)
    with pytest.raises(ValueError):
        manager.submit(job, "q", ["nope"], llm.MODELS[0], False)
    with pytest.raises(ValueError):
        manager.submit(job, "  ", ["blobs"], llm.MODELS[0], False)
    with pytest.raises(ValueError):
        manager.submit(job, "q", ["blobs"], "some-other-model", False)
    assert manager.get(job, "../../etc") is None


def test_original_discarded_before_the_run_starts(job, manager):
    run = manager.submit(job, "q", ["original", "blobs"], llm.MODELS[0], confirm_original=True)
    (job.dir / "original.mp4").unlink()
    manager.process(manager.queue.get_nowait())
    answers = manager.get(job, run.id)["answers"]
    assert answers["original"]["status"] == "error" and answers["blobs"]["text"] == "saw blobs"
    assert len(FakeClient.instances) == 1  # nothing was uploaded for the original

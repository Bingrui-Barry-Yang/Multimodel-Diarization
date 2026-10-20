# Multimodal Diarization on Privacy-Preserving Classroom Data

The goal is to study classroom interaction without keeping anything identifiable about the children.

1. **Blob tool** ([`blob_tool/`](blob_tool/)): turns raw classroom video into tracked colour blobs and altered voices. Today it is a localhost app; later it will run on a phone that records and saves only the processed output, never the original.
2. **Diarization on blobs** (next): feed the blob video, `tracks.json` and altered audio into local multimodal models plus traditional diarization (e.g. pyannote). The research question is whether multimodal models can still extract speaker and turn-taking features when all they get is blobs.

## Quick start

```bash
make run           # blob tool UI at http://127.0.0.1:8000
make sample-data   # openly licensed test clips
make test
```

See [`blob_tool/README.md`](blob_tool/README.md) for engines, outputs, test data and the phone roadmap.

## License

[GNU AGPL-3.0](LICENSE). The blob tool builds on Ultralytics YOLO (AGPL-3.0). Third-party components, datasets and references are listed in [`blob_tool/README.md`](blob_tool/README.md#license).

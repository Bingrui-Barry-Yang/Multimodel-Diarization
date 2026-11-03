# Blob Tool — run `make help` for targets.
HOST ?= 127.0.0.1
PORT ?= 8000
UV   ?= uv

.PHONY: help install install-sam run dev test sample-data all-data docker-build docker-run docker-stop clean

help:
	@echo "make install       install deps with uv (YOLO engine)"
	@echo "make install-sam   also install EdgeTAM / SAM 3 engines (transformers)"
	@echo "make run           start the UI at http://$(HOST):$(PORT) (uses Apple GPU / CUDA if present)"
	@echo "make dev           same, with auto-reload"
	@echo "make test          run unit + smoke tests"
	@echo "make sample-data   one short clip per dataset (AMI + each ClassBank corpus) → samples/sample/"
	@echo "make all-data      every recording in samples/raw/, full length → samples/all/"
	@echo "make docker-build  build the CPU-only container"
	@echo "make docker-run    run the container at http://localhost:$(PORT)"
	@echo "make clean         delete processed jobs in workspace/"

install:
	$(UV) sync --inexact  # --inexact keeps the optional sam extra if installed

install-sam:
	$(UV) sync --extra sam

run: install
	@echo "→ http://$(HOST):$(PORT)"
	$(UV) run uvicorn blob_tool.app:app --host $(HOST) --port $(PORT)

dev: install
	$(UV) run uvicorn blob_tool.app:app --host $(HOST) --port $(PORT) --reload --reload-dir src

test: install
	$(UV) run pytest -q

sample-data: install
	$(UV) run python scripts/build_samples.py

all-data: install
	$(UV) run python scripts/build_samples.py --all

docker-build:
	docker compose build

docker-run:
	PORT=$(PORT) docker compose up

docker-stop:
	docker compose down

clean:
	rm -rf workspace/*

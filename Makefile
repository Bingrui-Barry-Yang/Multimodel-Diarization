# Thin wrapper: every target is delegated to blob_tool/Makefile.
TOOL := blob_tool

.PHONY: help install install-sam run dev test sample-data all-data docker-build docker-run docker-stop clean

help install install-sam run dev test sample-data all-data docker-build docker-run docker-stop clean:
	$(MAKE) -C $(TOOL) $@

PYTHON ?= python
GO ?= go

.PHONY: test lint format build-go-fastsearch

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check drug_discovery tests

format:
	$(PYTHON) -m black drug_discovery tests

build-go-fastsearch:
	mkdir -p tools/bin
	cd tools/go/fastsearch && $(GO) build -o ../../bin/zane-fastsearch .

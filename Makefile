export HF_HOME := /goinfre/$(USER)/hf-cache
export UV_CACHE_DIR := /goinfre/$(USER)/uv-cache

UV := uv

.PHONY: all install run debug test clean lint lint-strict

all: install

install:
	$(UV) sync

run:
	$(UV) run python -m src

debug:
	$(UV) run python -m pdb -m src

test:
	$(UV) run pytest tests/

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .mypy_cache .pytest_cache
	rm -rf data/output/*

fclean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .mypy_cache .pytest_cache
	rm -rf data/output/*
	rm -rf .venv/

lint:
	$(UV) run flake8 src/ tests/
	$(UV) run mypy src/ tests/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 src/ tests/
	$(UV) run mypy src/ tests/ --strict
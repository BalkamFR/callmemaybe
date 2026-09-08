UV := uv

.PHONY: all install run debug clean lint lint-strict

all: install

install:
	$(UV) sync

run:
	$(UV) run python -m src

debug:
	$(UV) run python -m pdb -m src

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .mypy_cache .pytest_cache
	rm -rf data/output/*

lint:
	$(UV) run flake8 .
	$(UV) run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict
# Common developer tasks, one command each. Run `make help` for the list.
#
# Every recipe is a single obvious entry point so the project can be built, tested,
# and run without remembering flags. The commands are plain and cross-platform where
# possible; on Windows they run under Git Bash / WSL, or you can read them here and
# run the underlying command directly.

# Use the project's own interpreter. Override on the command line if needed:
#   make test PYTHON=python3.11
PYTHON ?= python
PORT ?= 8000

.DEFAULT_GOAL := help

.PHONY: help install snapshot test lint run clean

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime and test dependencies.
	$(PYTHON) -m pip install -r requirements.txt

snapshot:  ## Rebuild the normalised catalog snapshot from the raw export.
	$(PYTHON) -m scripts.build_snapshot

test:  ## Run the full test suite.
	$(PYTHON) -m pytest

lint:  ## Check formatting and lint rules (ruff).
	$(PYTHON) -m ruff check .

recall:  ## Print the retrieval scoreboard (mean Recall@10 against the samples).
	$(PYTHON) -m scripts.measure_recall

run:  ## Start the API locally with reload (http://127.0.0.1:$(PORT)).
	$(PYTHON) -m uvicorn shl_recommender.api.app:app --reload --port $(PORT)

clean:  ## Remove Python and tooling caches.
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

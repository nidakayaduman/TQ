.PHONY: setup install test run lint format docker-build docker-run clean smoke

PYTHON ?= python3

setup: install

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

run:
	$(PYTHON) -m streamlit run app.py

lint:
	$(PYTHON) -m compileall app.py src tests

format:
	@if command -v ruff >/dev/null 2>&1; then ruff format app.py src tests; else $(PYTHON) -m compileall app.py src tests; fi

docker-build:
	docker build -t tender-iq .

docker-run:
	@if [ -f .env ]; then docker run --rm -p 8501:8501 --env-file .env tender-iq; else docker run --rm -p 8501:8501 tender-iq; fi

clean:
	rm -rf .pytest_cache .ruff_cache .streamlit/cache __pycache__ src/**/__pycache__ tests/**/__pycache__
	find . -name "*.pyc" -delete

smoke:
	$(PYTHON) -m pytest tests/test_ui_smoke.py

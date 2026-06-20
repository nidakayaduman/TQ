.PHONY: install test run lint smoke

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest

run:
	streamlit run app.py

lint:
	python -m compileall app.py src tests

smoke:
	python -m pytest tests/test_ui_smoke.py

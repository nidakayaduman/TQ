.PHONY: install test run

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest

run:
	streamlit run app.py


.PHONY: install install-fijo test eda
install:
	pip install -e ".[dev]"
install-fijo:
	pip install -r requirements.txt
	pip install --no-deps -e .
test:
	pytest -q
eda:
	python -m dengue_pipeline eda

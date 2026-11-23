.PHONY: install test eda
install:
	pip install -e ".[dev]"
test:
	pytest -q
eda:
	python -m dengue_pipeline eda

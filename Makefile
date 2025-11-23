.PHONY: venv train validate all clean

venv:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install pre-commit
	.venv/bin/pre-commit install

train:
	python scripts/train.py --event R1 --outdir models/

validate:
	python scripts/validate_walkforward.py --event R2 --outdir reports/

train-validate:
	python scripts/train_and_validate.py

train-validate-all:
	python scripts/train_and_validate.py --all-scenarios

all: train validate

train-validate:
	python scripts/train_and_validate.py

train-validate-all:
	python scripts/train_and_validate.py --all-scenarios

clean:
	rm -rf .venv
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete



.PHONY: install test clean run lint format venv

venv:
	python -m venv venv
	@echo "Run 'source venv/bin/activate' to activate virtual environment"
	source venv/bin/activate

install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements.txt

test:
	pytest tests/ --cov=src

run:
	python src/main.py

train:
	python src/main.py --train

visualize:
	python src/main.py --visualize

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/
	# rm -rf venv/

lint:
	pylint src/ tests/
	mypy src/ tests/

format:
	black src/ tests/

setup: install
	@echo "Setup complete. Don't forget to activate your virtual environment!"
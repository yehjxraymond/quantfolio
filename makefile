.PHONY: all generate clean build install_requirements install_dev_requirements distribute setup_distribute_env test test_watch

all: install_dev_requirements

install_requirements:
	@echo "Installing requirements..."
	@pip install pipenv
	@pipenv install -e

install_dev_requirements:
	@echo "Installing dev requirements..."
	@sudo pip install pipenv pytest
	@pipenv install -e .[dev]

build: clean
	@echo "Building distribution..."
	@python3 setup.py sdist bdist_wheel

clean:
	@echo "Cleaning up build..."
	@rm -rf build dist

distribute: setup_distribute_env
	@pipenv run twine upload dist/*

setup_distribute_env:
	@echo "[pypi]" > ~/.pypirc
	@echo "username = ${PYPI_USERNAME}" >> ~/.pypirc
	@echo "password = ${PYPI_PASSWORD}" >> ~/.pypirc

test:
	pytest . -s

test_watch:
	ptw tests -- -s
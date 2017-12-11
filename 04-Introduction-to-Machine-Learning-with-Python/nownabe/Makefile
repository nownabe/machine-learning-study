.PHONY: start
start:
	. venv/bin/activate
	jupyter notebook --config .jupyter

.PHONY: init
init:
	virtualenv venv
	. venv/bin/activate
	pip install -r requirements.txt
	python -m spacy download en

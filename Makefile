venvname=venv_snpl
REQUIREMENTS=gh
REQUIREMENTS+=git
REQUIREMENTS+=virtualenv

all:
	@echo "usage: "
	@echo "       make env"
	@echo "       make cleanenv"

check:
	@for f in $(REQUIREMENTS); do \
		type $$f 2> /dev/null 1> /dev/null || (echo "$$f is missing ❌";exit 1); \
	done
	
	@echo "Requirements Check Passed ✅"

env:
	virtualenv $(venvname)
	(\
		source $(venvname)/bin/activate || source $(venvname)/bin/activate.fish; \
		pip install -r utils/requirements.txt \
	)
	
	@echo "Created env $(venvname) ✅"

cleanenv: $(venvname)
	rm -rf $(venvname)
	
	@echo "Removed env $(venvname) ✅"

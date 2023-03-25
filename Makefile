SHELL:=/bin/bash 

VENV           = venv_snlp
VENV_PYTHON    = $(VENV)/bin/python
SYSTEM_PYTHON  = $(or $(shell which python3), $(shell which python))
# If virtualenv exists, use it. If not, find python using PATH
PYTHON         = $(or $(wildcard $(VENV_PYTHON)), $(SYSTEM_PYTHON))

REQUIREMENTS	=	gh
REQUIREMENTS	+=	git
REQUIREMENTS	+=	virtualenv

all:
	@echo "social-network-link-prediction Makefile guide."
	@echo ""
	@echo "Install commands:"
	@echo "       make install      : Install library in the system."
	@echo ""
	@echo "       make install-dev  : Create dev environment (virtualenv) and"
	@echo "                           install dev requirements."
	@echo ""
	@echo "       make install-doc  : Insitall requirement to generate documentation."
	@echo "                           Needs the environment installed (run make install-dev)"
	@echo ""
	@echo "       make install-test : Install requirements for the testing."
	@echo "                           Needs the environment installed (run make install-dev)"
	@echo ""
	@echo "Clean commands:"
	@echo "       make clean-env   : Removes all environment files."
	@echo "       make clean-build : Remove all build files."
	@echo "       make clean       : Remove both."
	@echo ""
	@echo "Run commands:"
	@echo "       make build : Runs the build project process"
	@echo "       make tests : Runs the unit tests"
	@echo "       make docs  : Run the generating docs process"
	@echo ""
	@echo "Publish commands:"
	@echo "       make publish-release : Publish the library into the Pypi's release repo "
	@echo "       make publish-testing : Publish the library into the Pypi's testing repo"


# -- Requirements Check --
check:
	@echo "🟡 Check Requirements ..."

	@for f in $(REQUIREMENTS); do \
		type $$f 2> /dev/null 1> /dev/null || { echo "$$f is missing ❌"; exit -1; }; \
	done
	
	@echo "Requirements Check Passed ✅"
	@echo ""


# -- Install Section --
install:
	@echo "🟡 Installing library ..."

	# TODO: install requirements ?
	$(SYSTEM_PYTHON) -m pip install -e . 

	@echo "Library installed ✅"
	@echo ""

install-dev: check	
	@echo "🟡 Installing dev dependencies ..."

	@virtualenv $(VENV)
	@echo "Created env '$(VENV)' ✅"
	@echo ""
	
	@$(VENV_PYTHON) -m pip install -r utils/requirements.txt 
	@echo "dev dependencies installed ✅"
	@echo ""

install-doc:
	"🟡 Installing doc dependencies ... "
	@$(VENV_PYTHON) -m pip install -r utils/requirements-doc.txt # TODO: creare
	@echo "doc dependencies installed ✅"
	@echo ""

install-test:
	@echo "🟡 Installing testing dependencies ..."
	@$(VENV_PYTHON) -m pip install -r utils/requirements-test.txt # TODO: creare
	@echo "test dependencies installed ✅"
	@echo ""


# -- Clean Section --
clean-build:
	@echo "🟡 Cleaning build files ..."
	# TODO
	
	@echo "Build files cleaned ✅"

clean-env: $(VENV)
	@echo "🟡 Cleaning env files ..."
	rm -rf $(VENV)
	
	@echo "Removed env $(VENV) ✅"

# clean all !
clean: clean-env clean-build

# -- Runs Section
tests:
	@echo "🟡 Running unit tests..."

	$(VENV_PYTHON) -m unittest discover -v -s tests/

docs:
	# TODO
	@echo "Generate docs from code"

build:
	# TODO
	@echo "BUILD"


# -- Publish Section --
publish-release:
	# TODO
	@echo "RELEASE"

publish-testing:
	# TODO
	@echo "TESTING"

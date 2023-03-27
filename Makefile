SHELL:=/bin/bash 

VENV           = venv_snlp
VENV_PYTHON    = $(VENV)/bin/python
SYSTEM_PYTHON  = $(or $(shell which python3), $(shell which python))
# If virtualenv exists, use it. If not, find python using PATH
PYTHON         = $(or $(wildcard $(VENV_PYTHON)), $(SYSTEM_PYTHON))

REQUIREMENTS	=	gh
REQUIREMENTS	+=	git
REQUIREMENTS	+=	virtualenv

SRC = src
# W,E (ignore warning end errors). W (only warnings)
CODE_IGNORE_LEVEL = ""
DOCS_DIR = docs
# --html, --pdf or blank for markdown
DOCS_FORMAT = "--html"
DOCSTRINGS_FORMAT = numpydoc
LINT_FORMAT = pylint


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
	@echo "       make clean-env        : Removes all environment files."
	@echo "       make clean-build      : Remove all build files."
	@echo "       make clean-docstrings : Remove all docstrings (.patch) files."
	@echo "       make clean-docs       : Remove all documentation files."
	@echo "       make clean            : Runs all clean commands."
	@echo ""
	@echo "Auto-generation commands:"
	@echo "   code:"
	@echo "       make code-format : Format all python files (using yapf)."
	@echo "       make code-check  : Checks for warnings and errors. (using pylama linter)."
	@echo "       make code        : Runs -format and -check."
	@echo ""
	@echo "   documentation:"
	@echo "       make docstrings     : Generate docstring for all python files (need to manually apply patches)."
	@echo "       make documentation  : Generate documentation from python docstrings."
	@echo ""
	@echo "   make build : Runs the build project process"
	@echo "   make tests : Runs the unit tests"
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

	virtualenv $(VENV)
	@echo "Created env '$(VENV)' ✅"
	@echo ""
	
	$(VENV_PYTHON) -m pip install -r utils/requirements.txt 
	@echo "dev dependencies installed ✅"
	@echo ""

install-doc:
	"🟡 Installing doc dependencies ... "
	$(VENV_PYTHON) -m pip install -r utils/requirements-doc.txt
	@echo "doc dependencies installed ✅"
	@echo ""

install-test:
	@echo "🟡 Installing testing dependencies ..."
	$(VENV_PYTHON) -m pip install -r utils/requirements-test.txt # TODO: creare
	@echo "test dependencies installed ✅"
	@echo ""


# -- Clean Section --
clean-docs: $(DOCS_DIR)
	@echo "🟡 Cleaning documentation files ..."
	rm -rf $(DOCS_DIR)/*
	@echo "Documentation files cleaned ✅"

clean-build:
	@echo "🟡 Cleaning build files ..."
	# TODO
	
	@echo "Buildfiles cleaned ✅"

clean-env: $(VENV)
	@echo "🟡 Cleaning env files ..."
	rm -rf $(VENV)
	
	@echo "Removed env $(VENV) ✅"

clean-docstrings: *.patch
	@echo "🟡 Cleaning docstrings (.patch) files ..."
	rm -rf $^
	@echo "docstrings files cleaned ✅"

# clean all !
clean: clean-env clean-build clean-docs clean-docstrings


# -- Runs Section
tests:
	@echo "🟡 Running unit tests..."

	$(VENV_PYTHON) -m unittest discover -v -s tests/


# Code cheking & autoformatting
# NOTA:
# per cambiare la variabile CODE_IGNORE_LEVEL
# anteporre questo al target dove si vuole cambiare
#	target: CODE_IGNORE_LEVEL="W"
code-check:
	@echo "🟡 Checking errors and warnings ..."
	pylama -f $(LINT_FORMAT) -i $(CODE_IGNORE_LEVEL) $(SRC)
	@echo "No problems found ✅"

code-format:
	@echo "🟡 Refactoring code ..."
	yapf --recursive -i $(SRC)
	@echo "Done ✅"

code: code-format code-check


# Documentation
# TODO: una gestione interattiva di quale parte della patch
#	    applicare ?
docstrings: 
	@echo "🟡 Generating docstrings ..."
	pyment -o $(DOCSTRINGS_FORMAT) $(SRC)
	@echo "Done ✅"
	@echo ""
	@echo "You have to manually apply patches."
	@echo "Try with:"
	@echo "    git apply filename.py.patch"
	@echo "or with:"
	@echo "    patch -p1 < filename.py.patch"
	
# docstrings: $(shell pyment -o numpydoc $(SRC))
# 	for p in $(shell ls *.patch) ; do \
# 		editdiff $$p && git apply $$p; \
# 	done;
# 	rm *.patch

documentation:
	@echo "🟡 Generating docs from code ..."
	mkdir -p $(DOCS_DIR)
	pdoc $(DOCS_FORMAT) -o $(DOCS_DIR) $(SRC)
	@echo "Done ✅"


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

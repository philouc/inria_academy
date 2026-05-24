# Minimal Makefile for Sphinx documentation
# -----------------------------------------
#
# Usage:
#   make html         # build the HTML site into _build/html
#   make livehtml     # auto-reload server (requires sphinx-autobuild)
#   make clean        # remove the _build directory
#   make linkcheck    # validate all external links
#

SPHINXOPTS    ?=
SPHINXBUILD   ?= uv run sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html clean livehtml linkcheck

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo
	@echo "Build finished. HTML pages are in $(BUILDDIR)/html."

clean:
	rm -rf "$(BUILDDIR)" generated api/generated

linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

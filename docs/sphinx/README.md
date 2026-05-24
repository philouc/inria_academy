# Sphinx documentation for `inria_academy`

This directory contains the Sphinx documentation source.

## Quick build

```bash
# (one-time) install the docs dependencies
uv pip install -r requirements.txt

# build the HTML site
cd docs/sphinx
make html

# open the result
xdg-open _build/html/index.html        # Linux
open _build/html/index.html            # macOS
```

## Live preview

```bash
uv pip install sphinx-autobuild
make livehtml
```

This starts a local server at <http://localhost:8000> that auto-rebuilds on
every save.

## Layout

```
docs/sphinx/
├── conf.py              Sphinx configuration
├── index.rst            Landing page
├── installation.rst     Install & build instructions
├── usage.rst            How to run the demo scripts
├── reference.rst        Annotated example gallery
├── changelog.rst
├── license.rst
├── api/                 API reference (autodoc)
│   ├── index.rst
│   ├── mvmd.rst
│   ├── signals.rst
│   ├── swt_emd.rst
│   └── k_selection.rst
├── _static/             Custom CSS / assets
│   └── custom.css
├── _templates/          Template overrides (currently empty)
├── Makefile             GNU make targets
├── make.bat             Windows equivalent
└── requirements.txt     Doc-build dependencies
```

## Notes on `autodoc_mock_imports`

`PyEMD`, `ssqueezepy`, `vmdpy`, `neurokit2`, and `pymultifracs` are mocked
in `conf.py` so the documentation builds without installing those heavier
runtime dependencies. If you add a new optional dependency that is not in
the doc-build environment, extend the `autodoc_mock_imports` list in
`conf.py`.

## Coexistence with MkDocs

The project also ships an MkDocs site (`mkdocs.yml` + `docs/index.md`,
`docs/usage.md`, `docs/reference.md`). The two are independent — Sphinx
sources live under `docs/sphinx/` and write their output to
`docs/sphinx/_build/html/`. Pick the one you want to publish; both can
coexist in the same repository indefinitely.

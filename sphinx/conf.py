# Configuration file for the Sphinx documentation builder.
#
# Inria Academy — Advanced Signal Processing Tools.
#
# Full documentation: https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# Make the package importable so autodoc can find it.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent  # docs/sphinx -> docs -> project root
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# -- Project information -----------------------------------------------------
project = "Inria Academy"
author = "Philippe Ciuciu"
copyright = f"{datetime.now().year}, {author} — Inria, NeuroSpin"

# Read version dynamically from the installed package.
try:
    from inria_academy import __version__ as release  # type: ignore
except Exception:  # pragma: no cover
    release = "0.1.0"
version = ".".join(release.split(".")[:2])


# -- General configuration ---------------------------------------------------
extensions = [
    # Core autodoc machinery
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",        # NumPy-style and Google-style docstrings
    "sphinx.ext.viewcode",        # add [source] links to each documented object
    "sphinx.ext.intersphinx",     # cross-reference numpy / scipy / matplotlib
    "sphinx.ext.mathjax",         # render LaTeX in docstrings
    "sphinx.ext.todo",
    # Third-party
    "myst_parser",                # allow Markdown source files
    "sphinx_copybutton",          # copy-to-clipboard buttons on code blocks
    "sphinx_design",              # cards, grids, tabs
]

# Source files: both reStructuredText and Markdown (via MyST).
source_suffix = {
    ".rst": "restructuredtext",
    ".md":  "markdown",
}

# Top-level table of contents document.
master_doc = "index"
root_doc = "index"

# Templates / patterns to ignore.
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

# Suppress warnings that are noisy on first build.
nitpicky = False
suppress_warnings = ["myst.header"]


# -- Autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members":           True,
    "undoc-members":     False,
    "show-inheritance":  True,
    "member-order":      "bysource",
}
autodoc_typehints = "description"          # render type hints in the description
autodoc_typehints_format = "short"
autoclass_content = "class"                # use the class docstring (not __init__)

# Heavy / optional runtime dependencies that should NOT block the doc build.
# autodoc imports the modules to inspect them; the mocks let import succeed
# even if these packages are not installed in the doc-build environment.
autodoc_mock_imports = [
    "PyEMD",
    "ssqueezepy",
    "vmdpy",
    "neurokit2",
    "pymultifracs",
]


# -- Napoleon (NumPy / Google docstring) -------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True


# -- MyST (Markdown) ---------------------------------------------------------
myst_enable_extensions = [
    "amsmath",          # inline / display math
    "dollarmath",       # $...$ and $$...$$ math
    "colon_fence",      # ::: admonition fences
    "deflist",          # definition lists
    "tasklist",         # GitHub-style task lists
    "fieldlist",
    "substitution",
]
myst_heading_anchors = 3                   # generate anchors for h1..h3


# -- Intersphinx -------------------------------------------------------------
# Cross-link to upstream API docs for types referenced in docstrings.
intersphinx_mapping = {
    "python":     ("https://docs.python.org/3", None),
    "numpy":      ("https://numpy.org/doc/stable/", None),
    "scipy":      ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}


# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = "Inria Academy"
html_short_title = "Inria Academy"
html_static_path = ["_static"]
# A custom stylesheet (kept minimal — just brand accents).
html_css_files = ["custom.css"]

# Furo-specific theme options.
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/philouc/inria_academy",
    "source_branch": "main",
    "source_directory": "docs/sphinx/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/philouc/inria_academy",
            "html": "",
            "class": "fa-brands fa-github",
        },
    ],
    "light_css_variables": {
        "color-brand-primary":  "#3f51b5",   # Inria-ish indigo
        "color-brand-content":  "#3f51b5",
    },
    "dark_css_variables": {
        "color-brand-primary":  "#8e99f3",
        "color-brand-content":  "#8e99f3",
    },
}

# Show the "Edit this page" link only if the source dir matches the layout.
html_show_sourcelink = True
html_copy_source = True

# Favicon / logo placeholders — uncomment once assets exist.
# html_logo  = "_static/logo.svg"
# html_favicon = "_static/favicon.ico"


# -- Copybutton --------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False


# -- todo --------------------------------------------------------------------
todo_include_todos = False

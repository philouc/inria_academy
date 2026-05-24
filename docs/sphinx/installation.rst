Installation
============

Prerequisites
-------------

* Python ≥ 3.11
* `uv <https://docs.astral.sh/uv/>`_ — fast Python package and environment
  manager

Install ``uv`` once on your machine:

.. tab-set::

   .. tab-item:: macOS / Linux

      .. code-block:: bash

         curl -LsSf https://astral.sh/uv/install.sh | sh

   .. tab-item:: Windows (PowerShell)

      .. code-block:: powershell

         irm https://astral.sh/uv/install.ps1 | iex


Installing the package
----------------------

Clone the repository and synchronise the environment:

.. code-block:: bash

   git clone https://github.com/philouc/inria_academy.git
   cd inria_academy
   uv sync

This creates a ``.venv/`` and installs every dependency at the exact version
pinned in ``uv.lock``.

To also install the development tools (``mkdocs``, ``sphinx``, ``ruff``,
``pytest`` …):

.. code-block:: bash

   uv sync --group dev


Building this documentation
---------------------------

The documentation is built with `Sphinx <https://www.sphinx-doc.org/>`_, the
`furo <https://pradyunsg.me/furo/>`_ theme, and a few extensions:

* ``sphinx.ext.autodoc`` + ``sphinx.ext.napoleon`` — pull NumPy-style
  docstrings out of the source;
* ``sphinx.ext.autosummary`` — generate per-module summary tables;
* ``myst_parser`` — let Sphinx render Markdown sources alongside reST;
* ``sphinx_copybutton``, ``sphinx_design`` — UX niceties.

Install the documentation tooling (only needed once):

.. code-block:: bash

   uv pip install sphinx furo myst-parser sphinx-copybutton sphinx-design

Then build the HTML site:

.. code-block:: bash

   cd docs/sphinx
   sphinx-build -b html . _build/html

Open ``docs/sphinx/_build/html/index.html`` in your browser.

For a live-reloading server during writing:

.. code-block:: bash

   uv pip install sphinx-autobuild
   sphinx-autobuild docs/sphinx docs/sphinx/_build/html


Verifying the installation
--------------------------

A quick sanity check that the package is importable:

.. code-block:: python

   >>> import inria_academy
   >>> inria_academy.__version__
   '0.1.0'
   >>> from inria_academy.utils import MVMD, signals, swt_emd

If any of the heavier scientific dependencies (``ssqueezepy``, ``PyEMD``,
``vmdpy``) fails to import here, re-run ``uv sync`` — they are all pinned in
``uv.lock`` and should install cleanly.

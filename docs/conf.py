"""Config."""  # noqa: INP001
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Find the path to the source files we want to document, relative to the location of this file,
# convert it to an absolute path.
py_path = os.path.join(os.getcwd(), os.path.dirname(__file__), '../')
sys.path.insert(0, os.path.abspath(py_path))

project = "QCut"
copyright = "2024, Joonas Nivala"
author = "Joonas Nivala"

# The short X.Y version.
version = ''
# The full version, including alpha/beta/rc tags.
release = ''
try:
    from QCut import __version__ as version
except ImportError:
    pass
else:
    release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "myst_nb",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "jupyter_execute",
    "build",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.md': 'myst-nb',
}

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

nb_execution_mode = "off"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_logo = "_static/images/QCut-logo.jpg"
html_title = "QCut"
html_favicon = "_static/images/QCut-logo.jpg"
html_static_path = ["_static"]

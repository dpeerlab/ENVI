# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path
import datetime
import tomli

sys.path.insert(0, os.path.abspath("../../src"))

this_directory = Path(__file__).parent

# Get current year for copyright
current_year = datetime.datetime.now().year

# Get package version from pyproject.toml
try:
    pyproject_path = Path(__file__).parents[2] / "pyproject.toml"
    if pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomli.load(f)
            package_version = pyproject_data.get("tool", {}).get("poetry", {}).get("version", "0.1.0")
    else:
        package_version = "0.1.0"
except Exception as e:
    print(f"Warning: Could not read version from pyproject.toml: {e}")
    package_version = "0.1.0"

# -- Project information -----------------------------------------------------

project = "scenvi"
copyright = f"{current_year}, Doron Haviv"
author = 'Doron Haviv'

# The full version, including alpha/beta/rc tags
release = package_version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "nbsphinx",
    "sphinx.ext.napoleon",
    "sphinx_mdinclude",
]
if os.environ.get('READTHEDOCS') == 'True':
    extensions.append("sphinx_github_style")

source_suffix = [".rst", ".md"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# GitHub repository info
github_user = "dpeerlab"  # Replace with your actual GitHub username
github_repo = "scenvi"      # Replace with your actual repository name 
github_version = "main"     # Or your default branch

# For sphinx-github-style extension when on ReadTheDocs
if os.environ.get('READTHEDOCS') == 'True':
    linkcode_url = f"https://github.com/{github_user}/{github_repo}/blob/{github_version}"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
pygments_style = "tango"

highlight_language = "none"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
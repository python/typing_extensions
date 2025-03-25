# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os.path
import sys

from docutils.nodes import Element
from sphinx.writers.html5 import HTML5Translator

sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'typing_extensions'
copyright = '2023, Guido van Rossum and others'
author = 'Guido van Rossum and others'
release = '4.6.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.intersphinx', '_extensions.gh_link']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

intersphinx_mapping = {'py': ('https://docs.python.org/3', None)}

add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'


class MyTranslator(HTML5Translator):
    """Adds a link target to name without `typing_extensions.` prefix."""
    def visit_desc_signature(self, node: Element) -> None:
        desc_name = node.get("fullname")
        if desc_name:
            self.body.append(f'<span id="{desc_name}"></span>')
        super().visit_desc_signature(node)


def setup(app):
    app.set_translator('html', MyTranslator)

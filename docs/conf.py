# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Data-Driven Interatomic Potential WorkChain'
copyright = '2024, Maria Peressi, Antimo Marrazzi, Davide Bidoggi, Nataliia Manko'
author = 'Maria Peressi, Antimo Marrazzi, Davide Bidoggi, Nataliia Manko'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_autodoc_typehints',
    'sphinx.ext.todo',  # To handle TODO notes
    'myst_parser',  # To enable Markdown support   
    'autoapi.extension',  # For automatic API documentation
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Todo configuration
todo_include_todos = True

# Myst Parser (Markdown support) configuration
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "html_image",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# AutoAPI configuration
autoapi_type = 'python'
autoapi_dirs = ['../../aiida_scripts']

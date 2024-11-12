# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'hazardous'
copyright = '2023, Olivier Grisel and Vincent Maladière'
author = 'Olivier Grisel and Vincent Maladière'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_css_files = [
    "css/custom.css",
]
html_js_files = []

# sphinx_gallery options
sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to example scripts
    "gallery_dirs": "auto_examples",  # path to gallery generated output
    "within_subsection_order": "FileNameSortKey",  # See https://sphinx-gallery.github.io/stable/configuration.html#sorting-gallery-examples for alternatives
    "show_memory": False,
    "write_computation_times": False,
    'reference_url': {
        # Intersphinx links
        # The module you locally document uses None
        'hazardous': None,
        }

}

# intersphinx configuration
intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

numpydoc_show_class_members = False 

html_title = "hazardous"

html_theme_options = {
    "announcement": (
        "https://raw.githubusercontent.com/soda-inria/hazardous/main/doc/announcement.html"
    ),
        "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/soda-inria/hazardous/",
            "icon": "fa-brands fa-github",
        },
    ],

}

# Sphinx-Copybutton configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

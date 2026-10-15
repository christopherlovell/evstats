project = "evstats"
author = "Chris Lovell"
copyright = "2023, Chris Lovell"

extensions = ["myst_nb", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]
exclude_patterns = ["_build"]
myst_enable_extensions = ["dollarmath"]

# Run the notebooks on every build, so the docs fail if the examples break
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 600

autodoc_member_order = "bysource"

html_theme = "shibuya"
html_title = "evstats"
html_theme_options = {
    "github_url": "https://github.com/christopherlovell/evstats",
    "show_ai_links": False,
}

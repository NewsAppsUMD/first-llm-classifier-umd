"""Settings for Sphinx."""

from datetime import datetime
from typing import Any

project = "First LLM Classifier UMD"
year = datetime.now().year
copyright = f"{year} Derek Willis and Ben Welsh"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = "sphinx"

html_sidebars: dict[Any, Any] = {
    "**": [
        "about.html",
        "navigation.html",
    ]
}

extensions = [
    "myst_parser",
]

myst_enable_extensions = [
    "attrs_block",
    "colon_fence",
]

html_static_path = ["_static"]

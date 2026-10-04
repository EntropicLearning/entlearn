"""Check that the site's Markdown configuration preserves and marks up TeX."""

from pathlib import Path

import markdown
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def render():
    # mkdocs.yml names a Python function (the mermaid fence), which safe_load rejects.
    config = yaml.load((ROOT / "mkdocs.yml").read_text(), Loader=yaml.Loader)
    extensions, options = [], {}
    for item in config["markdown_extensions"]:
        if isinstance(item, str):
            extensions.append(item)
        else:
            for name, settings in item.items():
                extensions.append(name)
                options[name] = settings or {}

    def convert(text):
        return markdown.markdown(text, extensions=extensions, extension_configs=options)

    return convert


class TestDocumentationMath:
    @pytest.mark.parametrize("source", [r"$x \in \mathbb{R}^D$", r"\(x \in \mathbb{R}^D\)"])
    def test_inline_math_is_marked_for_the_renderer(self, render, source):
        html = render(source)
        assert '<span class="arithmatex">' in html
        assert r"\(x \in \mathbb{R}^D\)" in html

    @pytest.mark.parametrize(
        "source", [r"$$" + "\n" + r"\sum_{i=1}^{T} x_i" + "\n$$", r"\[\sum_{i=1}^{T} x_i\]"]
    )
    def test_display_math_is_marked_for_the_renderer(self, render, source):
        html = render(source)
        assert '<div class="arithmatex">' in html
        assert r"\[\sum_{i=1}^{T} x_i\]" in html.replace("\n", "")

    def test_code_examples_are_not_typeset(self, render):
        html = render("`$x$`\n\n```text\n$$x$$\n```")
        assert "arithmatex" not in html
        assert "$x$" in html

    def test_current_introduction_contains_math_markup(self, render):
        source = (ROOT / "docs_site/guide/introduction.md").read_text()
        html = render(source)
        assert r'<span class="arithmatex">\(x_t \in \mathbb{R}^D\)</span>' in html

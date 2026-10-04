"""Test the published static-site path, with Python running only in browser WASM.

Requires installed Chrome, Playwright and network access to the marimo/Pyodide CDN.
Run with: uv run --with playwright pytest tests/browser/test_intro_explorer_browser.py
"""

import functools
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

playwright_api = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def static_docs(tmp_path_factory):
    root = tmp_path_factory.mktemp("intro-site")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "--site-dir",
            str(root / "entlearn"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        functools.partial(SimpleHTTPRequestHandler, directory=str(root)),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/entlearn/guide/introduction/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class TestIntroductionIsland:
    def test_sliders_react_on_static_site_without_a_python_server(self, static_docs):
        with playwright_api.sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(channel="chrome", headless=True)
            except playwright_api.Error as error:
                pytest.skip(f"Installed Chrome is required: {error}")
            page = browser.new_page(viewport={"width": 1100, "height": 1000})
            errors, sockets = [], []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("websocket", lambda socket: sockets.append(socket.url))
            try:
                page.goto(static_docs, wait_until="networkidle", timeout=90000)
                equations = page.locator(".md-content .arithmatex")
                assert equations.count() > 0
                for equation in equations.all():
                    playwright_api.expect(equation.locator("mjx-container")).to_have_count(
                        1, timeout=30000
                    )
                    playwright_api.expect(equation.locator("mjx-container")).to_be_visible(
                        timeout=30000
                    )
                playwright_api.expect(page.locator("mjx-merror")).to_have_count(0)
                chart = page.locator("svg[data-feature-x]")
                chart.wait_for(state="visible", timeout=120000)
                sliders = page.get_by_role("slider")
                playwright_api.expect(sliders).to_have_count(2)
                original = chart.locator("g circle").evaluate_all(
                    "els => els.map(e => [e.getAttribute('cx'), e.getAttribute('cy')])"
                )
                assert len(original) == 165
                for slider in (sliders.nth(0), sliders.nth(1)):
                    playwright_api.expect(slider).to_have_attribute("aria-valuemin", "0")
                    playwright_api.expect(slider).to_have_attribute("aria-valuemax", "99")
                sliders.nth(0).focus()
                sliders.nth(0).press("Home")
                for _ in range(19):
                    sliders.nth(0).press("ArrowRight")
                sliders.nth(1).focus()
                sliders.nth(1).press("End")
                for _ in range(27):
                    sliders.nth(1).press("ArrowLeft")
                playwright_api.expect(chart).to_have_attribute("data-feature-x", "19")
                playwright_api.expect(chart).to_have_attribute("data-feature-y", "72")
                assert (
                    chart.locator("g circle").evaluate_all(
                        "els => els.map(e => [e.getAttribute('cx'), e.getAttribute('cy')])"
                    )
                    != original
                )
                playwright_api.expect(
                    page.locator(".feature-explorer marimo-island[data-reactive='false']")
                ).to_be_hidden()
                horizontal_label = page.get_by_text("Horizontal feature: 19", exact=True)
                vertical_label = page.get_by_text("Vertical feature: 72", exact=True)
                playwright_api.expect(horizontal_label).to_be_visible()
                playwright_api.expect(vertical_label).to_be_visible()
                label_colour = horizontal_label.evaluate("el => getComputedStyle(el).color")
                page.get_by_title("Switch to dark mode", exact=True).click()
                playwright_api.expect(page.locator("body")).to_have_attribute(
                    "data-md-color-scheme", "slate"
                )
                assert horizontal_label.evaluate("el => getComputedStyle(el).color") == label_colour
                for width in (1100, 390):
                    page.set_viewport_size({"width": width, "height": 1000})
                    bounds = chart.bounding_box()
                    assert bounds is not None
                    assert bounds["width"] == pytest.approx(bounds["height"], abs=1)
                    assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
                sliders.nth(0).press("Home")
                sliders.nth(1).press("Home")
                sliders.nth(1).press("ArrowRight")
                playwright_api.expect(chart).to_have_attribute("data-feature-x", "0")
                playwright_api.expect(chart).to_have_attribute("data-feature-y", "1")
                assert (
                    chart.locator("g circle").evaluate_all(
                        "els => els.map(e => [e.getAttribute('cx'), e.getAttribute('cy')])"
                    )
                    == original
                )
                assert not errors
                assert not sockets
            finally:
                browser.close()

"""
Playwright-based browser tool for the Frontend Developer and QA Engineer agents.

Used to visually verify rendered pages, capture screenshots, and detect
console errors without spinning up a full display server (headless mode).
"""

import base64

from langchain_core.tools import tool
from playwright.sync_api import sync_playwright


@tool
def render_and_screenshot(
    url: str,
    wait_selector: str | None = None,
) -> dict:
    """
    Load a URL in a headless Chromium browser, optionally wait for a CSS
    selector to appear, and return a base64-encoded full-page screenshot
    along with any console errors emitted by the page.

    Args:
        url: The URL to navigate to (e.g. 'http://localhost:3000').
        wait_selector: Optional CSS selector to wait for before screenshotting.

    Returns:
        dict with keys:
            screenshot_b64 (str): Base64-encoded PNG screenshot.
            console_errors (list[str]): Console error messages from the page.
    """
    console_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.on(
            "console",
            lambda msg: console_errors.append(msg.text)
            if msg.type == "error"
            else None,
        )
        page.goto(url, timeout=15_000)
        if wait_selector:
            page.wait_for_selector(wait_selector, timeout=10_000)
        screenshot_bytes = page.screenshot(full_page=True)
        browser.close()

    return {
        "screenshot_b64": base64.b64encode(screenshot_bytes).decode(),
        "console_errors": console_errors,
    }

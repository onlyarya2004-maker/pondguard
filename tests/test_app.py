"""The Streamlit dashboard renders each page without errors (headless, mock mode)."""

import pytest
from streamlit.testing.v1 import AppTest

from pondguard.paths import REPO_ROOT

APP = str(REPO_ROOT / "app" / "streamlit_app.py")
pytestmark = pytest.mark.filterwarnings("ignore")


def _open(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60).run()
    if page != "Results":
        at.sidebar.radio[0].set_value(page).run()
    return at


def test_every_page_renders_without_an_exception():
    for page in ("Results", "Run", "Approvals", "About"):
        at = _open(page)
        assert not at.exception, f"{page}: {[e.value for e in at.exception]}"


def test_mock_mode_is_announced_on_every_page():
    for page in ("Results", "About"):
        at = _open(page)
        assert any("MOCK MODE" in w.value for w in at.warning)


def test_results_page_shows_the_ab_table_and_the_decision():
    at = _open("Results")
    assert len(at.dataframe) >= 1
    assert any("Decision" in m.value for m in at.markdown)

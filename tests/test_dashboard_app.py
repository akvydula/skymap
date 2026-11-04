"""Streamlit AppTest smoke test of the dashboard on the Jan-16 test observation."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from skymap.dashboard import discovery  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "src" / "skymap" / "dashboard" / "app.py"
DATA = discovery.DEFAULT_DATA_ROOT / discovery.DEFAULT_SESSION / "310" / "GBTOBS_2026-01-16T112750.hdf5"

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="Jan-16 test data not available")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    return at


def test_app_loads_with_default_selection(app):
    assert not app.exception, [e.value for e in app.exception]
    labels = {s.label for s in app.sidebar.selectbox}
    assert {"Session", "Calibration file", "Pointing directory", "Receiver data (spectrometer)",
            "Cross-correlation", "Source"} <= labels
    corr = next(s for s in app.sidebar.selectbox if s.label == "Cross-correlation")
    assert corr.value == "AB*"
    assert corr.options[-1] == "ALL" and "AA*" in corr.options


@pytest.mark.parametrize(
    "view",
    ["Calibration", "Single scan", "Peak scans (az/el)", "Pointing & timing offsets", "HEALPix", "Beam fit"],
)
def test_each_view_renders_without_error(app, view):
    app.radio(key="view").set_value(view).run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]


def test_all_correlations_peak_scans(app):
    corr = next(s for s in app.sidebar.selectbox if s.label == "Cross-correlation")
    corr.set_value("ALL")
    app.radio(key="view").set_value("Peak scans (az/el)").run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]

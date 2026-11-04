"""Tests for the synthetic-session generator (skymap.dashboard.make_test_data)."""

import h5py
import numpy as np
import pytest

from skymap.dashboard import discovery, make_test_data
from tests.dashboard_helpers import make_raw_file, make_session


@pytest.fixture()
def src(tmp_path):
    return make_session(tmp_path, "TGBT25B_614_01")


def test_scales_spectra_and_copies_layout(tmp_path, src):
    dst = tmp_path / "TGBT25B_614_77"
    out = make_test_data.make_session(
        src, dst, scale=1.2, offset=0.5, files=["GBTOBS_2026-01-16T112750.hdf5"], log=lambda *_: None
    )
    assert [p.name for p in out] == ["GBTOBS_2026-01-16T112750.hdf5"]
    with h5py.File(src / "310" / out[0].name) as a, h5py.File(out[0]) as b:
        np.testing.assert_allclose(b["data/spec"][:], 1.2 * a["data/spec"][:] + 0.5, rtol=1e-5)
        np.testing.assert_array_equal(b["data/time"][:], a["data/time"][:])
        np.testing.assert_array_equal(b["data/freq"][:], a["data/freq"][:])
        assert b["data/spec"].attrs["synthetic_scale"] == pytest.approx(1.2)

    # discovered like a real session: cal, raw data and pointing directory
    assert [p.name for p in discovery.find_cal_files(dst)] == ["cal_file_2026_01_01.hdf5"]
    assert [p.name for p in discovery.find_data_files(dst)] == ["GBTOBS_2026-01-16T112750.hdf5"]
    assert len(discovery.find_pointing_dirs(dst)) == 1
    assert dst.name in [s.name for s in discovery.find_sessions(tmp_path)]
    # the source is untouched
    with h5py.File(src / "310" / out[0].name) as a:
        assert not a.attrs.get("synthetic")


def test_offset_fraction_is_per_correlation_and_constant_across_files(tmp_path, src):
    # the second raw file has different statistics; c must still come from the first file
    make_raw_file(src / "310" / "GBTOBS_2026-01-16T135150.hdf5", seed=5)
    dst = tmp_path / "TGBT25B_614_78"
    make_test_data.make_session(src, dst, scale=1.0, offset_frac=0.1, log=lambda *_: None)
    offs = []
    for name in ("GBTOBS_2026-01-16T112750.hdf5", "GBTOBS_2026-01-16T135150.hdf5"):
        with h5py.File(dst / "310" / name) as f:
            offs.append(f["data/spec"].attrs["synthetic_offset"])
    assert offs[0].shape == (10,)
    np.testing.assert_array_equal(offs[0], offs[1])
    assert (offs[0] > 0).all()


def test_refuses_to_overwrite_without_force(tmp_path, src):
    dst = tmp_path / "TGBT25B_614_79"
    make_test_data.make_session(src, dst, log=lambda *_: None)
    with pytest.raises(FileExistsError):
        make_test_data.make_session(src, dst, log=lambda *_: None)
    make_test_data.make_session(src, dst, force=True, log=lambda *_: None)


def test_cli_rejects_bad_session_name(tmp_path):
    with pytest.raises(SystemExit):
        make_test_data.main(["--data-root", str(tmp_path), "--session", "not a session"])

"""Tests for dashboard file discovery."""

from datetime import datetime

import pytest

from skymap.dashboard import discovery as d
from tests.dashboard_helpers import make_session


def test_pol_attributes_and_labels():
    assert d.pol_attributes("AB*") == ["AB_"]
    assert d.pol_attributes("ALL") == [
        "AA_", "BB_", "CC_", "DD_", "AB_", "BC_", "CD_", "AC_", "BD_", "AD_"
    ]
    assert d.CORRELATION_CHOICES[:3] == ["AA*", "BB*", "CC*"] and d.CORRELATION_CHOICES[-1] == "ALL"
    assert d.pol_label("AD_") == "AD*"
    with pytest.raises(ValueError):
        d.pol_attributes("ZZ*")


def test_find_sessions_matches_pattern_only(tmp_path):
    make_session(tmp_path, "TGBT25B_614_02")
    make_session(tmp_path, "TGBT25B_614_01")
    (tmp_path / "notes").mkdir()
    (tmp_path / "random_folder").mkdir()
    (tmp_path / "TGBT25B_614_03.txt").touch()
    names = [p.name for p in d.find_sessions(tmp_path)]
    assert names == ["TGBT25B_614_01", "TGBT25B_614_02"]
    assert d.find_sessions(tmp_path / "missing") == []


def test_classify_and_list_files(tmp_path):
    session = make_session(tmp_path)
    cal = d.find_cal_files(session)
    assert [p.name for p in cal] == ["cal_file_2026_01_01.hdf5"]
    raw = d.find_data_files(session)
    assert [p.name for p in raw] == ["GBTOBS_2026-01-16T112750.hdf5"]
    with_derived = d.find_data_files(session, include_derived=True)
    assert [p.name for p in with_derived] == [
        "GBTOBS_2026-01-16T112750.hdf5",
        "beam_obs_2026_01_16.hdf5",
    ]
    assert d.classify_hdf5(session / "310" / "beam_obs_2026_01_16.hdf5") == "derived"


@pytest.mark.parametrize("folder", ["Antenna", "ANTENNA", "antenna"])
def test_pointing_dir_case_insensitive(tmp_path, folder):
    session = make_session(tmp_path, antenna_dir=folder)
    dirs = d.find_pointing_dirs(session)
    assert len(dirs) == 1 and dirs[0].name == folder
    cov = d.pointing_coverage(dirs[0])
    assert cov.n_files == 2
    assert cov.first == datetime(2026, 1, 16, 18, 4, 7)
    assert cov.last == datetime(2026, 1, 16, 18, 5, 14)


def test_pointing_coverage_empty(tmp_path):
    (tmp_path / "Antenna").mkdir()
    cov = d.pointing_coverage(tmp_path / "Antenna")
    assert cov.n_files == 0 and cov.first is None

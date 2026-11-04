"""Command-line launcher: ``skymap-dashboard`` / ``python -m skymap.dashboard``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="skymap-dashboard",
        description="Launch the skymap beam dashboard. Extra arguments are passed to `streamlit run` "
        "(e.g. --server.port 8600).",
    )
    ap.add_argument("--data-root", type=Path, help="folder containing the session folders (default: ./data_310)")
    ap.add_argument("--port", type=int, help="server port (default: Streamlit's, 8501)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args, passthrough = ap.parse_known_args(argv)

    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print('streamlit is not installed. Run: pip install -e ".[dashboard]"', file=sys.stderr)
        return 1

    if args.data_root is not None:
        os.environ["SKYMAP_DATA_ROOT"] = str(args.data_root.expanduser().resolve())

    app = Path(__file__).with_name("app.py")
    cmd = ["streamlit", "run", str(app)]
    if args.port:
        cmd += ["--server.port", str(args.port)]
    if args.no_browser:
        cmd += ["--server.headless", "true"]
    cmd += passthrough
    sys.argv = cmd
    return stcli.main()


if __name__ == "__main__":
    sys.exit(main())

"""`python -m so101` (and ./run) open the SO-101 app: one window with the live
cameras, demo recording, and settings/tools in the menu bar.

`./run policy` instead runs your latest trained policy (sim by default; --real for the
arm). Anything else just opens the app."""

import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "policy":
        from .run_policy import run_from_launcher
        run_from_launcher(sys.argv[2:])
    else:
        from .record_ui import main as app_main
        app_main()


if __name__ == "__main__":
    main()

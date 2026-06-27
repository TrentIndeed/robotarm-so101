"""`python -m so101` (and ./run) open the SO-101 app: one window with the live
cameras, demo recording, and settings/tools in the menu bar."""


def main() -> None:
    from .record_ui import main as app_main
    app_main()


if __name__ == "__main__":
    main()

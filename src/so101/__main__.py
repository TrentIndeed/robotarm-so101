"""`python -m so101` opens the graphical launcher (falls back to the text menu)."""


def main() -> None:
    try:
        from .gui import main as gui_main
    except Exception as exc:  # Tkinter missing/unavailable -> text menu
        print(f"GUI unavailable ({exc}); falling back to the text menu.\n")
        from .app import main as text_main
        text_main()
        return
    gui_main()


if __name__ == "__main__":
    main()

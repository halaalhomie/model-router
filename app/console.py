import sys


def use_utf8(stream: object) -> None:
    """Force UTF-8 on a console stream.

    Windows consoles default to cp1252, which cannot encode characters that
    models emit constantly (em dashes, arrows, box drawing). Without this the
    answer is lost to a UnicodeEncodeError after we have already paid for it.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


def use_utf8_stdio() -> None:
    """Apply use_utf8 to both console streams, for a CLI entry point."""
    use_utf8(sys.stdout)
    use_utf8(sys.stderr)

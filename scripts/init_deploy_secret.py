"""Generate a Compose API secret locally, without printing or overwriting it."""
import os
from pathlib import Path
import secrets


def main():
    directory = Path(__file__).resolve().parents[1] / ".secrets"
    directory.mkdir(mode=0o700, exist_ok=True)
    try:
        # 0644 is needed by the non-root container UID with Compose file mounts.
        # The parent is 0700, so other host users cannot traverse it on POSIX.
        fd = os.open(directory / "api_key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        print("Existing deployment secret retained.")
        return
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(secrets.token_urlsafe(32) + "\n")
    print("Created deployment API secret. No key was printed.")


if __name__ == "__main__":
    main()

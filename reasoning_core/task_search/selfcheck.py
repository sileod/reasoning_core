"""CLI compatibility wrapper for the shared validation pipeline."""

import sys

from .validation import selfcheck_main


def main(argv=None):
    return selfcheck_main(argv)


if __name__ == "__main__":
    sys.exit(main())

#!/Users/Martin/Downloads/stellar_feedback/athenapk_stars/.venv/bin/python
import sys
from clang_format.clang_format_diff import main
if __name__ == '__main__':
    sys.argv[0] = sys.argv[0].removesuffix('.exe')
    sys.exit(main())

from pathlib import Path
import sys


APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from ddo.main import main


if __name__ == "__main__":
    main()

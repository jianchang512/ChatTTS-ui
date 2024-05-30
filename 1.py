from pathlib import Path
import sys
print(Path(sys.executable).parent.as_posix())
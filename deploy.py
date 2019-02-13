import shutil, errno
from pathlib import Path
import os

from motion.utils.filename_maker import del_folder


def copyanything(src, dst):
    del_folder(dst)
    shutil.copytree(src, dst)


def check_tests_passed():
    with open("test_results.txt", "r") as f:
        for line in f:
            if line[0] == "F":
                Warning("Seems your test do not pass")

    os.remove("test_results.txt")


def main():

    os.system("python -m pytest --result-log=test_results.txt")
    check_tests_passed()
    path = Path("/home", "tako", "devtools")
    copyanything(
        Path(path / "furry-geras" / "motion"),
        Path(path / "furry-geras-app" / "app" / "motion"),
    )


if __name__ == "__main__":
    main()

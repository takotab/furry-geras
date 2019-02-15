import os
from pathlib import Path
import datetime


def filename_maker(name=None):
    if name is None:
        name = make_name()
    else:
        name = name.split("/")[-1].split(".")[0]
    dir = Path("output/temp_" + name)
    if os.path.exists(dir):
        del_folder(dir, "*temp*")
    if not os.path.exists("output"):
        os.mkdir("output")

    os.mkdir(dir)
    return Path(dir)


def make_name():
    return datetime.datetime.now().isoformat()


def del_folder(folder, glob_pattern="*"):
    if not os.path.exists(folder):
        return
    for f in folder.glob(glob_pattern):
        if os.path.isdir(f):
            del_folder(f)
        else:
            print("delete", f)
            os.remove(f)
    try:
        os.removedirs(folder)
    except:
        pass

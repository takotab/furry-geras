import os
from motion import utils


def test_utils_dest():
    urls_dest = utils.mdl_url_dest()
    for key in urls_dest:
        assert os.path.exists(urls_dest[key]["dest"]), urls_dest[key]["dest"]

import os


def mdl_url_dest(app_path=None):
    if app_path is None:
        app_path = "/" + os.path.join(*__file__.split(os.sep)[:-3])

    return {
        "pose": {
            "url": "https://drive.google.com/uc?export=download&id=1RVKXdDggdXnb9UU4efosCBLZMkj0vXtN",
            "dest": app_path + "/models/pose_resnet_50_256x192.pth.tar",
        },
        "orient": {
            "url": "https://drive.google.com/uc?export=download&id=1RVKXdDggdXnb9UU4efosCBLZMkj0vXtN",
            "dest": app_path + "/models/OrientMdlv_2.pth",
        },
        "detect_human_ssd": {
            "url": "https://drive.google.com/uc?export=download&id=1RVKXdDggdXnb9UU4efosCBLZMkj0vXtN",
            "dest": app_path + "/models/mb2-ssd-lite-mp-0_686.pth",
        },
    }


# https://syncwithtech.blogspot.com/p/direct-download-link-generator.html

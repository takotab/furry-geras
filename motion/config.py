input_size_posenet = {"width": 192, "height": 256}


def get_image_size(key=None):
    if key is None:
        return input_size_posenet
    elif key == "both":
        return (input_size_posenet["width"], input_size_posenet["height"])
    else:
        return input_size_posenet[key]


def output_shape(key, input=None):
    if key == "height":
        return 64
    elif key == "width":
        return 48
    else:
        raise KeyError()


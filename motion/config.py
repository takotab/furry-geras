input_size_posenet = {"height": 256, "width": 192}


def get_image_size(key=None):
    if key is None:
        return input_size_posenet
    else:
        return input_size_posenet[key]


def output_shape(key):
    if input_size_posenet["height"] == 256 and input_size_posenet["width"] == 192:
        if key == "height":
            return 64
        elif key == "width":
            return 48
    else:
        raise NotImplementedError()


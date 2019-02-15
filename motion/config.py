resize = {"height": 256, "width": 192}


def get_image_size(key=None):
    if key is None:
        return resize
    else:
        return resize


def output_shape(key):
    if resize["height"] == 256 and resize["width"] == 192:
        if key == "height":
            return 64
        elif key == "width":
            return 48
    else:
        raise NotImplementedError()


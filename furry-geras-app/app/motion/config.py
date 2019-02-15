resize = {"height": 256, "width": 192}


def output_shape(key):
    if resize["height"] == 256 and resize["width"] == 192:
        if key == "height":
            return 64
        elif key == "width":
            return 48
    else:
        raise NotImplementedError()


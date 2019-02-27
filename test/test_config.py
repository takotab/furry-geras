from motion import config


def test_get_image_size():
    posenet_input = config.get_image_size()
    assert set(posenet_input.keys()) == set(["width", "height"])
    assert config.get_image_size("both") == (
        posenet_input["width"],
        posenet_input["height"],
    )
    for key in posenet_input:
        assert posenet_input[key] == config.get_image_size(key)


def test_output_shape():
    assert config.output_shape("height") == 64
    assert config.output_shape("width") == 48
    try:
        config.output_shape("both")
        assert False
    except KeyError:
        assert True

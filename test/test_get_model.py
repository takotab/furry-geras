import motion


def test_get_model():
    model = motion.get_model()
    assert hasattr(model, "final_layer")

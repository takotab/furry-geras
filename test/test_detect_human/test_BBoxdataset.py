import numpy as np
from motion.detect_human import BBoxDataset


def test_make_inference():
    bbox_ds = BBoxDataset(None, fastai_out=True)
    assert len(bbox_ds) == 1


def test_bbox_dataset():
    size = 500
    bbox_data = BBoxDataset(
        "test/assets/one_sample_dataset.csv", size=size, type="valid", fastai_out=True
    )
    sample = bbox_data[0]
    assert sample[0].shape == (3, size, size)
    assert sample[0].dtype == np.float32
    # assert list(np.round(sample[1], 3)) == [0.548, 0.236, 0.802, 0.868]


def test_one_dim():
    bbox_ds = BBoxDataset("coco/val2017_one_human.csv", type="valid")

    def test_fn(fn, start, length, orign_len, size, **kwags):
        # plt.figure()
        or_img = np.ones(orign_len)
        or_img[start : start + length] = 2
        n_img = np.random.rand(size)
        # plt.plot(or_img)
        # plt.plot(n_img)
        # plt.figure()
        for pick_start in [np.min, np.max]:
            x, bbox = fn(start, length, orign_len)
            assert bbox[1] > bbox[0] >= 0
            assert x[1] > x[0] >= 0
            assert x[2] >= 0
            assert x[3] > 0
            # print(x)
            n_img[x[0] : x[1]] = or_img[x[2] : x[3]]
        # plt.plot(n_img)
        # plt.plot([bbox[0], bbox[1]], [1, 1])
        # plt.title(str(x) + str(bbox))

    start, length, orign_len, size = 50, 550, 650, 500
    test_fn(bbox_ds._resize_one_dims, start, length, orign_len, size)

    start, length, orign_len, size = 50, 151, 651, 501
    test_fn(bbox_ds._resize_one_dims, start, length, orign_len, size)

    start, length, orign_len, size = 50, 150, 400, 500
    test_fn(bbox_ds._resize_one_dims, start, length, orign_len, size)

    start, length, orign_len, size = 344, 127, 500, 500
    test_fn(bbox_ds._resize_one_dims, start, length, orign_len, size)
    start, length, orign_len, size = 100, 501, 640, 500
    test_fn(bbox_ds._resize_one_dims, start, length, orign_len, size)

    start, length, orign_len, size = 92, 338, 640, 500
    test_fn(bbox_ds._resize_one_dims, start, length, orign_len, size)

    start, length, orign_len, size = 269, 358, 640, 500
    test_fn(bbox_ds._resize_one_dims, start, length, orign_len, size)

    start, length, orign_len, size = 0, 250, 640, 500
    test_fn(bbox_ds._resize_one_dims, start, length, orign_len, size)

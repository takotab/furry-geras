import numpy as np
from motion.detect_human import BBoxDataset


def test_bbox_dataset():
    size = 500
    bbox_data = BBoxDataset("coco/val2017_one_human.csv", size=size)
    sample = bbox_data[0]
    assert sample["image"].shape == (size, size, 3)
    assert ["image", "bboxes", "category_id"] == list(sample.keys())


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
        x, bbox = fn(start, length, orign_len)
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

    start, length, orign_len, size = 344, 127, 640, 500
    test_fn(bbox_ds._resize_one_dims, start, length, orign_len, size)
    start, length, orign_len, size = 100, 501, 640, 500
    test_fn(bbox_ds._resize_one_dims, start, length, orign_len, size)


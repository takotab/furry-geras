from motion.orient_mdl.dataset import RotationDataset


def test_training_eval():
    rot_ds = RotationDataset("test/assets/one_sample_dataset.csv")
    assert len(rot_ds) == 1
    rot_ds[0]
    assert True

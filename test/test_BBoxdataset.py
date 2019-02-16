from torchvision import transforms  # datasets, models,
from motion.detect_human import BBoxDataset


def test_bbox_dataset():
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    bbox_data = BBoxDataset("coco/one_human.csv", transform=transform)
    print(bbox_data[0])

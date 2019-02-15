import matplotlib.pyplot as plt

from motion.detect_human import COCOHumanBBoxDataset


def main():
    dataDir = "/home/tako/devtools/furry-geras/coco"
    dataType = "val2017"
    annFile = "{}/instances_{}.json".format(dataDir, dataType)
    coco = COCOHumanBBoxDataset(annFile)
    img, bbox = coco.get_one_pic_w_bb(1490)

    img.show(y=bbox)
    plt.show
    print("doen")


if __name__ == "__main__":
    main()

from motion.detect_human import COCOHumanBBoxDataset


def main():
    dataDir = "/home/tako/devtools/furry-geras/coco"
    dataType = "val2017"
    annFile = "{}/instances_{}.json".format(dataDir, dataType)
    coco = COCOHumanBBoxDataset(annFile)
    print(coco.get_one_pic_w_bb(1490))


if __name__ == "__main__":
    main()

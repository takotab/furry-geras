import cv2


label_path = "models/voc-model-labels.txt"


def predict_image(image_path, mdl, save_result=False, top_k=10, prob_threshold=0.4):

    orig_image = cv2.imread(image_path)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = mdl.predict(image, top_k, prob_threshold)

    if save_result:
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            cv2.rectangle(
                orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4
            )
            # label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
            if mdl.class_names is None:
                class_names = [name.strip() for name in open(label_path).readlines()]
            else:
                class_names = mdl.class_names
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.putText(
                orig_image,
                label,
                (box[0] + 20, box[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2,
            )  # line type
        path = "run_ssd_example_output.jpg"
        cv2.imwrite(path, orig_image)
        print(f"Found {len(probs)} objects. The output image is {path}")
    return boxes, labels, probs


def predict_video():
    pass

import cv2


label_path = "models/voc-model-labels.txt"


def predict_image(image_path, mdl, save_result=False, top_k=10, prob_threshold=0.4):

    orig_image = cv2.imread(image_path)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    bbox_preds = mdl.predict(image, top_k, prob_threshold)

    if save_result:
        save_results(orig_image, bbox_preds, mdl)
    return bbox_preds


def predict_video(video_array, mdl, top_k=10, prob_threshold=0.4):
    return mdl.predict_video(video_array, top_k, prob_threshold)


def save_results(orig_image, bbox_preds, mdl):
    for bbox_pred in bbox_preds:
        box = bbox_pred["bbox"].get_all(conv_type=int)
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        # label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""

        label = f"{bbox_pred['label']}: {bbox_pred['prob']:.2f}"
        cv2.putText(
            orig_image,
            label,
            (box[0] + 20, box[1] + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,  # font scale
            (255, 0, 255),
            2,
        )  # line type
    path = "run_ssd_test_output.jpg"
    cv2.imwrite(path, orig_image)
    print(f"Found {len(bbox_preds)} objects. The output image is {path}")

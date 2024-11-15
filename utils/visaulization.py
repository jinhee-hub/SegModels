import torch
import cv2
import numpy as np

def save_samples(images, labels, outputs, file_names, batch_index, output_dir, ignore_classids):
    # RGB
    palette = np.array([
        [0, 0, 0],  # Class 0 - 검은색
        [255, 0, 0],  # Class 1 - 빨간색
        [0, 0, 255],  # Class 2 - 파란색
    ], dtype=np.uint8)

    alpha = 0.8
    palette = palette[:, [2, 1, 0]] # RGB -> BRG for cv2
    images_np = images.cpu().detach().numpy().transpose(0, 2, 3, 1) * 255
    images_np = images_np.astype(np.uint8)

    labels_np = labels.cpu().detach().numpy()
    outputs_np = outputs.cpu().detach().numpy()
    predictions_np = np.argmax(outputs_np, axis=1)
    for i in range(images.shape[0]):
        img_rgb = images_np[i]
        file_name = file_names[i]

        label_color_map = np.zeros((labels_np[i].shape[0], labels_np[i].shape[1], 3), dtype=np.uint8)
        pred_color_map = np.zeros((predictions_np[i].shape[0], predictions_np[i].shape[1], 3), dtype=np.uint8)

        overlay_label = img_rgb.copy()
        overlay_pred = img_rgb.copy()

        mask_label = (labels_np[i] >= 0) & (np.isin(labels_np[i], ignore_classids) == False)
        mask_pred = (predictions_np[i] >= 0) & (np.isin(predictions_np[i], ignore_classids) == False)

        print(f"Debug Info for Batch {batch_index + i}")
        print(f"mask_label unique values: {np.unique(mask_label)}, shape: {mask_label.shape}")
        print(f"mask_pred unique values: {np.unique(mask_pred)}, shape: {mask_pred.shape}")
        print(f"Outputs shape: {outputs_np[i].shape}, Predictions unique values: {np.unique(predictions_np[i])}")

        for j in range(palette.shape[0]):
            if j not in ignore_classids:
                label_color_map[labels_np[i] == j] = palette[j]
                pred_color_map[predictions_np[i] == j] = palette[j]


        overlay_label[mask_label] = cv2.addWeighted(
            img_rgb[mask_label], 1 - alpha, label_color_map[mask_label], alpha, 0
        )
        overlay_pred[mask_pred] = cv2.addWeighted(
            img_rgb[mask_pred], 1 - alpha, pred_color_map[mask_pred], alpha, 0
        )

        # 저장
        cv2.imwrite(f"{output_dir}/image_{file_name}", img_rgb)  # image
        # cv2.imwrite(f"{output_dir}/labelmap_{file_name}", label_color_map)       # label map
        # cv2.imwrite(f"{output_dir}/predmap_{file_name}", pred_color_map)   # pred map
        cv2.imwrite(f"{output_dir}/label_{file_name}", overlay_label)   # overlay label map on the image
        cv2.imwrite(f"{output_dir}/pred_{file_name}", overlay_pred) # overlay pred map on the image
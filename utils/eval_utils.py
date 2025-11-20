import cv2
import os
import numpy as np


def specify_resolution(image_list, score_list, mask_list, resolution: tuple = (400, 400)):
    resize_image = []
    resize_score = []
    resize_mask = []
    print(f"调整分辨率到: {resolution}")

    # 确保所有列表长度一致
    n = len(score_list)
    if image_list:
        assert len(image_list) == n, "image_list长度与score_list不一致"
    if mask_list:
        assert len(mask_list) == n, "mask_list长度与score_list不一致"

    # 使用score_list的长度作为主要参考
    for i in range(n):
        # 处理分数图
        score = score_list[i]
        if score is not None:
            # 确保分数图是浮点类型
            if score.dtype != np.float32:
                score = score.astype(np.float32)
            score_resized = cv2.resize(score, (resolution[0], resolution[1]), interpolation=cv2.INTER_CUBIC)
        else:
            score_resized = None

        # 处理图像（如果有）
        image = image_list[i] if i < len(image_list) and image_list else None
        if image is not None:
            image_resized = cv2.resize(image, (resolution[0], resolution[1]), interpolation=cv2.INTER_CUBIC)
        else:
            image_resized = None

        # 处理掩码（如果有）
        mask = mask_list[i] if i < len(mask_list) and mask_list else None
        if mask is not None:
            # 确保掩码是整型
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            mask_resized = cv2.resize(mask, (resolution[0], resolution[1]), interpolation=cv2.INTER_NEAREST)
            # 确保二值化
            mask_resized = (mask_resized > 0).astype(np.uint8)
        else:
            mask_resized = None

        resize_image.append(image_resized)
        resize_score.append(score_resized)
        resize_mask.append(mask_resized)

    return resize_image, resize_score, resize_mask

def normalize(scores):

    max_value = np.max(scores)
    min_value = np.min(scores)

    norml_scores = (scores - min_value) / (max_value - min_value)
    return norml_scores

def save_single_result(classification_score, segmentation_score, root_dir, shot_name, experiment_indx, subset_name, defect_type, name, use_defect_type):

    if use_defect_type:
        # mvtec2d mvtec3d
        save_dir = os.path.join(root_dir, shot_name, experiment_indx, subset_name, defect_type)
    else:
        # visa
        save_dir = os.path.join(root_dir, shot_name, experiment_indx, subset_name)

    os.makedirs(save_dir, exist_ok=True)

    classification_dir = os.path.join(save_dir, 'classification')
    segmentation_dir = os.path.join(save_dir, 'segmentation')
    os.makedirs(classification_dir, exist_ok=True)
    os.makedirs(segmentation_dir, exist_ok=True)

    classification_path = os.path.join(classification_dir, f'{name}.txt')
    segmentation_path = os.path.join(segmentation_dir, f'{name}.npz')

    with open(classification_path, "w") as f:
        f.write(f'{classification_score:.5f}')

    segmentation_score = np.round(segmentation_score * 255).astype(np.uint8)
    np.savez_compressed(segmentation_path, img=segmentation_score)

def save_results(classification_score_list, segmentation_score_list, root_dir, shot_name, experiment_indx, name_list, use_defect_type):

    for classification_score, segmentation_score, full_name in zip(classification_score_list,
                                                                           segmentation_score_list,
                                                                           name_list):
        subset_name, defect_type, name = full_name.split('-')
        save_single_result(classification_score, segmentation_score, root_dir, shot_name, experiment_indx, subset_name, defect_type, name, use_defect_type)

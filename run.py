# run.py

import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.load_config import load_yaml
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from utils.eval_utils import *
from zcr import ZCR
from datasets import *
from utils.visualization import (
    denormalize_safer,
    plot_score_distributions,
    plot_image_level_score_distributions,
    plot_feature_distribution_tsne,
    create_heatmap_overlay,
    generate_comparison_figure,
    create_confidence_map_visualization
)


def test(model,
         dataloader: DataLoader,
         device: str,
         is_vis: bool,
         img_dir: str,
         class_name: str,
         cal_pro: bool,
         train_data: DataLoader,
         resolution: int):
    # 切换到评估模式
    model.eval()

    # 初始化存储容器
    image_scores_list = []
    pixel_scores_list = []
    confidence_maps_list = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    names = []
    global_features_list = []
    full_paths_list = []

    # 批次处理测试数据
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        masks = batch['gt'].cpu().numpy()
        labels = batch['label'].cpu().numpy()
        names_batch = batch['name']

        if 'path' in batch:
            full_paths_list.extend(batch['path'])
        else:
            if is_vis:
                print("警告: Dataloader的batch中未找到'path'键，无法生成Overlay和Comparison图。")

        with torch.no_grad():
            batch_pixel_scores, batch_image_scores, batch_global_features,batch_confidence_maps = model(images)

        pixel_scores_list.extend(batch_pixel_scores)
        image_scores_list.extend(batch_image_scores)
        # ✨ 将置信度图存入新列表
        confidence_maps_list.extend(batch_confidence_maps)
        global_features_list.append(batch_global_features)

        gt_list.extend(labels.astype(int))
        gt_mask_list.extend((masks > 0).astype(np.uint8))
        names.extend(names_batch)
        if is_vis:
            test_imgs.extend([denormalize_safer(img.cpu()) for img in images])

    # 后处理与评估
    if is_vis:
        test_imgs, processed_pixel_scores, gt_mask_list_resized = specify_resolution(
            test_imgs, pixel_scores_list, gt_mask_list,
            resolution=(resolution, resolution)
        )
    else:
        _, processed_pixel_scores, gt_mask_list_resized = specify_resolution(
            [], pixel_scores_list, gt_mask_list,
            resolution=(resolution, resolution)
        )

    # 指标计算 (注意：MVTec数据集可能没有图像级标签，这里假设有)
    result_dict = metric_cal(
        np.array(processed_pixel_scores),
        gt_list=gt_list,
        gt_mask_list=np.array(gt_mask_list_resized),
        cal_pro=cal_pro
    )

    # 可视化输出
    if is_vis:
        # 绘制分数分布图和t-SNE图
        all_scores_np = np.array(processed_pixel_scores)
        all_masks_np = np.array(gt_mask_list_resized)
        scores_flat = all_scores_np.flatten()
        masks_flat = all_masks_np.flatten()
        plot_score_distributions(
            normal_scores=scores_flat[masks_flat == 0],
            abnormal_scores=scores_flat[masks_flat != 0],
            save_dir=img_dir, class_name=class_name
        )
        all_image_scores_np = np.array(image_scores_list)
        all_image_labels_np = np.array(gt_list)
        plot_image_level_score_distributions(
            normal_image_scores=all_image_scores_np[all_image_labels_np == 0],
            abnormal_image_scores=all_image_scores_np[all_image_labels_np == 1],
            save_dir=img_dir, class_name=class_name
        )
        all_global_features = np.concatenate(global_features_list, axis=0)
        plot_feature_distribution_tsne(
            features=all_global_features,
            labels=np.array(gt_list),
            save_dir=img_dir, class_name=class_name
        )

        # 生成Overlay和Comparison图
        if full_paths_list:
            overlay_save_dir = os.path.join(img_dir, "overlays")
            comparison_save_dir = os.path.join(img_dir, "comparisons_vertical")
            os.makedirs(overlay_save_dir, exist_ok=True)
            os.makedirs(comparison_save_dir, exist_ok=True)
            confidence_save_dir = os.path.join(img_dir, "confidence")
            os.makedirs(confidence_save_dir, exist_ok=True)

            print(f"\n🚀 正在为 {len(full_paths_list)} 个样本生成可视化图...")

            for i in range(len(full_paths_list)):
                original_path = full_paths_list[i]
                anomaly_map = processed_pixel_scores[i]
                gt_mask_for_vis = gt_mask_list_resized[i]
                image_label = gt_list[i]
                # ✨ 获取当前样本的置信度图
                confidence_map = confidence_maps_list[i]

                base_name = os.path.basename(original_path)
                file_name_no_ext = os.path.splitext(base_name)[0]

                # ✨ 关键：根据标签添加文件名状态前缀
                status_prefix = "abnormal" if image_label == 1 else "normal"

                # 1. 生成Overlay图
                overlay_output_path = os.path.join(overlay_save_dir, f"{class_name}_{status_prefix}_{file_name_no_ext}_overlay.jpg")
                create_heatmap_overlay(
                    original_image_path=original_path,
                    anomaly_map_raw=anomaly_map,
                    output_path=overlay_output_path,
                    image_weight=0.4,  # 原图权重，值越小热力图越明显
                    sigma=4,  # 高斯平滑核大小
                    percentile=95  # 只显示最明显的5%异常，减少背景噪声
                )

                # 2. 生成三图垂直对比图
                comparison_output_path = os.path.join(comparison_save_dir,
                                                      f"{class_name}_{status_prefix}_{file_name_no_ext}_comparison.jpg")
                generate_comparison_figure(
                    original_image_path=original_path,
                    anomaly_map_raw=anomaly_map,
                    gt_mask=gt_mask_for_vis,
                    output_path=comparison_output_path,
                    image_weight=0.4,
                    sigma=4,
                    percentile=95,
                    figsize=(8, 18)  # <-- 从 (8, 24) 修改为 (8, 18)
                )

                confidence_output_path = os.path.join(confidence_save_dir,
                                                      f"{class_name}_{status_prefix}_{file_name_no_ext}_confidence.jpg")
                create_confidence_map_visualization(
                    original_image_path=original_path,
                    confidence_map=confidence_map,
                    output_path=confidence_output_path
                )

            print(f"✅ Overlay 图已保存至: {overlay_save_dir}")
            print(f"✅ Comparison 图已保存至: {comparison_save_dir}")

    return result_dict


def main():
    cfgs = load_yaml("./config/cfg_aaclip.yaml")

    for cls in cfgs['classes_name']:
        cfgs['class_name'] = cls
        print(f"\n{'=' * 20} Processing class: {cls} {'=' * 20}")

        model_dir, img_dir, logger_dir, model_name, csv_path = get_dir_from_args(
            root_dir=cfgs['output_dir'],
            class_name=cls,
            dataset=cfgs['dataset'],
            k_shot=cfgs['k_shot'],
            experiment_indx=cfgs['experiment_indx']
        )

        # 确保日志和图像目录存在
        os.makedirs(logger_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        # logger.start(os.path.join(logger_dir, f"log_{time.strftime('%Y-%m-%d-%H-%M-%S')}.log"))
        print(f"=== Running experiment for class: {cls} ===")
        print(f"    - Output images will be saved to: {img_dir}")

        # ... (设置随机种子等) ...

        device = "cuda:0" if cfgs['use_cpu'] == 0 else "cpu"
        cfgs['device'] = device

        train_dataloader = None
        if cfgs.get('k_shot', 0) > 0:
            train_dataloader, _ = get_dataloader_from_args(phase='train', **cfgs)

        test_dataloader, _ = get_dataloader_from_args(phase="test", **cfgs)

        cfgs['out_size_h'] = cfgs['resolution']
        cfgs['out_size_w'] = cfgs['resolution']

        model = ZCR(cls, **cfgs).to(device)

        metrics = test(
            model=model,
            dataloader=test_dataloader,
            device=device,
            is_vis=cfgs['visualize'],
            img_dir=img_dir,
            class_name=cls,
            cal_pro=cfgs.get('cal_pro', False),
            train_data=train_dataloader,
            resolution=cfgs['resolution']
        )

        print(f"\n=== Metrics for {cls} ===")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

        save_metric(
            metrics=metrics,
            dataset=cfgs['dataset'],
            total_classes=dataset_classes.get(cfgs['dataset'], [cls]),  # 修正
            class_name=cfgs['class_name'],
            csv_path=csv_path
        )


if __name__ == "__main__":
    main()
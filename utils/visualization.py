# visualization.py

import cv2
import matplotlib
import numpy as np
import os
import seaborn as sns
import torch
from scipy.ndimage import gaussian_filter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image

matplotlib.use("Agg")


def denormalize_safer(img_tensor: torch.Tensor) -> np.ndarray:
    """健壮的 PyTorch Tensor 反归一化函数"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_tensor.cpu().detach().numpy().transpose(1, 2, 0)
    img_np = (img_np * std) + mean
    img_np = np.clip(img_np, 0, 1)
    img_uint8 = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    return img_bgr


# In visualization.py

def _process_anomaly_map(anomaly_map_raw, percentile=99, sigma=4):  # <-- 默认percentile改为99
    """
    (最新优化版) 对异常图进行归一化、平滑和更严格的阈值化处理，使异常区域更集中。
    """
    anomaly_map = anomaly_map_raw.copy()

    # 1. 全局归一化到 [0, 1]，这是最关键的一步，确保分数的绝对性
    min_val, max_val = anomaly_map.min(), anomaly_map.max()
    if max_val > min_val:
        anomaly_map = (anomaly_map - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(anomaly_map)  # 如果所有值都一样，返回全零图

    # 2. 高斯平滑，减少噪声，使热力图更平滑
    if sigma > 0:
        anomaly_map = gaussian_filter(anomaly_map, sigma=sigma)

    # 3. 阈值化：计算阈值，并将低于阈值的分数置零，以突出最显著区域
    if percentile is not None and percentile < 100:
        threshold = np.percentile(anomaly_map, percentile)
        anomaly_map[anomaly_map < threshold] = 0

    # ✨ 新增：在阈值化后，对剩余的非零值进行二次归一化，以增强显著异常的对比度
    # 这确保了即使percentile很高，被保留下来的异常也能显示为从0到1的完整色谱，更清晰
    high_anomaly_mask = anomaly_map > 0
    if np.any(high_anomaly_mask):
        high_scores = anomaly_map[high_anomaly_mask]
        # 重新拉伸到 [0, 1]，但只针对阈值化后的非零区域
        renormalized_scores = (high_scores - high_scores.min()) / (high_scores.max() - high_scores.min() + 1e-8)
        anomaly_map[high_anomaly_mask] = renormalized_scores

    return anomaly_map

def create_heatmap_overlay(original_image_path, anomaly_map_raw, output_path,
                           image_weight=0.4, sigma=4, percentile=95):
    """
    使用 grad-cam 风格和优化的热力图逻辑创建高质量叠加图
    """
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        print(f"错误: 无法加载原始图像 {original_image_path}")
        return

    original_image_float = np.float32(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)) / 255
    img_height, img_width = original_image_float.shape[:2]

    # ✨ 使用新的核心函数处理异常图
    processed_map = _process_anomaly_map(anomaly_map_raw, percentile, sigma)

    # 调整尺寸
    processed_map_resized = cv2.resize(processed_map, (img_width, img_height), interpolation=cv2.INTER_LINEAR)

    # 使用 grad-cam 库的函数来生成叠加图
    # 这里的 anomaly_map 已经是处理过的，可以直接使用
    overlay_rgb = show_cam_on_image(original_image_float, processed_map_resized, use_rgb=True,
                                    image_weight=image_weight)

    final_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, final_bgr)


# In visualization.py

# In visualization.py

# In visualization.py

def generate_comparison_figure(original_image_path, anomaly_map_raw, gt_mask, output_path,
                               image_weight=0.4, sigma=4, percentile=95, figsize=(8, 18)):
    """
    (再修改版) 生成包含原图、纯黑白标签图和Overlay图的三图垂直对比图
    """
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        print(f"错误: 无法加载原始图像 {original_image_path}")
        return
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_float = np.float32(original_image_rgb) / 255
    img_height, img_width = original_image_rgb.shape[:2]

    # ✨ 核心修改部分：生成纯黑白的二值标签图
    # 1. 创建一个与原图同样大小的黑色画布 (RGB三通道)
    gt_binary_image_rgb = np.zeros_like(original_image_rgb)

    # 2. 调整 ground truth mask 的尺寸以匹配图像
    gt_mask_resized = cv2.resize(gt_mask.astype(np.uint8), (img_width, img_height), interpolation=cv2.INTER_NEAREST)

    # 3. 将 mask 中非零（即缺陷）的区域在黑色画布上设置为白色
    #    gt_mask_resized > 0 会创建一个布尔索引
    gt_binary_image_rgb[gt_mask_resized > 0] = [255, 255, 255]

    # --- 后续部分与之前类似 ---

    # 2. 生成优化后的 Overlay 图
    processed_map = _process_anomaly_map(anomaly_map_raw, percentile, sigma)
    processed_map_resized = cv2.resize(processed_map, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    overlay_image = show_cam_on_image(original_image_float, processed_map_resized, use_rgb=True,
                                      image_weight=image_weight)

    # 3. 创建对比图 (垂直布局，三张图)
    plt.figure(figsize=figsize, dpi=150)

    # 子图1: 原图
    plt.subplot(3, 1, 1)
    plt.imshow(original_image_rgb)
    plt.title('Original Image', fontsize=16)
    plt.axis('off')

    # 子图2: 纯黑白标签图
    plt.subplot(3, 1, 2)
    plt.imshow(gt_binary_image_rgb)  # <-- 修改：显示我们新创建的黑白图
    plt.title('Ground Truth (Binary)', fontsize=16)  # <-- 修改：标题也更新一下
    plt.axis('off')

    # 子图3: Overlay图
    plt.subplot(3, 1, 3)
    plt.imshow(overlay_image)
    plt.title('Overlay on Image', fontsize=16)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()



def plot_score_distributions(normal_scores, abnormal_scores, save_dir, class_name):
    # ... (此函数无改动) ...
    try:
        plot_save_path = os.path.join(save_dir, "diagnostic_distributions")
        os.makedirs(plot_save_path, exist_ok=True)
        plt.figure(figsize=(10, 6))
        sns.histplot(normal_scores, color="blue", label=f"Normal Pixels (count: {len(normal_scores)})", stat="density",
                     bins=100, kde=True)
        sns.histplot(abnormal_scores, color="red", label=f"Abnormal Pixels (count: {len(abnormal_scores)})",
                     stat="density", bins=100, kde=True)
        plt.title(f'Pixel-Level Score Distribution for Class: {class_name}')
        plt.xlabel('Pixel Anomaly Score')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        save_path = os.path.join(plot_save_path, f"{class_name}_pixel_distribution.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"⚠️ [可視化警告] 無法繪製像素級分佈圖: {e}")


def plot_image_level_score_distributions(normal_image_scores, abnormal_image_scores, save_dir, class_name):
    # ... (此函数无改动) ...
    try:
        plot_save_path = os.path.join(save_dir, "diagnostic_distributions")
        os.makedirs(plot_save_path, exist_ok=True)
        plt.figure(figsize=(10, 6))
        sns.histplot(normal_image_scores, color="blue", label=f"Normal Images (count: {len(normal_image_scores)})",
                     stat="density", bins=50, kde=True)
        sns.histplot(abnormal_image_scores, color="red", label=f"Abnormal Images (count: {len(abnormal_image_scores)})",
                     stat="density", bins=50, kde=True)
        plt.title(f'Image-Level Score Distribution for Class: {class_name}')
        plt.xlabel('Image-Level Anomaly Score')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        save_path = os.path.join(plot_save_path, f"{class_name}_image_distribution.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"⚠️ [可視化警告] 無法繪製圖像級分佈圖: {e}")


def plot_feature_distribution_tsne(features, labels, save_dir, class_name,
                                   title="t-SNE Visualization of Image Features"):
    # ... (此函数已优化，无须再改) ...
    if len(features) < 2: return
    try:
        if features.shape[1] > 50:
            features = PCA(n_components=50, random_state=42).fit_transform(features)
        perplexity_value = min(30, len(features) - 1)
        if perplexity_value <= 0: return
        tsne = TSNE(n_components=2, perplexity=perplexity_value, n_iter=2500, random_state=42, learning_rate='auto',
                    init='random')
        tsne_results = tsne.fit_transform(features)
        plt.figure(figsize=(10, 10))
        palette = {0: "#6ab04c", 1: "#eb4d4b"}
        markers = {0: "o", 1: "X"}
        ax = sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, style=labels, markers=markers,
                             palette=palette, s=100, alpha=0.8, edgecolor="w", linewidth=0.5)
        plt.title(f'{title} for Class: {class_name}', fontsize=18, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=14)
        plt.ylabel('t-SNE Component 2', fontsize=14)
        handles, _ = ax.get_legend_handles_labels()
        legend_labels = ['Normal', 'Abnormal']
        unique_labels = np.unique(labels)
        final_labels = [legend_labels[i] for i in unique_labels] if len(unique_labels) < len(
            legend_labels) else legend_labels
        ax.legend(handles, final_labels, title='Classes', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plot_save_path = os.path.join(save_dir, "diagnostic_distributions")
        os.makedirs(plot_save_path, exist_ok=True)
        save_path = os.path.join(plot_save_path, f"{class_name}_tsne_features.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"⚠️ [可視化警告] 無法繪製 t-SNE 分佈圖: {e}")

def create_confidence_map_visualization(original_image_path: str, confidence_map: np.ndarray, output_path: str):
    """
    将文本置信度图可视化并保存。
    """
    if confidence_map is None:
        return # 如果没有置信度图，则不执行任何操作

    # 使用 viridis 或 plasma 色谱，它们对于表示强度信息非常有效
    plt.imshow(confidence_map, cmap='viridis')
    plt.colorbar(label='Text Confidence for "Abnormal"')
    plt.axis('off')
    plt.title('Text Confidence Map', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
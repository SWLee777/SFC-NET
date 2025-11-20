import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from typing import List, Dict, Tuple
from ._clip import AdaptedCLIP, tokenize, create_model
from ._clip import clip
from scipy.ndimage import gaussian_filter
import os
from glob import glob
from skimage import measure

template_level_prompts = [
    'a photo of a {}',
    'a photo of the {}',
    'a cropped photo of a {}',
    'a cropped photo of the {}',
    'a close-up photo of a {}',
    'a close-up photo of the {}',
    'a macro photo of a {}',
    'a macro photo of the {}',
    'a photo of a small {}',
    'a photo of the small {}',
    'a photo of a large {}',
    'a photo of the large {}',
    'a top-down view of the {}',
    'an angled view of the {}',
    'a texture detail of the {}',
    'a bright photo of a {}',
    'a bright photo of the {}',
    'a dark photo of a {}',
    'a dark photo of the {}',
    'a blurry photo of a {}',
    'a blurry photo of the {}',
    'a jpeg corrupted photo of a {}',
    'a high-resolution scan of the {}',
    'a photo of the {} for visual inspection',
    'a photo of a {} for quality control',
    'a photo of the {} for anomaly detection',
    'a photo of the {} on a production line',
]

state_level_normal_prompts = [
    '{}',
    'a normal {}',
    'a standard {}',
    'an intact {}',
    'a clean {}',
    'flawless {}',
    'perfect {}',
    'unblemished {}',
    '{} without flaw',
    '{} without defect',
    '{} without damage',
    '{} in good condition',
    '{} as expected',
    'a correctly assembled {}',
]

state_level_abnormal_prompts = [
    'an abnormal {}',
    'a defective {}',
    'a damaged {}',
    'a broken {}',
    '{} with flaw',
    '{} with defect',
    '{} with damage',
    'scratched {}',
    '{} with a scratch',
    'stained {}',
    '{} with a stain',
    'contaminated {}',
    '{} with contamination',
    'dented {}',
    '{} with a dent',
    'corroded {}',
    '{} with corrosion',
    'peeled {}',
    '{} with peeling surface',
    'bubbly {}',
    '{} with bubbles',
    'smudged {}',
    '{} with a smudge',
    'rusty {}',
    '{} with rust',
    'dirty {}',
    '{} with dirt',
    'cracked {}',
    '{} with a crack',
    'fractured {}',
    '{} with a fracture',
    'torn {}',
    '{} with a tear',
    'deformed {}',
    '{} with a deformation',
    'bent {}',
    'a bent {}',
    'chipped {}',
    '{} with a chip',
    'frayed {}',
    '{} with frayed edges',
    '{} with a hole',
    'misaligned {}',
    '{} with misalignment',
    '{} with a missing part',
    '{} with excess material',
    '{} with a burr',
    'an incorrectly assembled {}',
    'a misplaced {}',
    'discolored {}',
    '{} with discoloration',
    'a burnt {}',
    '{} with a burn mark',
    'a faded {}',
    '{} with faded color',
    '{} with the wrong color',
    '{} with inconsistent color',
]


class ZCR(torch.nn.Module):
    def __init__(self, cls, out_size_h, out_size_w, device, backbone, pretrained_dataset, r_list, features_list,
                 aaclip_path, precision='fp32', text_confidence_weight_alpha=0.3,
                 **kwargs):  # 【修改点 1】: 增加 text_confidence_weight_alpha 超参数
        super(ZCR, self).__init__()
        self.cls = cls
        self.out_size_h = out_size_h
        self.out_size_w = out_size_w
        self.precision = precision
        self.device = device
        self.features_list = features_list
        self.r_list = r_list
        self.text_confidence_weight_alpha = text_confidence_weight_alpha  # 【修改点 2】: 保存超参数

        self.get_aaclip_model(backbone, pretrained_dataset, aaclip_path, self.features_list)
        self.create_block_metadata(self.r_list)

        self.text_feature_version = "V2"
        self.build_text_features_from_gallery()

        print("使用 aa-clip 模型，视觉与文本特征维度已对齐，使用恒等映射 (nn.Identity)。")
        self.visual_projection = nn.Identity()

        self.viz_data = []
        self.fusion_version = "textual_visual"
        print(
            f"模型初始化完成。类别: '{self.cls}', 文本版本: {self.text_feature_version}, 融合策略: {self.fusion_version}, 文本置信度权重alpha: {self.text_confidence_weight_alpha}")

    def get_aaclip_model(self, backbone, pretrained_dataset, aaclip_path, features_list):
        print("--- 正在加載並配置 aa-clip 模型 (最終修正) ---")
        base_clip_model = clip.create_model(
            model_name=backbone,
            img_size=336,
            pretrained=pretrained_dataset,
            device=self.device
        )
        base_clip_model.eval()

        # levels = [6, 12, 18, 24]
        # print(f"根據權重文件結構，aa-clip 將從第 {levels} 層提取特徵。")
        # print("根據權重文件結構，設置 relu=False。")
        # 这是 aaclip 固定的输出层级
        fixed_output_levels = [6, 12, 18, 24]
        print(f"AA-CLIP 模型将固定从第 {fixed_output_levels} 层提取特征。")

        # <--- 关键修改点 2: 计算用户请求的层在固定输出中的索引
        # =========================================================================
        # features_list 是用户想要的层，例如 [24] 或 [6, 24]
        # self.layer_indices_to_use 将会是对应的索引，例如 [3] 或 [0, 3]
        self.layer_indices_to_use = [fixed_output_levels.index(l) for l in features_list if l in fixed_output_levels]
        print(f"用户请求的层级: {features_list}。")
        print(f"实际将使用的特征索引: {self.layer_indices_to_use}。")
        # =========================================================================

        model = AdaptedCLIP(
            clip_model=base_clip_model,
            levels=fixed_output_levels,
            relu=False,
            text_adapt_weight=0.1,
            image_adapt_weight=0.1,
            text_adapt_until=3,
            image_adapt_until=6
        ).to(self.device)
        model.eval()

        text_ckpt_path = os.path.join(aaclip_path, "text_adapter.pth")
        if os.path.exists(text_ckpt_path):
            print(f"從 {text_ckpt_path} 加載 text_adapter 權重...")
            text_checkpoint = torch.load(text_ckpt_path, map_location=self.device)
            model.text_adapter.load_state_dict(text_checkpoint["text_adapter"])
            print("text_adapter 權重加載成功。")
        else:
            print(f"警告：在 {aaclip_path} 中未找到 text_adapter.pth，文本特徵將不被適配。")

        image_ckpt_path = os.path.join(aaclip_path, "image_adapter.pth")
        image_ckpt_files = sorted(glob(os.path.join(aaclip_path, "image_adapter_*.pth")))
        if os.path.exists(image_ckpt_path):
            image_checkpoint = torch.load(image_ckpt_path, map_location=self.device)
        elif image_ckpt_files:
            image_ckpt_path_glob = image_ckpt_files[-1]
            image_checkpoint = torch.load(image_ckpt_path_glob, map_location=self.device)
        else:
            raise FileNotFoundError(
                f"錯誤：在目錄 {aaclip_path} 中找不到 'image_adapter.pth' 或任何 'image_adapter_*.pth' 權重文件。")
        model.image_adapter.load_state_dict(image_checkpoint["image_adapter"])
        print(f"image_adapter 權重 (來自 epoch {image_checkpoint.get('epoch', 'N/A')}) 加載成功。")

        self.clip_model = model
        self.tokenizer = tokenize
        self.grid_size = self.clip_model.image_encoder.grid_size
        print("aa-clip 模型及 Adapter 權重已正確加載. Grid Size:", self.grid_size)

    def create_block_metadata(self, r_list: List[int]):
        print("--- 预计算窗口元数据 (掩码和中心坐标) ---")
        self.masks = []
        self.block_centers = []
        H, W = self.grid_size
        for r in r_list:
            num_blocks_h = H - r + 1
            num_blocks_w = W - r + 1
            for i in range(num_blocks_h):
                for j in range(num_blocks_w):
                    mask = []
                    center_y = i + (r - 1) / 2.0
                    center_x = j + (r - 1) / 2.0
                    for row in range(r):
                        for col in range(r):
                            mask.append((i + row) * W + (j + col))
                    self.masks.append(torch.tensor(mask, dtype=torch.long))
                    self.block_centers.append((center_y, center_x))
        self.block_centers = torch.tensor(self.block_centers, device=self.device)

    @torch.no_grad()
    def encode_text(self, text: torch.Tensor):
        text_features = self.clip_model.encode_text(text)
        return text_features

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor, features_list: List[int]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.precision == 'fp16':
            image = image.half()

        seg_tokens, det_token = self.clip_model(image)
        global_feature = det_token
        patch_tokens_list = seg_tokens

        return global_feature, patch_tokens_list

    def fast_block_features(self, patch_features: torch.Tensor, r_list: List[int]) -> torch.Tensor:
        B, HW, D = patch_features.shape
        H, W = self.grid_size
        assert H * W == HW, f"Grid size {self.grid_size} 不匹配 patch 數量 {HW}"

        features_perm = patch_features.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        block_features_list = []
        for r in r_list:
            pool = nn.AvgPool2d(kernel_size=r, stride=1).to(self.device)
            pooled_features = pool(features_perm)
            blocks = pooled_features.permute(0, 2, 3, 1).reshape(B, -1, D)
            block_features_list.append(blocks)
        return torch.cat(block_features_list, dim=1)

    def build_text_features_from_gallery(self):
        print(
            f"--- 步骤 0: 基于组合模板为类别 '{self.cls}' 构建全局文本特征 (版本: {self.text_feature_version}) ---")
        category = self.cls

        normal_phrases = []
        abnormal_phrases = []
        for template in template_level_prompts:
            for normal_prompt in state_level_normal_prompts:
                normal_phrases.append(template.format(normal_prompt.format(category)))
            for abnormal_prompt in state_level_abnormal_prompts:
                abnormal_phrases.append(template.format(abnormal_prompt.format(category)))

        normal_tokenized = self.tokenizer(normal_phrases).to(self.device)
        abnormal_tokenized = self.tokenizer(abnormal_phrases).to(self.device)

        if self.text_feature_version == "V1":
            normal_features = self.encode_text(normal_tokenized)
            abnormal_features = self.encode_text(abnormal_tokenized)
        elif self.text_feature_version == "V2":
            normal_features_raw = self.encode_text(normal_tokenized)
            normal_features = normal_features_raw / normal_features_raw.norm(dim=-1, keepdim=True)

            abnormal_features_raw = self.encode_text(abnormal_tokenized)
            abnormal_features = abnormal_features_raw / abnormal_features_raw.norm(dim=-1, keepdim=True)
        else:
            raise NotImplementedError(f"文本特征版本 {self.text_feature_version} 未实现")

        avr_normal_text_features = torch.mean(normal_features, dim=0, keepdim=True)
        avr_abnormal_text_features = torch.mean(abnormal_features, dim=0, keepdim=True)

        self.avr_normal_text_features = avr_normal_text_features
        self.avr_abnormal_text_features = avr_abnormal_text_features

        self.text_features = torch.cat([self.avr_normal_text_features, self.avr_abnormal_text_features], dim=0)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        self.normal_text_prototype = self.text_features[0].unsqueeze(0)

        print(f"全局文本特征构建完成, 最终形状: {self.text_features.shape}")

    # 【修改点 3】: 修改函数，使其同时返回异常图和概率图
    def calculate_textual_anomaly_score(self, block_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, D = block_features.shape
        H, W = self.grid_size

        anomaly_maps_per_r = []
        prob_maps_per_r = []  # 新增: 用于存储每个尺度的概率图
        block_cursor = 0

        for r in self.r_list:
            num_blocks_r = (H - r + 1) * (W - r + 1)
            block_end = block_cursor + num_blocks_r

            blocks_r = block_features[:, block_cursor:block_end, :]
            masks_r = self.masks[block_cursor:block_end]

            block_cursor = block_end

            logit_scale = self.clip_model.clipmodel.logit_scale.exp()
            projected_features = self.visual_projection(blocks_r)
            raw_logits = logit_scale * projected_features @ self.text_features.T

            probabilities = raw_logits.softmax(dim=-1)
            normality_score = probabilities[..., 0]
            block_anomaly_scores_r = 1.0 / (normality_score + 1e-8)

            token_scores_r = torch.zeros((B, H * W), device=self.device)
            token_weights_r = torch.zeros_like(token_scores_r)

            # 新增: 初始化用于聚合概率的张量
            token_probs_r = torch.zeros((B, H * W, 2), device=self.device)

            for indx, mask in enumerate(masks_r):
                mask = mask.to(self.device)

                # 聚合异常分数
                current_block_score = block_anomaly_scores_r[:, indx].unsqueeze(1)
                token_scores_r.scatter_add_(1, mask.repeat(B, 1), current_block_score.repeat(1, mask.shape[0]))

                # 聚合权重
                token_weights_r.scatter_add_(1, mask.repeat(B, 1),
                                             torch.ones_like(current_block_score).repeat(1, mask.shape[0]))

                # 新增: 聚合每个token的[正常, 异常]概率
                current_block_probs = probabilities[:, indx, :].unsqueeze(1)  # [B, 1, 2]
                expanded_mask = mask.repeat(B, 1).unsqueeze(-1).expand(-1, -1, 2)
                token_probs_r.scatter_add_(1, expanded_mask, current_block_probs.repeat(1, mask.shape[0], 1))

            # 计算最终的像素级异常分数图
            final_scores_aggregated_r = token_scores_r / (token_weights_r + 1e-8)
            anomaly_map_r = final_scores_aggregated_r.reshape((B, H, W)).unsqueeze(1)
            anomaly_maps_per_r.append(anomaly_map_r)

            # 新增: 计算最终的像素级概率图
            final_probs_aggregated_r = token_probs_r / (token_weights_r.unsqueeze(-1) + 1e-8)
            prob_map_r = final_probs_aggregated_r.reshape((B, H, W, 2)).permute(0, 3, 1, 2)  # [B, 2, H, W]
            prob_maps_per_r.append(prob_map_r)

        stacked_maps = torch.stack(anomaly_maps_per_r, dim=0)
        averaged_map = torch.mean(stacked_maps, dim=0)

        # 新增: 对多尺度的概率图进行平均
        stacked_prob_maps = torch.stack(prob_maps_per_r, dim=0)
        averaged_prob_map = torch.mean(stacked_prob_maps, dim=0)

        final_anomaly_map = 1.0 - 1.0 / (averaged_map + 1e-8)

        # 返回异常图和概率图
        return final_anomaly_map, averaged_prob_map

    def _calculate_anomaly_score_mutual_text_guided(self,
                                                    all_blocks: torch.Tensor,
                                                    topmin_max: float = 0.2,
                                                    # Hyperparameter changed from threshold to a ratio
                                                    text_guidance_top_k_ratio: float = 0.9
                                                    ) -> torch.Tensor:
        print("     -> 步驟 2b: 執行 [自适应语义纯化 Top-K] 批次級互評...")
        B, N, D = all_blocks.shape
        H, W = self.grid_size
        all_blocks = F.normalize(all_blocks, p=2, dim=-1)

        batch_anomaly_scores_list = []
        for i in range(B):
            query_blocks = all_blocks[i]
            per_image_min_dists = []

            for j in range(B):
                if i == j:
                    continue

                ref_blocks = all_blocks[j]

                # ==================== CORE MODIFICATION START ====================
                # The fixed threshold logic is replaced with a more robust Top-K selection.

                with torch.no_grad():
                    # projected_ref_blocks = self.visual_projection(ref_blocks)
                    # # Calculate similarity of each reference block to the "normal" text prototype
                    # text_sim = F.cosine_similarity(projected_ref_blocks, self.normal_text_prototype)
                    # 1. 计算每个参考块的 [正常, 异常] 概率
                    logit_scale = self.clip_model.clipmodel.logit_scale.exp()
                    projected_ref_blocks = self.visual_projection(ref_blocks)
                    # self.text_features 的形状是 [2, D]，分别代表“正常”和“异常”
                    raw_logits = logit_scale * projected_ref_blocks @ self.text_features.T
                    probabilities = raw_logits.softmax(dim=-1)  # shape: [num_ref_blocks, 2]

                    p_normal = probabilities[..., 0]
                    p_abnormal = probabilities[..., 1]

                    # 2. 根据您提出的新公式计算“可靠性分数”
                    # 添加一个很小的 epsilon 防止除以零
                    reliability_score = p_normal + 1.0 / (p_abnormal + 1e-8)

                # 1. Determine the number of blocks to keep based on the ratio.
                num_ref_blocks = ref_blocks.shape[0]
                # Ensure at least one block is kept as a reference.
                k = max(1, int(num_ref_blocks * text_guidance_top_k_ratio))

                # 2. Find the indices of the top-k blocks with the highest similarity to "normal".
                # torch.topk returns a tuple of (values, indices). We only need the indices.
                top_k_indices = torch.topk(reliability_score, k=k, largest=True).indices

                # 3. Select the most reliable reference blocks using these indices.
                candidate_ref_blocks = ref_blocks[top_k_indices]

                # ===================== CORE MODIFICATION END =====================

                # The rest of the logic remains the same
                dists_ij = torch.cdist(query_blocks, candidate_ref_blocks, p=2)
                min_dists_ij, _ = torch.min(dists_ij, dim=1)
                per_image_min_dists.append(min_dists_ij.unsqueeze(1))

            # Handle edge case where batch size is 1
            if not per_image_min_dists:
                batch_anomaly_scores_list.append(torch.zeros(N, device=all_blocks.device))
                continue

            all_min_dists = torch.cat(per_image_min_dists, dim=1)
            k_dist = max(1, int(all_min_dists.shape[1] * topmin_max))
            topk_min_dists, _ = torch.topk(all_min_dists, k=k_dist, largest=False, sorted=True, dim=1)
            final_scores_i = torch.mean(topk_min_dists, dim=1)
            batch_anomaly_scores_list.append(final_scores_i)

        block_anomaly_scores = torch.stack(batch_anomaly_scores_list)

        # Token-level aggregation logic remains unchanged
        token_anomaly_scores = torch.zeros((B, H * W), device=self.device)
        token_weights = torch.zeros((B, H * W), device=self.device)
        for indx in range(N):
            current_block_score = block_anomaly_scores[:, indx].unsqueeze(1)
            mask = self.masks[indx].to(self.device)
            token_anomaly_scores.scatter_add_(1, mask.repeat(B, 1), current_block_score.repeat(1, mask.shape[0]))
            token_weights.scatter_add_(1, mask.repeat(B, 1),
                                       torch.ones_like(current_block_score).repeat(1, mask.shape[0]))

        final_scores = token_anomaly_scores / (token_weights + 1e-8)
        anomaly_map = final_scores.reshape((B, H, W)).unsqueeze(1)

        return anomaly_map

    def forward(self, images: torch.Tensor, **kwargs):
        # 检查输入是否为空
        if images.shape[0] == 0:
            print("警告：输入批次为空！")
            return [], [], []

        print(f"\n=== 開始執行 ZCR 前向傳播 (模式: {self.fusion_version}) ===")

        # 确保模型处于评估模式
        self.eval_mode()

        # 提取特征
        global_features, patch_features_list = self.encode_image(images, self.features_list)

        # 检查特征提取是否成功
        if not patch_features_list or len(patch_features_list) == 0:
            print("错误：特征提取失败！")
            return [], [], []
        selected_patch_features = [patch_features_list[i] for i in self.layer_indices_to_use]
        print(f"已从4个可用层中筛选出 {len(selected_patch_features)} 个层用于后续计算。")

        last_layer_patches = patch_features_list[-1]

        print("\n--- 步驟 1.5: 執行圖像級異常評估 (相似度直接映射模式) ---")
        similarities = global_features @ self.text_features.T
        abnormal_similarity = similarities[:, 1]
        image_level_anomaly_scores = (abnormal_similarity + 1) / 2
        print("圖像級異常分數計算完成。")

        visual_anomaly_map = None
        textual_anomaly_map = None
        textual_prob_map = None

        all_visual_maps = []
        all_textual_maps = []
        all_prob_maps = []

        print(f"\n--- 開始執行【多層次】像素級異常評估 (共 {len(patch_features_list)} 層) ---")

        # 文本分支只使用最后一层特征
        if self.fusion_version in ['textual', 'textual_visual']:
            print("  -> 文本分支: 僅使用最後一層特徵進行圖文對比")
            current_blocks = self.fast_block_features(last_layer_patches, r_list=self.r_list)
            scale_textual_map, scale_prob_map = self.calculate_textual_anomaly_score(current_blocks)
            all_textual_maps.append(scale_textual_map)
            all_prob_maps.append(scale_prob_map)

        # 视觉分支使用所有层级特征
        if self.fusion_version in ['visual', 'textual_visual']:
            print("  -> 視覺分支: 使用所有層級特徵進行視覺對比")
            for i, scale_patches in enumerate(selected_patch_features):
                print(f"    - 正在處理第 {i + 1}/{len(selected_patch_features)} 層特徵...")
                current_blocks = self.fast_block_features(scale_patches, r_list=self.r_list)
                scale_visual_map = self._calculate_anomaly_score_mutual_text_guided(current_blocks)
                all_visual_maps.append(scale_visual_map)

        if all_visual_maps:
            visual_anomaly_map = torch.mean(torch.stack(all_visual_maps, dim=0), dim=0)
            print("多層次【視覺】異常圖融合完成。")

        if all_textual_maps:
            textual_anomaly_map = torch.mean(torch.stack(all_textual_maps, dim=0), dim=0)
            print("【文本】異常圖融合完成。")
            textual_prob_map = torch.mean(torch.stack(all_prob_maps, dim=0), dim=0)
            print("【文本】概率圖融合完成。")

        print("\n--- 步驟 4: 融合最終像素級得分 ---")
        if self.fusion_version == 'visual':
            anomaly_map = visual_anomaly_map
        elif self.fusion_version == 'textual':
            anomaly_map = textual_anomaly_map
        elif self.fusion_version == 'textual_visual':
            print("  -> 執行【方案二: 非對稱置信度】加權融合...")
            abnormal_prob_map = textual_prob_map[:, 1, :, :].unsqueeze(1)
            w_text = self.text_confidence_weight_alpha * abnormal_prob_map
            w_visual = 1.0 - w_text
            anomaly_map = w_text * textual_anomaly_map + w_visual * visual_anomaly_map
            print(f"  -> 融合完成。文本权重由 P(abnormal) 驱动，alpha 上限: {self.text_confidence_weight_alpha}")
        else:
            raise ValueError(f"未知的融合模式: {self.fusion_version}")
        print(f"已選擇並執行 [{self.fusion_version}] 像素級模式。")

        # 确保anomaly_map不为空
        if anomaly_map is None:
            print("警告：异常图生成失败！")
            return [], image_level_anomaly_scores.detach().cpu().numpy(), global_features.detach().cpu().numpy()

        # 打印异常图的形状以帮助调试
        print(f"异常图形状 (插值前): {anomaly_map.shape}")

        anomaly_map = F.interpolate(anomaly_map, size=(self.out_size_h, self.out_size_w), mode='bilinear',
                                    align_corners=False)

        # 打印插值后的形状
        print(f"异常图形状 (插值后): {anomaly_map.shape}")

        am_np = anomaly_map.squeeze(1).cpu().numpy()
        pixel_level_results = [am_np[i] for i in range(am_np.shape[0])]

        # 确保有像素级结果输出
        if len(pixel_level_results) == 0:
            print("警告：没有生成像素级异常分数图！")
            return [], image_level_anomaly_scores.detach().cpu().numpy(), global_features.detach().cpu().numpy()

        print(f"像素级结果数量: {len(pixel_level_results)}, 每个形状: {pixel_level_results[0].shape}")

        print("=== 前向傳播完成 ===")
        return pixel_level_results, image_level_anomaly_scores.detach().cpu().numpy(), global_features.detach().cpu().numpy()

    def train_mode(self):
        self.clip_model.train()

    def eval_mode(self):
        self.clip_model.eval()
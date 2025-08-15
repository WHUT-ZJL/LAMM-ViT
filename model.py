import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MaskProcessor(nn.Module):
    """Processes region masks, projecting and combining them for attention heads."""

    def __init__(self, patch_size=16, img_size=224, num_heads=12):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # Project each of the 8 region masks to the patch dimension
        self.mask_projection = nn.ModuleList([
            nn.Linear(self.num_patches, self.num_patches) for _ in range(8)
        ])
        # Learnable weights for combining masks for the last 4 heads
        self.combine_weights_eye_forehead = nn.Parameter(torch.ones(3))
        self.combine_weights_mouth_chin = nn.Parameter(torch.ones(3))
        self.combine_weights_nose_contour = nn.Parameter(torch.ones(2))
        self.combine_weights_all = nn.Parameter(torch.ones(8))

    def forward(self, masks):
        # Input masks are [B, 8, 14, 14]
        B = masks.shape[0]
        masks_flat = masks.view(B, 8, -1)  # [B, 8, 196]

        projected_masks = [proj(masks_flat[:, i]) for i, proj in enumerate(self.mask_projection)]

        # Head 9: Eyes (0, 1) + Eyebrows/Forehead (7)
        mask_9 = (self.combine_weights_eye_forehead[0] * projected_masks[0] +
                  self.combine_weights_eye_forehead[1] * projected_masks[1] +
                  self.combine_weights_eye_forehead[2] * projected_masks[7])

        # Head 10: Mouth area (3, 4, 5)
        mask_10 = (self.combine_weights_mouth_chin[0] * projected_masks[3] +
                   self.combine_weights_mouth_chin[1] * projected_masks[4] +
                   self.combine_weights_mouth_chin[2] * projected_masks[5])

        # Head 11: Nose (2) + Face Contour (6)
        mask_11 = (self.combine_weights_nose_contour[0] * projected_masks[2] +
                   self.combine_weights_nose_contour[1] * projected_masks[6])

        # Head 12: Weighted sum of all regions
        weights_all = F.softmax(self.combine_weights_all, dim=0)
        mask_12 = sum(w * m for w, m in zip(weights_all, projected_masks))

        all_masks = torch.stack(projected_masks + [mask_9, mask_10, mask_11, mask_12], dim=1)  # [B, 12, 196]
        return all_masks


class LAMM(nn.Module):
    """Layer-Aware Mask Modulation module."""

    def __init__(self, dim, num_layers, num_heads=12, embed_dim=64):
        super().__init__()
        self.num_heads = num_heads

        # Layer-specific context generation
        self.layer_embeddings = nn.Parameter(torch.randn(num_layers, embed_dim))
        self.global_feature_fusion = nn.Linear(dim * 2, dim)
        self.context_mlp = nn.Sequential(nn.Linear(dim + embed_dim, dim), nn.GELU(), nn.LayerNorm(dim))

        # Region importance analysis
        self.region_importance_mlp = nn.Sequential(nn.Linear(dim + num_heads, 64), nn.GELU(), nn.Linear(64, num_heads))

        # Memory control unit
        self.memory_gate = nn.Linear(dim, 1)  # beta
        self.new_info_gate = nn.Linear(dim, 1)  # gamma

        # Mask parameter generator
        self.lambda_generator = nn.Sequential(nn.Linear(dim + num_heads, 32), nn.Tanh(), nn.Linear(32, num_heads))
        self.theta_generator = nn.Sequential(nn.Linear(dim + num_heads, 32), nn.Tanh(), nn.Linear(32, num_heads))
        self.base_theta = nn.Parameter(torch.zeros(1, num_heads))

    def forward(self, x, layer_idx, prev_mask_weights=None):
        B, N, D = x.shape

        # 1. Layer Context Encoding
        layer_embed = self.layer_embeddings[layer_idx].unsqueeze(0).expand(B, -1)
        global_features = self.global_feature_fusion(torch.cat([x.mean(dim=1), x.max(dim=1)[0]], dim=-1))
        context_vector = self.context_mlp(torch.cat([global_features, layer_embed], dim=-1))

        # 2. Region Importance Analysis
        if prev_mask_weights is None:
            prev_mask_weights = torch.ones(B, self.num_heads, device=x.device) / self.num_heads

        current_importance = self.region_importance_mlp(torch.cat([context_vector, prev_mask_weights], dim=-1))
        current_importance = F.softmax(current_importance, dim=-1)  # alpha

        # Memory Control Unit
        beta = torch.sigmoid(self.memory_gate(context_vector))
        gamma = torch.sigmoid(self.new_info_gate(context_vector))
        mask_weights = gamma * current_importance + beta * prev_mask_weights

        # 3. Mask Parameter Generator
        param_input = torch.cat([context_vector, mask_weights], dim=-1)
        lambda_r = F.softplus(self.lambda_generator(param_input))  # Ensure positivity
        theta_r = self.base_theta + torch.tanh(self.theta_generator(param_input))  # Allow range around base

        return mask_weights, lambda_r, theta_r


class RegionGatedMultiHeadAttention(nn.Module):
    """Region-Guided Multi-Head Attention with dynamic gating."""

    def __init__(self, dim, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, masks, lambda_r, theta_r, mask_weights):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Construct attention gate from masks
        gate_mask = masks.unsqueeze(-1) * masks.unsqueeze(-2)  # [B, H, N, N]
        gate = torch.sigmoid(lambda_r.view(B, -1, 1, 1) * (gate_mask - theta_r.view(B, -1, 1, 1)))

        # Modulate attention scores
        attn = attn * gate
        attn = self.dropout(F.softmax(attn, dim=-1))

        # Apply head-specific weights from LAMM
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = x * mask_weights.view(B, 1, self.num_heads).repeat_interleave(self.head_dim, dim=-1)

        return self.proj(x)


class HeadInteractionAttention(nn.Module):
    """Simple self-attention mechanism between head representations."""

    def __init__(self, dim, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_heads = nn.Linear(self.head_dim, self.head_dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        x_heads = x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, D_h]

        # Global representation for each head
        head_global_repr = x_heads.mean(dim=2)  # [B, H, D_h]

        qkv = self.qkv_heads(head_global_repr).reshape(B, self.num_heads, 3, self.head_dim).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, D_h]

        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        interacted_heads = (attn @ v).unsqueeze(2).expand_as(x_heads)  # [B, H, N, D_h]

        x = x + self.proj(interacted_heads.permute(0, 2, 1, 3).reshape(B, N, D))
        return x


class MaskedTransformerBlock(nn.Module):
    """Transformer Block integrating RG-MHA and LAMM."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RegionGatedMultiHeadAttention(dim, num_heads, dropout)
        self.head_interaction = HeadInteractionAttention(dim, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout)
        )

    def forward(self, x, masks, lambda_r, theta_r, mask_weights):
        # First residual connection
        attn_out = self.attn(self.norm1(x), masks, lambda_r, theta_r, mask_weights)
        x = x + self.head_interaction(attn_out)

        # Second residual connection
        x = x + self.mlp(self.norm2(x))
        return x, mask_weights


class MaskGuidedVisionTransformer(nn.Module):
    """The main LAMM-ViT model."""

    def __init__(self, num_classes=2, dim=768, num_layers=12, num_heads=12, mlp_dim=3072, patch_size=16, img_size=224):
        super().__init__()
        self.num_layers = num_layers
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))

        self.mask_processor = MaskProcessor(patch_size, img_size, num_heads)
        self.lamm = LAMM(dim, num_layers, num_heads)

        self.blocks = nn.ModuleList([MaskedTransformerBlock(dim, num_heads, mlp_dim / dim) for _ in range(num_layers)])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, region_masks):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed

        # Process masks to match patch sequence and add a zero mask for the CLS token
        processed_masks = self.mask_processor(region_masks)  # [B, 12, 196]
        cls_mask = torch.zeros(B, processed_masks.shape[1], 1, device=x.device)
        extended_masks = torch.cat([cls_mask, processed_masks], dim=2)  # [B, 12, 197]

        prev_mask_weights = None
        all_mask_weights = []

        for i, block in enumerate(self.blocks):
            mask_weights, lambda_r, theta_r = self.lamm(x, i, prev_mask_weights)
            x, updated_mask_weights = block(x, extended_masks, lambda_r, theta_r, mask_weights)
            prev_mask_weights = updated_mask_weights

            # Store weights from the last few layers for the diversity loss
            if i >= self.num_layers - 3:
                all_mask_weights.append(updated_mask_weights)

        cls_token_final = self.norm(x)[:, 0]
        logits = self.head(cls_token_final)

        return {'logits': logits, 'mask_weights': all_mask_weights}
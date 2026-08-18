"""
Multimodal Vision-Language-Embodiment Backbone: PatchVisionEncoder, TextLanguageEncoder & Cross-Attention
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any, Union
from .modules.embodiment_encoder import EmbodimentEmbedding, FiLMLayer


class PatchVisionEncoder(nn.Module):
    """
    SigLIP / ViT-style Patch Vision Transformer Encoder:
    - Bilinear adaptive interpolation to 224x224
    - 16x16 Conv2D Patch projection (196 patches)
    - Learnable [CLS] token and 1D Position Embeddings
    - Multi-layer Transformer with Pre-LayerNorm
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # Ensure num_heads divides embed_dim
        if embed_dim % num_heads != 0:
            num_heads = 12 if embed_dim % 12 == 0 else (16 if embed_dim % 16 == 0 else 8)
        self.num_heads = num_heads
        self.grid_size = image_size // patch_size  # 14
        self.num_patches = self.grid_size * self.grid_size  # 196

        self.patch_proj = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=self.num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.kaiming_normal_(self.patch_proj.weight, mode="fan_out")
        if self.patch_proj.bias is not None:
            nn.init.zeros_(self.patch_proj.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Automatic dtype conversion: convert uint8 (unsigned char) to float normalized to [0.0, 1.0]
        if x.dtype == torch.uint8:
            x = x.to(dtype=self.patch_proj.weight.dtype) / 255.0
        elif not torch.is_floating_point(x):
            x = x.to(dtype=self.patch_proj.weight.dtype)
        elif x.dtype != self.patch_proj.weight.dtype:
            x = x.to(dtype=self.patch_proj.weight.dtype)

        # Automatic shape handling: (C, H, W) or (H, W, C) -> (1, C, H, W); (B, H, W, 3) -> (B, 3, H, W)
        if x.ndim == 3:
            if x.shape[0] == 3:
                x = x.unsqueeze(0)
            elif x.shape[-1] == 3:
                x = x.permute(2, 0, 1).unsqueeze(0)
        elif x.ndim == 4 and x.shape[-1] == 3 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)

        B, C, H, W = x.shape
        if H != self.image_size or W != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        patches = self.patch_proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)
        tokens = tokens + self.pos_embed

        encoded = self.transformer(tokens)
        encoded = self.norm(encoded)

        global_feat = encoded[:, 0]
        patch_feats = encoded[:, 1:]
        return global_feat, patch_feats


class TextLanguageEncoder(nn.Module):
    """
    Qwen3-VL style Text Transformer Encoder with learned word & positional embeddings.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 768,
        max_length: int = 32,
        num_heads: int = 12,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        if embed_dim % num_heads != 0:
            num_heads = 12 if embed_dim % 12 == 0 else (16 if embed_dim % 16 == 0 else 8)
        self.num_heads = num_heads

        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.pos_embeddings = nn.Parameter(torch.zeros(1, max_length, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=self.num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.word_embeddings.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_embeddings, std=0.02)

    @staticmethod
    def _deterministic_hash(s: str) -> int:
        """Deterministic 32-bit djb2 hash independent of Python session seed."""
        h = 5381
        for c in s:
            h = ((h << 5) + h + ord(c)) & 0xFFFFFFFF
        return h

    def simple_tokenize(self, text_list: List[str], device: torch.device) -> torch.Tensor:
        B = len(text_list)
        token_ids = torch.zeros((B, self.max_length), dtype=torch.long, device=device)
        for i, text in enumerate(text_list):
            words = text.lower().replace(",", " ").replace(".", " ").split()
            for j, w in enumerate(words[: self.max_length]):
                hash_id = (self._deterministic_hash(w) % (self.vocab_size - 2)) + 1
                token_ids[i, j] = hash_id
        return token_ids

    def forward(self, text_inputs: Any, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]

        if isinstance(text_inputs, (list, tuple)):
            dev = device if device is not None else next(self.parameters()).device
            input_ids = self.simple_tokenize(list(text_inputs), dev)
        elif isinstance(text_inputs, torch.Tensor):
            input_ids = text_inputs
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            if input_ids.dtype != torch.long:
                input_ids = input_ids.long()
            if device is not None and input_ids.device != device:
                input_ids = input_ids.to(device)
        else:
            raise TypeError(f"Unsupported text_inputs type: {type(text_inputs)}")

        B, L = input_ids.shape
        L = min(L, self.max_length)
        input_ids = input_ids[:, :L]

        embeds = self.word_embeddings(input_ids) + self.pos_embeddings[:, :L, :]
        encoded = self.transformer(embeds)
        encoded = self.norm(encoded)

        global_text_feat = encoded.mean(dim=1)
        token_feats = encoded
        return global_text_feat, token_feats


class MultiModalCrossAttentionBackbone(nn.Module):
    """
    Multimodal & Multi-Embodiment Fusion Backbone:
    Cross-attends Vision Patch tokens with Text tokens, conditioned on Embodiment ID.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        vision_num_heads: int = 12,
        vision_num_layers: int = 4,
        text_num_heads: int = 12,
        text_num_layers: int = 4,
        cross_attn_heads: int = 12,
        image_size: int = 224,
        patch_size: int = 16,
        num_embodiments: int = 3,
        embodiment_embed_dim: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        if embed_dim % cross_attn_heads != 0:
            cross_attn_heads = 12 if embed_dim % 12 == 0 else (16 if embed_dim % 16 == 0 else 8)
        self.cross_attn_heads = cross_attn_heads
        self.grid_size = image_size // patch_size  # 14
        self.embodiment_embed_dim = embodiment_embed_dim

        self.vision_encoder = PatchVisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=vision_num_heads,
            num_layers=vision_num_layers,
        )
        self.text_encoder = TextLanguageEncoder(
            embed_dim=embed_dim,
            num_heads=text_num_heads,
            num_layers=text_num_layers,
        )
        self.embodiment_encoder = EmbodimentEmbedding(num_embodiments=num_embodiments, embed_dim=embodiment_embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=self.cross_attn_heads,
            batch_first=True,
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + embodiment_embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        obs_image: torch.Tensor,
        instruction: Union[List[str], torch.Tensor],
        embodiment_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if obs_image.ndim == 3:
            if obs_image.shape[0] == 3:
                obs_image = obs_image.unsqueeze(0)
            elif obs_image.shape[-1] == 3:
                obs_image = obs_image.permute(2, 0, 1).unsqueeze(0)
        elif obs_image.ndim == 4 and obs_image.shape[-1] == 3 and obs_image.shape[1] != 3:
            obs_image = obs_image.permute(0, 3, 1, 2)

        B = obs_image.shape[0]
        device = obs_image.device

        # Default embodiment 0 (WidowX) if none provided
        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)
        elif embodiment_id.device != device:
            embodiment_id = embodiment_id.to(device)

        global_img, patch_feats = self.vision_encoder(obs_image)
        global_text, text_tokens = self.text_encoder(instruction, device=device)
        emb_feat = self.embodiment_encoder(embodiment_id)  # (B, 128)

        # 1. Vision-to-Text Fusion (for context vectors)
        # Query: Patches (196), Key: Text (32). Softmax is over Text.
        fused_patches, _ = self.cross_attn(
            query=patch_feats,
            key=text_tokens,
            value=text_tokens,
            need_weights=False,
        )

        # 2. Text-to-Vision Affordance Extraction
        # Query: Global Text (1), Key: Patches (196). Softmax is over Patches (196).
        # This provides a mathematically valid spatial probability distribution over the image!
        _, text_to_vision_attn = self.cross_attn(
            query=global_text.unsqueeze(1),
            key=patch_feats,
            value=patch_feats,
            need_weights=True,
            average_attn_weights=True,  # Averages across all attention heads automatically
        )
        # text_to_vision_attn shape: (B, 1, 196)
        
        # Safe normalization to spatial map
        mean_attn = text_to_vision_attn.squeeze(1)  # (B, 196)
        mean_attn = mean_attn / (mean_attn.sum(dim=-1, keepdim=True) + 1e-8)
        spatial_attn = mean_attn.view(-1, 1, self.grid_size, self.grid_size)

        pooled_fused = torch.nan_to_num(fused_patches.mean(dim=1), nan=0.0)
        global_img = torch.nan_to_num(global_img, nan=0.0)
        fused_cat = torch.cat([global_img, pooled_fused, emb_feat], dim=-1)
        context_c = self.fusion_mlp(fused_cat)
        context_c = torch.nan_to_num(context_c, nan=0.0, posinf=5.0, neginf=-5.0)

        return {
            "context_c": context_c,
            "spatial_attention": spatial_attn,
            "global_img_feat": global_img,
            "global_text_feat": global_text,
            "patch_feats": patch_feats,
            "embodiment_feat": emb_feat,
        }

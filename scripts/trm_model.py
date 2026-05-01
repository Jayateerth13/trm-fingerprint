"""
SiamViTRM — Siamese Vision Tiny Recursive Model for Fingerprint Verification.

Architecture
------------
Adapts the Vision Tiny Recursion Model (ViTRM, Akazan et al. 2025) as a shared
encoder inside a Siamese network.  Both fingerprint images in a pair pass through
the same encoder; cosine similarity between their final prediction tokens gives a
verification score.

Recursive encoder design (per image):
  x  — image tokens from patch embedding (fixed throughout recursion)
  y  — prediction token  (B, 1, d): distils spatial info into a match-relevant vector
  z  — latent memory     (B, K, d): stores intermediate spatial reasoning

At each recursion cycle (shared Transformer block, weight-tied):
  1. z-update  (n_latent_steps times):  z ← block([x, y, z])[-K:]
  2. y-update  (once):                  y ← block([y, z])[:1]

    T_recursion cycles all run WITH gradient by default (T_recursion=1 recommended).
    When T_recursion > 1, the first T-1 cycles run without gradient (memory-saving warmup)
    and the final cycle runs with full gradient.  Note: with T_recursion > 1, y_init and
    z_init receive no gradient because they are only used during the no-grad warmup;
    set T_recursion=1 to train those parameters via gradient descent.

Training uses deep supervision with N_supervision independent weight updates per
batch (one per unrolling step).  An Exponential Moving Average (EMA) of weights
is maintained for evaluation.

References
----------
  TRM   : Jolicoeur-Martineau (2025)  arXiv:2510.04871
  ViTRM : Akazan et al.       (2025)  arXiv:2603.19503
"""
from __future__ import annotations

import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Primitives
# ──────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization — no bias, no mean subtraction."""

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * (x / rms)


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward block (no bias).
    hidden_dim = round_up_32((2/3) * 4 * d) following LLaMA / TRM convention.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        hidden = max(32, int(2 / 3 * 4 * d))
        hidden = ((hidden + 31) // 32) * 32   # nearest multiple of 32
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TRMLayer(nn.Module):
    """
    One Transformer layer: RMSNorm → MHA → residual → RMSNorm → SwiGLU → residual.
    All linear projections are bias-free (following TRM design).
    """

    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn  = nn.MultiheadAttention(d, n_heads, bias=False, batch_first=True)
        self.norm2 = RMSNorm(d)
        self.ffn   = SwiGLU(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Patch Embedding
# ──────────────────────────────────────────────────────────────────────────────

class FingerprintPatchEmbed(nn.Module):
    """
    Splits a single-channel fingerprint image into non-overlapping P×P patches
    and linearly projects each flattened patch to d_model dimensions.
    A learnable positional embedding is added to each token.

    Input : (B, 1, img_size, img_size)
    Output: (B, n_patches, d_model)   where n_patches = (img_size // P)²
    """

    def __init__(
        self,
        img_size: int  = 128,
        patch_size: int = 16,
        d_model: int   = 128,
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.n_patches  = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size          # grayscale: C = 1

        self.proj      = nn.Linear(patch_dim, d_model, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        P = self.patch_size
        # (B, 1, H, W) → (B, H/P, W/P, P, P, 1) → (B, n_patches, P*P)
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, P * P * C)
        return self.proj(x) + self.pos_embed


# ──────────────────────────────────────────────────────────────────────────────
# ViTRM Encoder
# ──────────────────────────────────────────────────────────────────────────────

class ViTRMEncoder(nn.Module):
    """
    Recursive encoder that progressively refines two internal states:
      y — prediction token (B, 1, d)
      z — latent memory    (B, K, d)

    The SAME n_blocks-layer Transformer block is reused at every recursion step,
    providing parameter efficiency through weight sharing (TRM / ViTRM philosophy).

    One recursion cycle:
      for _ in range(n_latent_steps):
          z ← block( cat([x, y, z]) )[-K:]    # z queries image + prediction
      y ← block( cat([y, z]) )[:1]            # y reads from z only (no direct x)

    T_recursion-1 cycles run without gradient (warm-up from learned init).
    The final cycle runs with full gradient through all n_latent_steps+1 calls.
    """

    def __init__(
        self,
        d_model: int        = 128,
        n_heads: int        = 4,
        n_blocks: int       = 3,
        K: int              = 16,
        n_latent_steps: int = 3,
        T_recursion: int    = 3,
    ) -> None:
        super().__init__()
        self.K              = K
        self.n_latent_steps = n_latent_steps
        self.T_recursion    = T_recursion

        # Shared block — the same weights are used for every z-update and y-update
        self.block = nn.Sequential(
            *[TRMLayer(d_model, n_heads) for _ in range(n_blocks)]
        )

        # Learnable initial states (broadcast over batch at runtime)
        self.y_init = nn.Parameter(torch.zeros(1, 1, d_model))
        self.z_init = nn.Parameter(torch.zeros(1, K, d_model))
        nn.init.trunc_normal_(self.y_init, std=0.02)
        nn.init.trunc_normal_(self.z_init, std=0.02)

    # ------------------------------------------------------------------
    def _step_z(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """z ← block([x, y, z])[-K:]"""
        tokens = torch.cat([x, y, z], dim=1)   # (B, Lx+1+K, d)
        return self.block(tokens)[:, -self.K:]  # last K tokens

    def _step_y(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """y ← block([y, z])[:1]"""
        tokens = torch.cat([y, z], dim=1)       # (B, 1+K, d)
        return self.block(tokens)[:, :1]        # first token

    def _recursion_cycle(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One full cycle: n_latent_steps z-updates then 1 y-update."""
        for _ in range(self.n_latent_steps):
            z = self._step_z(x, y, z)
        y = self._step_y(y, z)
        return y, z

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        x : (B, Lx, d) — fixed image tokens from patch embedding
        y : (B, 1,  d) — previous prediction token; None → use learned y_init
        z : (B, K,  d) — previous latent memory;    None → use learned z_init

        Returns
        -------
        y_det  : (B, 1, d) — detached y for the next supervision step
        z_det  : (B, K, d) — detached z for the next supervision step
        y_grad : (B, 1, d) — y with gradient for computing the loss
        """
        B = x.size(0)
        if y is None:
            y = self.y_init.expand(B, -1, -1)
        if z is None:
            z = self.z_init.expand(B, -1, -1)

        # Optional no-grad warmup cycles (T_recursion > 1 only)
        # When T_recursion=1 (default), all cycles run with full gradient so that
        # y_init and z_init are properly trained via backprop.
        if self.T_recursion > 1:
            with torch.no_grad():
                for _ in range(self.T_recursion - 1):
                    y, z = self._recursion_cycle(x, y, z)

        # Final recursion cycle with full gradient
        y_grad, z_grad = self._recursion_cycle(x, y, z)

        # Detached copies serve as warm init for the next supervision step
        return y_grad.detach(), z_grad.detach(), y_grad


# ──────────────────────────────────────────────────────────────────────────────
# Siamese ViTRM (Verification Model)
# ──────────────────────────────────────────────────────────────────────────────

class SiamViTRM(nn.Module):
    """
    Siamese wrapper around ViTRMEncoder for fingerprint verification.

    Both images share the same patch embedding and encoder (identical weights).
    The verification score is the cosine similarity between their final prediction
    tokens, rescaled to [0, 1].

    A halting head predicts whether the current prediction is already reliable
    (used for early stopping during training; BCE loss target = 1 if score gives
    the correct verification decision, 0 otherwise).
    """

    def __init__(
        self,
        img_size: int       = 128,
        patch_size: int     = 16,
        d_model: int        = 128,
        n_heads: int        = 4,
        n_blocks: int       = 3,
        K: int              = 16,
        n_latent_steps: int = 3,
        T_recursion: int    = 3,
    ) -> None:
        super().__init__()
        self.patch_embed = FingerprintPatchEmbed(img_size, patch_size, d_model)
        self.encoder     = ViTRMEncoder(d_model, n_heads, n_blocks, K, n_latent_steps, T_recursion)
        self.halt_head   = nn.Linear(d_model, 1, bias=False)

    def encode(
        self,
        img: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """img (B,1,H,W) → (y_det, z_det, y_grad)"""
        x = self.patch_embed(img)
        return self.encoder(x, y, z)

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        y1: Optional[torch.Tensor] = None,
        z1: Optional[torch.Tensor] = None,
        y2: Optional[torch.Tensor] = None,
        z2: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Returns
        -------
        score : (B,)  — verification score in [0, 1]   (1 = genuine, 0 = impostor)
        q     : (B,)  — halting confidence in [0, 1]
        y1_det, z1_det, y2_det, z2_det — detached states for next supervision step
        v1, v2 : (B, d) — raw embeddings (prediction tokens) used for verification loss
        """
        y1_det, z1_det, y1_grad = self.encode(img1, y1, z1)
        y2_det, z2_det, y2_grad = self.encode(img2, y2, z2)

        v1 = y1_grad.squeeze(1)                              # (B, d)
        v2 = y2_grad.squeeze(1)                              # (B, d)

        sim   = F.cosine_similarity(v1, v2, dim=1)          # (B,)  in [-1, 1]
        score = (sim + 1.0) / 2.0                           # rescale to [0, 1]

        # Halting: confident when average of the two embeddings signals correctness
        q = torch.sigmoid(self.halt_head((v1 + v2) / 2.0).squeeze(-1))  # (B,)

        return score, q, y1_det, z1_det, y2_det, z2_det, v1, v2


# ──────────────────────────────────────────────────────────────────────────────
# Exponential Moving Average
# ──────────────────────────────────────────────────────────────────────────────

class EMA:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow copy of the model with smoothed weights for evaluation.
    Usage:
        ema = EMA(model, decay=0.999)
        # after each optimizer.step():
        ema.update(model)
        # for evaluation:
        ema.eval_model.eval()
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay     = decay
        self.eval_model = copy.deepcopy(model)
        self.eval_model.eval()
        for p in self.eval_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, p in zip(
            self.eval_model.parameters(), model.parameters()
        ):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# uncertainty.py
"""
Multi-sample variance uncertainty estimation for the Dreamer 4 dynamics model.

The diffusion-based dynamics model starts each prediction from random noise z ~ N(0,1)
and integrates to a prediction via Euler ODE steps. Different noise seeds produce
different predictions. In well-covered regions, predictions converge regardless of seed.
In poorly-covered regions, predictions diverge. This variance is a free, reward-free
uncertainty estimator that requires no architectural changes.
"""
from typing import Dict, Any, Optional

import torch
from torch.amp import autocast

from model import (
    Dynamics, Encoder, Decoder,
    temporal_patchify, pack_bottleneck_to_spatial,
)
from train_dynamics import (
    sample_one_timestep_packed,
    decode_packed_to_frames,
)


# ---------------------------------------------------------------------------
# Unified sampler + scorers (used by curiosity MPC to support multiple signals)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_predictions_for_actions(
    dyn: Dynamics,
    *,
    past_packed: torch.Tensor,                     # (1, t, Sz, Dz)
    candidate_actions: torch.Tensor,               # (K, t+1, A)
    k_max: int,
    sched: Dict[str, Any],
    act_mask: Optional[torch.Tensor] = None,
    tau_ctx: float = 0.1,
    lang_emb: Optional[torch.Tensor] = None,
    n_samples: int = 2,
    use_kv_cache: bool = False,
) -> torch.Tensor:
    """
    Run N independent diffusion samples for each of K candidate action sequences.
    Returns (K, N, Sz, Dz) float32.

    With `use_kv_cache=True` the t context tokens are run through the
    transformer once and reused across all K denoising steps — the same
    optimization as in `train_dynamics.sample_one_timestep_packed`. Default
    off; enable for the per-env-step uncertainty logging in collect_data.py,
    where this is the dominant per-step cost.
    """
    K = candidate_actions.shape[0]
    _, t, Sz, Dz = past_packed.shape

    dtype = next(dyn.parameters()).dtype
    past_packed = past_packed.to(dtype)
    candidate_actions = candidate_actions.to(dtype)

    past_KN = past_packed.expand(K, -1, -1, -1).unsqueeze(1).expand(-1, n_samples, -1, -1, -1)
    past_KN = past_KN.reshape(K * n_samples, t, Sz, Dz)

    T_act, A = candidate_actions.shape[1], candidate_actions.shape[2]
    actions_KN = candidate_actions.unsqueeze(1).expand(-1, n_samples, -1, -1)
    actions_KN = actions_KN.reshape(K * n_samples, T_act, A)

    lang_KN = None if lang_emb is None else lang_emb.expand(K * n_samples, -1)

    predictions = sample_one_timestep_packed(
        dyn,
        past_packed=past_KN,
        k_max=k_max,
        sched=sched,
        actions=actions_KN,
        act_mask=act_mask,
        tau_ctx=tau_ctx,
        lang_emb=lang_KN,
        use_kv_cache=use_kv_cache,
    )  # (K*N, Sz, Dz)
    return predictions.float().reshape(K, n_samples, Sz, Dz)


class CrossSeedScorer:
    """
    Score K candidates by per-element variance across N diffusion seeds — the
    inter-seed denoising-variance predictor (u_s). Raw cross-seed variance is
    motion-invariant only in aggregate, so it confounds with scene motion.
    """

    def score_components(self, predictions_KN: torch.Tensor, z_prev_K: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return the per-candidate score components. Only "u_r_norm" (here the
        inter-seed variance) is returned; the `u_r` and `motion` keys do not
        apply to this scorer."""
        return {
            "u_r_norm": predictions_KN.float().var(dim=1).mean(dim=(1, 2)),  # (K,)
        }

    def __call__(self, predictions_KN: torch.Tensor, z_prev_K: torch.Tensor) -> torch.Tensor:
        # Hot path: CEM scoring. predictions_KN: (K, N, Sz, Dz); z_prev_K unused.
        return self.score_components(predictions_KN, z_prev_K)["u_r_norm"]


class URNormScorer:
    """
    Score K candidates by the tokenizer round-trip residual of the mean predicted
    latent, normalized by the predicted latent-space step motion:

        u_r      = RMS( z_pred - encode(decode(z_pred)) )
        motion   = RMS( z_pred - z_prev )
        u_r_norm = u_r / max(motion, eps)

    This is the motion-normalized tokenizer round-trip residual signal (u_norm)
    and is label-free.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        *,
        patch: int,
        packing_factor: int,
        n_spatial: int,
        H: int,
        W: int,
        C: int = 3,
        motion_eps: float = 1e-3,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.patch = patch
        self.packing_factor = packing_factor
        self.n_spatial = n_spatial
        self.H = H
        self.W = W
        self.C = C
        self.motion_eps = motion_eps

    @torch.no_grad()
    def score_components(self, predictions_KN: torch.Tensor, z_prev_K: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Same compute as __call__, but returns the raw `u_r`, `motion`, and
        the ratio `u_r_norm` separately.

        Returns dict with three (K,)-shaped float tensors:
          - "u_r":      RMS( z_pred - encode(decode(z_pred)) ) — round-trip residual
          - "motion":   RMS( z_pred - z_prev )                 — predicted step
          - "u_r_norm": u_r / max(motion, motion_eps)          — what __call__ returns
        """
        # Collapse over N via mean to get one predicted latent per candidate.
        z_pred_K = predictions_KN.float().mean(dim=1)                        # (K, Sz, Dz)

        motion_K = (z_pred_K - z_prev_K.float()).pow(2).mean(dim=(1, 2)).sqrt()  # (K,)

        enc_dtype = next(self.encoder.parameters()).dtype
        z_pred_in = z_pred_K.unsqueeze(1).to(enc_dtype)                       # (K, 1, Sz, Dz)
        with autocast(device_type=z_pred_K.device.type, dtype=torch.bfloat16):
            frames = decode_packed_to_frames(
                self.decoder,
                z_packed=z_pred_in,
                H=self.H, W=self.W, C=self.C,
                patch=self.patch,
                packing_factor=self.packing_factor,
            )                                                                 # (K, 1, C, H, W)
            patches = temporal_patchify(frames, self.patch)                   # (K, 1, Np, Dp)
            z_recon_btLd, _ = self.encoder(patches)                           # (K, 1, L, D_b)
        z_recon_K = pack_bottleneck_to_spatial(
            z_recon_btLd, n_spatial=self.n_spatial, k=self.packing_factor,
        )[:, 0].float()                                                        # (K, Sz, Dz)

        u_r_K = (z_pred_K - z_recon_K).pow(2).mean(dim=(1, 2)).sqrt()          # (K,)
        u_r_norm_K = u_r_K / motion_K.clamp(min=self.motion_eps)
        return {"u_r": u_r_K, "motion": motion_K, "u_r_norm": u_r_norm_K}

    @torch.no_grad()
    def __call__(self, predictions_KN: torch.Tensor, z_prev_K: torch.Tensor) -> torch.Tensor:
        # Hot path: CEM scoring. Returns just the u_r_norm ratio scalar.
        return self.score_components(predictions_KN, z_prev_K)["u_r_norm"]

# curiosity.py
"""
MPC-based curiosity exploration policy for active data collection.

Uses CEM (Cross-Entropy Method) to plan action sequences that maximize a
pluggable per-candidate "curiosity score", computed from dynamics-model
predictions. The scorer is injected by the caller (collect_data.py), so the
CEM plumbing here is scorer-agnostic; the published curiosity-driven
collection uses the motion-normalized u_r_norm predictor (see uncertainty.py).
"""
from typing import Callable, Dict, Any, Optional

import torch

from model import Dynamics
from train_dynamics import sample_one_timestep_packed
from uncertainty import sample_predictions_for_actions


# A scorer takes (predictions_KN, z_prev_K) and returns (K,) scalar scores.
# predictions_KN: (K, N, Sz, Dz) float
# z_prev_K:       (K, Sz, Dz)    float — last observed latent per candidate
#                                        (used for motion normalization)
ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@torch.no_grad()
def curiosity_mpc_action(
    dyn: Dynamics,
    *,
    past_packed: torch.Tensor,                     # (1, t, Sz, Dz)
    past_actions: torch.Tensor,                    # (1, t, A) — actions aligned to past frames
    scorer: ScoreFn,
    k_max: int,
    sched: Dict[str, Any],
    act_mask: Optional[torch.Tensor] = None,       # (A,) active action dimensions
    lang_emb: Optional[torch.Tensor] = None,       # (1, lang_dim)
    tau_ctx: float = 0.1,
    n_candidates: int = 256,
    horizon: int = 1,
    n_samples: int = 4,
    n_elite: int = 32,
    n_cem_iters: int = 4,
    cem_init_std: float = 1.0,
    cem_min_std: float = 0.05,
    action_dim: int = 16,
    device: Optional[torch.device] = None,
    use_kv_cache: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Select an action (or action sequence) that maximizes the given per-candidate
    score via CEM.

    Returns dict with:
      "action":           (A,)           — first action of the chosen plan in [-1, 1]
      "action_sequence":  (H, A)         — full plan (only present for horizon > 1)
      "uncertainty":      scalar         — score of the chosen plan
      "mean_uncertainty": scalar         — mean score across the last CEM iter's candidates
      "max_uncertainty":  scalar         — max score across the last CEM iter's candidates
    ("uncertainty" is the generic name for the injected scorer's signal.)
    """
    if device is None:
        device = past_packed.device

    dtype = next(dyn.parameters()).dtype
    past_packed = past_packed.to(dtype)
    past_actions = past_actions.to(dtype)

    A = action_dim
    if act_mask is not None:
        if act_mask.dim() == 1:
            active_mask = act_mask                     # (A,)
        else:
            active_mask = act_mask[0, 0]               # (A,) from (B, T, A)
    else:
        active_mask = torch.ones(A, device=device)

    if horizon == 1:
        return _cem_single_step(
            dyn=dyn,
            past_packed=past_packed,
            past_actions=past_actions,
            scorer=scorer,
            k_max=k_max, sched=sched,
            act_mask=act_mask, lang_emb=lang_emb, tau_ctx=tau_ctx,
            n_candidates=n_candidates, n_samples=n_samples,
            n_elite=n_elite, n_cem_iters=n_cem_iters,
            init_std=cem_init_std, min_std=cem_min_std,
            action_dim=A, active_mask=active_mask, device=device,
        )
    return _cem_multi_step(
        dyn=dyn,
        past_packed=past_packed,
        past_actions=past_actions,
        scorer=scorer,
        k_max=k_max, sched=sched,
        act_mask=act_mask, lang_emb=lang_emb, tau_ctx=tau_ctx,
        n_candidates=n_candidates, horizon=horizon, n_samples=n_samples,
        n_elite=n_elite, n_cem_iters=n_cem_iters,
        init_std=cem_init_std, min_std=cem_min_std,
        action_dim=A, active_mask=active_mask, device=device,
        use_kv_cache=use_kv_cache,
    )


@torch.no_grad()
def _cem_single_step(
    dyn, past_packed, past_actions, scorer,
    k_max, sched, act_mask, lang_emb, tau_ctx,
    n_candidates, n_samples, n_elite, n_cem_iters,
    init_std, min_std,
    action_dim, active_mask, device,
) -> Dict[str, torch.Tensor]:
    # Gaussian CEM (same scheme as plan_cem.py): sample (mu + std * noise).clamp(-1, 1),
    # refit mean/std on elites, floor std at min_std.
    mu = torch.zeros(action_dim, device=device)
    sigma = torch.full((action_dim,), float(init_std), device=device)

    best_action = None
    best_score = torch.tensor(-float("inf"), device=device)

    # Motion reference: last observed latent, broadcast to K at score time.
    z_last = past_packed[0, -1].float()  # (Sz, Dz)

    for cem_iter in range(n_cem_iters):
        noise = torch.randn(n_candidates, action_dim, device=device)
        candidate_acts = (mu + sigma * noise).clamp(-1, 1)
        candidate_acts = candidate_acts * active_mask.unsqueeze(0)

        past_acts_K = past_actions.expand(n_candidates, -1, -1)                # (K, t, A)
        new_act = candidate_acts.unsqueeze(1)                                   # (K, 1, A)
        full_actions = torch.cat([past_acts_K, new_act], dim=1)                # (K, t+1, A)

        predictions = sample_predictions_for_actions(
            dyn,
            past_packed=past_packed,
            candidate_actions=full_actions,
            k_max=k_max, sched=sched,
            act_mask=act_mask, tau_ctx=tau_ctx,
            lang_emb=lang_emb, n_samples=n_samples,
        )                                                                       # (K, N, Sz, Dz)
        z_prev_K = z_last.unsqueeze(0).expand(n_candidates, -1, -1)            # (K, Sz, Dz)
        scores = scorer(predictions, z_prev_K)                                  # (K,)

        max_idx = scores.argmax()
        if scores[max_idx] > best_score:
            best_score = scores[max_idx]
            best_action = candidate_acts[max_idx]

        _, elite_idx = scores.topk(n_elite)
        elite_acts = candidate_acts[elite_idx]
        mu = elite_acts.mean(dim=0)
        sigma = elite_acts.std(dim=0).clamp_min(float(min_std))

    return {
        "action": best_action,
        "uncertainty": best_score,
        "mean_uncertainty": scores.mean(),
        "max_uncertainty": scores.max(),
    }


@torch.no_grad()
def _cem_multi_step(
    dyn, past_packed, past_actions, scorer,
    k_max, sched, act_mask, lang_emb, tau_ctx,
    n_candidates, horizon, n_samples, n_elite, n_cem_iters,
    init_std, min_std,
    action_dim, active_mask, device,
    use_kv_cache: bool = False,
) -> Dict[str, torch.Tensor]:
    t = past_packed.shape[1]

    # Gaussian CEM (same scheme as plan_cem.py) with per-timestep (mu, std):
    # sample (mu + std * noise).clamp(-1, 1), refit on elites, floor std at min_std.
    mu = torch.zeros(horizon, action_dim, device=device)
    sigma = torch.full((horizon, action_dim), float(init_std), device=device)

    best_action_seq = None
    best_total = torch.tensor(-float("inf"), device=device)

    for cem_iter in range(n_cem_iters):
        noise = torch.randn(n_candidates, horizon, action_dim, device=device)
        cand_seqs = (mu.unsqueeze(0) + sigma.unsqueeze(0) * noise).clamp(-1, 1)
        cand_seqs = cand_seqs * active_mask.unsqueeze(0).unsqueeze(0)

        totals = _rollout_scores(
            dyn=dyn,
            past_packed=past_packed,
            past_actions=past_actions,
            candidate_seqs=cand_seqs,
            scorer=scorer,
            k_max=k_max, sched=sched,
            act_mask=act_mask, lang_emb=lang_emb, tau_ctx=tau_ctx,
            n_samples=n_samples,
            max_ctx=t,
            use_kv_cache=use_kv_cache,
        )                                                                       # (K,)

        max_idx = totals.argmax()
        if totals[max_idx] > best_total:
            best_total = totals[max_idx]
            best_action_seq = cand_seqs[max_idx]

        _, elite_idx = totals.topk(n_elite)
        elite_seqs = cand_seqs[elite_idx]
        mu = elite_seqs.mean(dim=0)
        sigma = elite_seqs.std(dim=0).clamp_min(float(min_std))

    return {
        "action": best_action_seq[0],
        "action_sequence": best_action_seq,
        "uncertainty": best_total,
        "mean_uncertainty": totals.mean(),
        "max_uncertainty": totals.max(),
    }


@torch.no_grad()
def _rollout_scores(
    dyn, past_packed, past_actions, candidate_seqs, scorer,
    k_max, sched, act_mask, lang_emb, tau_ctx, n_samples,
    max_ctx: int = 0,
    use_kv_cache: bool = False,
) -> torch.Tensor:
    """
    Roll out K candidate action sequences through the dynamics model,
    scoring per horizon step and returning an aggregated score per candidate.

    Aggregation: 0.5 * mean_per_step + 0.5 * max_per_step (balances average
    vs. peak score along the plan).
    """
    K, H, A = candidate_seqs.shape
    t = past_packed.shape[1]
    Sz, Dz = past_packed.shape[2], past_packed.shape[3]

    past_K = past_packed.expand(K, -1, -1, -1)                                  # (K, t, Sz, Dz)
    past_acts_K = past_actions.expand(K, -1, -1)                                # (K, t, A)
    lang_K = None if lang_emb is None else lang_emb.expand(K, -1)

    z_history = [past_K[:, i] for i in range(t)]
    act_history = [past_acts_K[:, i] for i in range(t)]

    sum_score = torch.zeros(K, device=past_packed.device)
    max_score = torch.zeros(K, device=past_packed.device)

    for h in range(H):
        act_h = candidate_seqs[:, h]                                            # (K, A)

        if max_ctx > 0 and len(z_history) > max_ctx:
            z_win = z_history[-max_ctx:]
            act_win = act_history[-max_ctx:]
        else:
            z_win = z_history
            act_win = act_history

        ctx_len = len(z_win)
        z_seq = torch.stack(z_win, dim=1)                                       # (K, ctx_len, Sz, Dz)
        acts_seq = torch.stack(act_win + [act_h], dim=1)                        # (K, ctx_len+1, A)

        # Run N samples per candidate in one batch (K*N).
        KN = K * n_samples
        z_KN = z_seq.unsqueeze(1).expand(-1, n_samples, -1, -1, -1).reshape(KN, ctx_len, Sz, Dz)
        acts_KN = acts_seq.unsqueeze(1).expand(-1, n_samples, -1, -1).reshape(KN, ctx_len + 1, A)
        lang_KN = None if lang_K is None else lang_K.unsqueeze(1).expand(-1, n_samples, -1).reshape(KN, -1)

        z_next = sample_one_timestep_packed(
            dyn,
            past_packed=z_KN,
            k_max=k_max, sched=sched,
            actions=acts_KN, act_mask=act_mask, tau_ctx=tau_ctx,
            lang_emb=lang_KN,
            use_kv_cache=use_kv_cache,
        )                                                                       # (K*N, Sz, Dz)
        predictions = z_next.float().reshape(K, n_samples, Sz, Dz)

        z_prev_K = z_win[-1].float()                                            # (K, Sz, Dz)
        score_h = scorer(predictions, z_prev_K)                                 # (K,)
        sum_score = sum_score + score_h
        max_score = torch.max(max_score, score_h)

        # Mean prediction drives AR continuation.
        z_mean = predictions.mean(dim=1).to(past_packed.dtype)                   # (K, Sz, Dz)
        z_history.append(z_mean)
        act_history.append(act_h)

    return 0.5 * (sum_score / max(H, 1)) + 0.5 * max_score

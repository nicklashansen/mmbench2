# task_set.py

# 200 pretraining tasks (the 10 held-out unseen tasks are in UNSEEN_TASK_SET below)
TASK_SET = [
    # dmcontrol (21 tasks)
    'walker-stand', 'walker-walk', 'walker-run', 'cheetah-run', 'reacher-easy',
    'reacher-hard', 'acrobot-swingup', 'pendulum-swingup', 'cartpole-balance', 'cartpole-balance-sparse',
    'cartpole-swingup', 'cartpole-swingup-sparse', 'cup-catch', 'finger-spin', 'finger-turn-easy',
    'finger-turn-hard', 'fish-swim', 'hopper-stand', 'hopper-hop', 'quadruped-walk',
    'quadruped-run',
    # dmcontrol-ext (16 tasks)
    'walker-walk-backward', 'walker-run-backward', 'cheetah-run-backward', 'cheetah-run-front', 'cheetah-run-back',
    'cheetah-jump', 'hopper-hop-backward', 'reacher-three-easy', 'reacher-three-hard', 'cup-spin',
    'pendulum-spin', 'jumper-jump', 'spinner-spin', 'spinner-spin-backward', 'spinner-jump',
    'giraffe-run',
    # meta-world (49 tasks)
    'mw-assembly', 'mw-basketball', 'mw-button-press-topdown', 'mw-button-press-topdown-wall', 'mw-button-press',
    'mw-button-press-wall', 'mw-coffee-button', 'mw-coffee-pull', 'mw-coffee-push', 'mw-dial-turn',
    'mw-disassemble', 'mw-door-open', 'mw-door-close', 'mw-drawer-close', 'mw-drawer-open',
    'mw-faucet-open', 'mw-faucet-close', 'mw-hammer', 'mw-handle-press-side', 'mw-handle-press',
    'mw-handle-pull-side', 'mw-handle-pull', 'mw-lever-pull', 'mw-peg-insert-side', 'mw-peg-unplug-side',
    'mw-pick-out-of-hole', 'mw-pick-place', 'mw-pick-place-wall', 'mw-plate-slide', 'mw-plate-slide-side',
    'mw-plate-slide-back', 'mw-plate-slide-back-side', 'mw-push-back', 'mw-push', 'mw-push-wall',
    'mw-reach', 'mw-reach-wall', 'mw-soccer', 'mw-stick-push', 'mw-stick-pull', 
    'mw-sweep-into', 'mw-sweep', 'mw-window-open', 'mw-window-close', 'mw-bin-picking',
    'mw-box-close', 'mw-door-lock', 'mw-door-unlock', 'mw-hand-insert',
    # maniskill (36 tasks)
    'ms-ant-walk', 'ms-ant-run', 'ms-cartpole-balance', 'ms-cartpole-swingup', 'ms-hopper-stand',
    'ms-hopper-hop', 'ms-pick-cube', 'ms-pick-cube-eepose', 'ms-pick-cube-so', 'ms-poke-cube',
    'ms-push-cube', 'ms-pull-cube', 'ms-pull-cube-tool', 'ms-stack-cube', 'ms-place-sphere',
    'ms-lift-peg', 'ms-pick-apple', 'ms-pick-banana', 'ms-pick-can', 'ms-pick-hammer',
    'ms-pick-fork', 'ms-pick-knife', 'ms-pick-mug', 'ms-pick-orange', 'ms-pick-screwdriver',
    'ms-pick-spoon', 'ms-pick-tennis-ball', 'ms-pick-baseball', 'ms-pick-cube-xarm6', 'ms-pick-sponge',
    'ms-anymal-reach', 'ms-reach', 'ms-reach-eepose', 'ms-reach-xarm6', 'ms-cartpole-balance-sparse',
    'ms-cartpole-swingup-sparse',
    # mujoco (6 tasks)
    'mujoco-ant', 'mujoco-halfcheetah', 'mujoco-hopper', 'mujoco-inverted-pendulum', 'mujoco-reacher',
    'mujoco-walker',
    # box2d (8 tasks)
    'bipedal-walker-flat', 'bipedal-walker-uneven', 'bipedal-walker-rugged', 'bipedal-walker-hills', 'bipedal-walker-obstacles',
    'lunarlander-land', 'lunarlander-hover', 'lunarlander-takeoff',
    # robodesk (6 tasks)
    'rd-push-red', 'rd-push-green', 'rd-push-blue', 'rd-open-slide', 'rd-open-drawer',
    'rd-flat-block-in-bin',
    # ogbench (12 tasks)
    'og-ant', 'og-antball', 'og-point-arena', 'og-point-maze', 'og-point-bottleneck',
    'og-point-circle', 'og-point-spiral', 'og-ant-arena', 'og-ant-maze', 'og-ant-bottleneck',
    'og-ant-circle', 'og-ant-spiral',
    # pygame (19 tasks)
    'pygame-cowboy', 'pygame-coinrun', 'pygame-spaceship', 'pygame-pong', 'pygame-bird-attack',
    'pygame-highway', 'pygame-landing', 'pygame-air-hockey', 'pygame-rocket-collect', 'pygame-chase-evade',
    'pygame-coconut-dodge', 'pygame-cartpole-balance', 'pygame-cartpole-swingup', 'pygame-cartpole-balance-sparse', 'pygame-cartpole-swingup-sparse',
    'pygame-cartpole-tremor', 'pygame-point-maze-var1', 'pygame-point-maze-var2', 'pygame-point-maze-var3',
    # atari (27 tasks)
    'atari-alien', 'atari-assault', 'atari-asterix', 'atari-atlantis', 'atari-bank-heist',
    'atari-battle-zone', 'atari-beamrider', 'atari-boxing', 'atari-chopper-command', 'atari-crazy-climber',
    'atari-double-dunk', 'atari-gopher', 'atari-ice-hockey', 'atari-jamesbond', 'atari-kangaroo',
    'atari-krull', 'atari-ms-pacman', 'atari-name-this-game', 'atari-phoenix', 'atari-pong',
    'atari-road-runner', 'atari-robotank', 'atari-seaquest', 'atari-space-invaders', 'atari-tutankham',
    'atari-upndown', 'atari-yars-revenge',
]


# Domain spans in TASK_SET ordering.  Task-set authoritatively defines the
# 10 domains below; keep in sync with the task blocks above.
DOMAIN_SPANS: list[tuple[str, int, int]] = [
    ('dmcontrol',      0,  21),
    ('dmcontrol-ext', 21,  37),
    ('metaworld',     37,  86),
    ('maniskill',     86, 122),
    ('mujoco',       122, 128),
    ('box2d',        128, 136),
    ('robodesk',     136, 142),
    ('ogbench',      142, 154),
    ('pygame',       154, 173),
    ('atari',        173, 200),
]
DOMAINS: list[str] = [name for name, _, _ in DOMAIN_SPANS]

_TASK_TO_DOMAIN: dict[str, str] = {}
for _name, _lo, _hi in DOMAIN_SPANS:
    for _t in TASK_SET[_lo:_hi]:
        _TASK_TO_DOMAIN[_t] = _name


def task_to_domain(task: str) -> str:
    """Map a task name to its canonical domain string.

    Covers TASK_SET ∪ UNSEEN_TASK_SET. Raises KeyError otherwise — the
    caller should fall back to a default domain or update the table if
    this fires.
    """
    return _TASK_TO_DOMAIN[task]


# ---------------------------------------------------------------------------
# Seen / unseen task sets for the targeted-data-collection evaluation.
#   - SEEN_TASK_SET: subset of TASK_SET (tasks the world model trained on).
#   - UNSEEN_TASK_SET: tasks defined in envs/*.py but NOT in TASK_SET (also
#     enumerated as keys in interactive.TEST_TASK_SET).
# Each matched UNSEEN entry is the in-domain analog of a SEEN entry (same
# domain / control morphology), since out-of-domain tasks transfer poorly
# zero-shot.
#
# IMPORTANT: three matched pairs share IDENTICAL CLIP text embeddings
# (cosine = 1.0000) by design: cup-catch ↔ cup-catch-var1, finger-turn-easy ↔
# finger-turn-easy-var1, pygame-point-maze-var1 ↔ pygame-point-maze-var4. The
# model thus has no language signal to disambiguate them, so any UNSEEN-partner
# gain is attributable to visual adaptation, not language conditioning (and
# such pairs are not "true zero-shot" in the language sense). The three pygame
# tasks dungeon-explorer1/foraging/whirlpool have no SEEN partner.
# ---------------------------------------------------------------------------

SEEN_TASK_SET: list[str] = [
    'cup-catch',                # dmcontrol
    'finger-turn-easy',         # dmcontrol
    'mw-push',                  # metaworld
    'ms-push-cube',             # maniskill
    'lunarlander-hover',        # box2d
    'og-point-maze',            # ogbench
    'og-point-bottleneck',      # ogbench
    'pygame-point-maze-var1',   # pygame
    'pygame-pong',              # pygame
    'pygame-bird-attack',       # pygame
]

UNSEEN_TASK_SET: list[str] = [
    # Matched-pair UNSEEN: each entry has a SEEN partner with shared visual style.
    'cup-catch-var1',           # dmcontrol; SEEN partner: cup-catch        (lang cosine 1.0000)
    'finger-turn-easy-var1',    # dmcontrol; SEEN partner: finger-turn-easy (lang cosine 1.0000)
    'ms-push-banana',           # maniskill; SEEN partner: ms-push-cube     (lang cosine 0.986)
    'og-point-var1',            # ogbench;   SEEN partner: og-point-bottleneck (lang cosine 0.992)
    'og-point-var2',            # ogbench;   SEEN partner: og-point-maze       (lang cosine 0.994)
    'pygame-point-maze-var4',   # pygame;    SEEN partner: pygame-point-maze-var1 (lang cosine 1.0000)
    'pygame-reacher-easy',      # pygame;    SEEN partner: reacher-easy (dmcontrol; same instruction)
    # Completely unseen: no SEEN partner.
    'pygame-dungeon-explorer1',
    'pygame-foraging',
    'pygame-whirlpool',
]

# UNSEEN tasks are not part of the contiguous DOMAIN_SPANS over TASK_SET, so
# extend the lookup table explicitly (matched-per-domain home per the
# inline comments above).
_UNSEEN_TASK_TO_DOMAIN: dict[str, str] = {
    'cup-catch-var1':           'dmcontrol',
    'finger-turn-easy-var1':    'dmcontrol',
    'ms-push-banana':           'maniskill',
    'og-point-var1':            'ogbench',
    'og-point-var2':            'ogbench',
    'pygame-point-maze-var4':   'pygame',
    'pygame-reacher-easy':      'pygame',
    'pygame-dungeon-explorer1': 'pygame',
    'pygame-foraging':          'pygame',
    'pygame-whirlpool':         'pygame',
}
_TASK_TO_DOMAIN.update(_UNSEEN_TASK_TO_DOMAIN)


def compute_task_weights(
    tasks: list[str],
    mode: str,
    *,
    targeted_alpha: float = 0.5,
    targeted_tasks: list[str] | None = None,
) -> list[float]:
    """Build per-task sampling weights for WMDataset.

    Modes (all weights are relative; WMDataset normalizes internally):
      - "valid_starts": returns None-equivalent (signals caller to pass
        task_weights=None for legacy P(task) ∝ total valid_starts).
      - "uniform": every task gets weight 1.0 (equal-per-task sampling).
      - "targeted": split total weight `α` uniformly over the focused task
        set and `1-α` uniformly over the rest, biasing sampling toward the
        targeted tasks while retaining the rest. Defaults: `targeted_alpha=0.5`,
        `targeted_tasks=SEEN_TASK_SET ∪ UNSEEN_TASK_SET`.

    The `tasks` list is what the dataset actually loaded (post filter);
    unknown tasks get weight 1.0 under "uniform" (shouldn't happen with
    the canonical TASK_SET).
    """
    if mode == "valid_starts":
        # Sentinel: caller should pass task_weights=None.
        return [1.0] * len(tasks)
    if mode == "uniform":
        return [1.0] * len(tasks)
    if mode == "targeted":
        if not (0.0 <= targeted_alpha <= 1.0):
            raise ValueError(
                f"targeted_alpha must be in [0,1], got {targeted_alpha!r}"
            )
        targeted_set = set(SEEN_TASK_SET + UNSEEN_TASK_SET if targeted_tasks is None else targeted_tasks)
        n_targeted = sum(1 for t in tasks if t in targeted_set)
        n_non = len(tasks) - n_targeted
        # Auto-collapse alpha when one side is empty so we degrade
        # gracefully to "uniform over what's loaded" instead of all-zero
        # weights (which would crash WMDataset's sampler).
        if n_targeted == 0:
            alpha_eff = 0.0
        elif n_non == 0:
            alpha_eff = 1.0
        else:
            alpha_eff = float(targeted_alpha)
        w_targeted = (alpha_eff / n_targeted) if n_targeted > 0 else 0.0
        w_non = ((1.0 - alpha_eff) / n_non) if n_non > 0 else 0.0
        return [w_targeted if t in targeted_set else w_non for t in tasks]
    raise ValueError(f"Unknown task_weighting mode: {mode!r}")

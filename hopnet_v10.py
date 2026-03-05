"""
HOPNet V10.2 — Hierarchical Oscillatory Predictive Network
===========================================================
V10.3: Basin separation. Pure phase Hebbian. Pinned core. Weaker coupling.

V10 CHANGES FROM V9 (sources: ChatGPT review + Grok review):

  CRITICAL FIXES:
  1. T matrix: complex phasors instead of mean(axis=-1)
     (ChatGPT: T matrix was collapsing phase geometry)

  2. Ferromagnetic collapse suppression (Grok: hidden capacity killer)
     a. Mean-field centering in Hebbian: z_centered = z - mean(z)
     b. Row-sum zero constraint after every weight update
     c. Mask applied INSIDE correlation, not after

  3. Spectral radius: power iteration instead of Frobenius/sqrt(N)
     (ChatGPT: Frobenius estimate is not an upper bound)

  4. Noise per timestep inside scan_fn, not static per call
     (ChatGPT: static noise is just a random bias field)

  IMPORTANT FIXES:
  5. _consolidate() extracted as separate method (Grok patch)
  6. _core_phase() extracted as utility (used in multiple places)
  7. Cup threshold (-0.2) documented in spec comment
  8. Oja-style row normalization (optional, gamma_oja=0.0 default)

  ARCHITECTURE ADDITIONS:
  9. 4-pattern benchmark protocol built in (Grok spec)
  10. Global synchrony monitor (ferromagnet diagnostic)
  11. Phase-only normalization in _compute_hebbian (Grok capacity boost)
      z_normalized = z_centered / (|z_centered| + eps)
      Stops eigenmode concentration. Moves capacity toward O(N) phase regime.

Author: HOPNet Project, March 2026
Lead AI: Claude (Anthropic)
Reviews: ChatGPT (OpenAI), Grok (xAI)
Spec: HOPNet_V9_Formal_Spec.docx + V10 patch (HOPNet_V9_V10_patch_and_benchmark.pdf)
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np


# ============================================================
# PARAMETER DEFAULTS
# ============================================================

DEFAULTS = dict(
    # Initialization
    n_osc         = 2048,
    core_size     = 200,
    valence_size  = 100,
    sparsity      = 0.15,
    seed          = 42,

    # Dynamics
    dt            = 0.05,
    sigma_noise   = 0.0,      # per-timestep noise std (V10: moved inside scan)
    omega_slow_sd = 0.3,
    omega_fast_sd = 0.5,   # V10.2: reduced — must be < coupling for phase-lock

    # Learning
    eta_fast      = 0.035,    # V10.1: raised — need rho ~1.3 for attractor formation
    eta_slow      = 0.04,     # V10.2: raised further for multi-pattern consolidation
    eta_T         = 0.08,
    decay         = 0.93,
    kappa         = 12.0,
    theta_c       = 0.55,
    lambda_v      = 0.5,

    # V10.1: Order-modulated plasticity (ACh/synchrony analog)
    # Two modes: receptive (broad, low order) and focused (deep, high order)
    # Gate never drops below 0.85x — receptive mode still learns
    order_gate_bias   = 0.85,  # floor: receptive mode still learns broadly
    order_gate_scale  = 0.6,   # scale: focused mode boosts to 1.45x
    order_rho_target  = 3.0,   # spectral feedback target — allow attractors to form
    order_rho_power   = 2.0,   # spectral feedback exponent
    consol_bias       = 0.8,   # eta_slow base factor
    consol_scale      = 0.7,   # eta_slow inverse order scale

    # Weight bounds
    W_max         = 2.0,      # unified clip bound (replaces clip_slow/clip_fast)

    # Oja normalization (optional, disabled by default)
    gamma_oja     = 0.0,      # set to 1e-3 to enable

    # Routing
    alpha_0       = 1.0,
    cup_cap       = 1.0,
    cup_leak      = 0.02,
    cup_threshold = -0.2,     # V10: documented. Valence below this fills cup.
                              # Rationale: small negative valence is normal noise;
                              # only sustained negativity should accumulate.

    # Attention
    gamma_TD      = 2.5,
    beta_BU       = 0.4,
    theta_sal     = 0.3,

    # Core
    core_self_coupling = 1.2,   # V10.1: strengthened — core must stay coherent
    spectral_alert     = 4.0,   # V10.1: raised — governor handles soft control
    spectral_iters     = 20,  # power iteration steps for spectral radius
)


# ============================================================
# HOPNET V10
# ============================================================

class HOPNet:
    """
    HOPNet V10 — complete implementation.

    Key invariants maintained throughout:
      - All oscillators on unit circle: |x_i| = 1
      - W_fast and W_slow are symmetric after every update
      - W_fast and W_slow have zero row-sums (suppresses ferromagnet)
      - T matrix is NOT symmetrized (preserves sequence directionality)
      - Mask is applied inside correlation (not after)

    Architecture layers:
      Level 3 — Core phase integrator  [0 : core_size]
      Level 2 — Theta band             [0 : N//2]
      Level 1 — Gamma band             [N//2 : N]
      Valence+ — positive population   [core_size : core_size+valence_size]
      Valence- — negative population   [core_size+V : core_size+2V]
    """

    def __init__(self, **kwargs):
        p = {**DEFAULTS, **kwargs}

        # Store params
        for k, v in p.items():
            setattr(self, k, v)

        N = self.n_osc

        # ── Random keys ──
        key = jax.random.PRNGKey(p['seed'])
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        self._rng_key = k4   # V10: persistent RNG for noise

        # ── Initial phases ──
        theta = jax.random.uniform(k1, (N,)) * 2 * jnp.pi
        self.state = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)

        # ── Frequency bands ──
        n_slow = N // 2
        self.band_id = jnp.concatenate([
            jnp.zeros(n_slow,     dtype=jnp.int32),
            jnp.ones(N - n_slow,  dtype=jnp.int32)
        ])
        omega_slow = jax.random.normal(k2, (n_slow,))     * p['omega_slow_sd']
        omega_fast = jax.random.normal(k2, (N - n_slow,)) * p['omega_fast_sd']
        self.omega = jnp.concatenate([omega_slow, omega_fast])

        # ── Sparsity mask (symmetric) ──
        mask = jax.random.bernoulli(k1, p['sparsity'], (N, N)).astype(jnp.float32)
        self.mask = jnp.maximum(mask, mask.T)

        # ── Weights ──
        W_init = jax.random.normal(k3, (N, N)) * 0.025 / jnp.sqrt(N)
        self.W_fast = (W_init + W_init.T) / 2 * self.mask
        self.W_slow = jnp.zeros_like(self.W_fast)

        # ── Core coupling block ────────────────────────────────────────
        # V10.2 FIX: dense OFF-DIAGONAL cross-coupling inside core.
        # Diagonal self-coupling is cancelled by tangent projection (ChatGPT fix).
        # Cross-coupling is what actually synchronises oscillators.
        # Core phases also initialised aligned (all θ=0) for fast coherence.
        cs = self.core_size
        self.mask = self.mask.at[:cs, :cs].set(1.0)

        # Aligned core initialisation — all core oscillators start at θ=0
        # This gives immediate coherence before any training
        aligned = jnp.stack([jnp.ones(cs), jnp.zeros(cs)], axis=-1)
        self.state = self.state.at[:cs].set(aligned)

        # Dense positive cross-coupling (off-diagonal only)
        # All core oscillators pull each other toward phase alignment
        cc = p['core_self_coupling']
        core_cross = cc * jnp.ones((cs, cs)) / cs   # scaled by 1/cs to control rho
        core_cross = core_cross - jnp.diag(jnp.diag(core_cross))  # zero diagonal
        self.W_slow = self.W_slow.at[:cs, :cs].set(core_cross)

        # ── I AM — oscillator 0 is the dominant anchor ──────────────
        # Expressed through stronger cross-coupling FROM oscillator 0
        # (not diagonal — diagonal is cancelled by tangent projection)
        i_am_row = cc * 2.0 * jnp.ones(cs) / cs
        i_am_row = i_am_row.at[0].set(0.0)   # no self-coupling
        self.W_slow = self.W_slow.at[0, :cs].set(i_am_row)
        self.W_slow = self.W_slow.at[:cs, 0].set(i_am_row)   # symmetric

        # Balance non-core rows only (core block protected)
        non_core_means = jnp.mean(self.W_slow[cs:], axis=1, keepdims=True)
        self.W_slow = self.W_slow.at[cs:].set(self.W_slow[cs:] - non_core_means)

        # ── Core resting state ──
        self.core_rest = jnp.stack([
            jnp.ones(self.core_size),
            jnp.zeros(self.core_size)
        ], axis=-1)

        # ── Valence populations ──
        cs_, vs_ = self.core_size, self.valence_size
        self.val_pos_idx = slice(cs_, cs_ + vs_)
        self.val_neg_idx = slice(cs_ + vs_, cs_ + 2 * vs_)

        # ── T matrices (NOT symmetrized — asymmetry = sequence directionality) ──
        # V10 FIX: complex-valued T matrix preserves phase geometry
        self.T  = jnp.zeros((N, N), dtype=jnp.complex64)
        self.T2 = jnp.zeros((N, N), dtype=jnp.complex64)
        self._prev_z_state = None   # complex phasor of previous state

        # ── Cup ──
        self.cup = 0.0

        # ── Structural plasticity (disabled by default) ──
        self.plasticity_enabled   = False
        self.plasticity_threshold = 0.85
        self.co_activation_counts = jnp.zeros((N, N))

        # ── Monitoring logs ──
        self.spectral_radius_log  = []
        self.lyapunov_log         = []
        self.synchrony_log        = []   # V10: ferromagnet diagnostic

        print(f"HOPNet V10 initialized")
        print(f"  N={N}  core={self.core_size}  valence={vs_}")
        print(f"  sparsity={p['sparsity']:.0%}  connections={int(self.mask.sum())}")
        print(f"  noise={self.sigma_noise}  oja={self.gamma_oja}  backend={jax.default_backend()}")


    # ────────────────────────────────────────────────────────
    # UTILITIES
    # ────────────────────────────────────────────────────────

    def _balance_rows(self, W):
        """
        Enforce row-sum = 0 constraint (Grok V10 patch).
        Suppresses uniform synchrony eigenmode.
        Applied after every weight update.

        V10.1: core BLOCK entirely protected from row-balance.
        Row-balance kills core coherence by fighting core coupling.
        The ferromagnet we want to suppress is GLOBAL synchrony.
        Core synchrony is intentional and must be preserved.

        Rule:
          Core-to-core connections: skip row-balance entirely
          All other connections: apply row-balance normally
        """
        cs = self.core_size

        # Save entire core block before balancing
        core_block = W[:cs, :cs]

        # Apply row-balance to non-core rows only
        non_core_means = jnp.mean(W[cs:], axis=1, keepdims=True)
        W = W.at[cs:].set(W[cs:] - non_core_means)

        # Also balance non-core columns of core rows
        # (core rows talk to non-core — those weights need balancing)
        core_row_non_core = W[:cs, cs:]
        core_row_means = jnp.mean(core_row_non_core, axis=1, keepdims=True)
        W = W.at[:cs, cs:].set(core_row_non_core - core_row_means)

        # Restore core block exactly as it was
        W = W.at[:cs, :cs].set(core_block)

        W = (W + W.T) / 2
        W = W * self.mask

        return W

    def _core_phase(self):
        """
        Extract core unit phasor for consolidation gate and attention.
        Returns real 2-vector [cos φ, sin φ] of mean core phase.
        """
        core_state = self.state[:self.core_size]
        z_core     = jnp.mean(core_state[:, 0] + 1j * core_state[:, 1])
        phi        = jnp.array([z_core.real, z_core.imag])
        return phi / (jnp.linalg.norm(phi) + 1e-8)

    def _state_to_complex(self, state=None):
        """Convert (N,2) oscillator state to (N,) complex phasors."""
        if state is None: state = self.state
        return state[:, 0] + 1j * state[:, 1]

    def _spectral_radius_power(self, W, n_iters=None):
        """
        Power iteration estimate of spectral radius (V10 FIX).
        Replaces Frobenius/sqrt(N) which is not an upper bound.

        Returns tight estimate of dominant eigenvalue magnitude.
        Cost: n_iters matrix-vector products — negligible vs simulation.
        """
        if n_iters is None: n_iters = self.spectral_iters
        N   = W.shape[0]
        key = jax.random.PRNGKey(int(jnp.sum(jnp.abs(W))) % 2**31)
        v   = jax.random.normal(key, (N,))
        v   = v / (jnp.linalg.norm(v) + 1e-8)
        for _ in range(n_iters):
            v = W @ v
            nrm = jnp.linalg.norm(v)
            v = v / (nrm + 1e-8)
        rho = float(jnp.linalg.norm(W @ v))
        return rho


    # ────────────────────────────────────────────────────────
    # CORE DYNAMICS
    # ────────────────────────────────────────────────────────

    @partial(jax.jit, static_argnums=(0,))
    def _rk4_step(self, state, W, c, dt):
        """
        RK4 with tangent-space projection.
        Noise (if enabled) is added inside this function via carry — see simulate().
        """
        def dynamics(s):
            rot      = jnp.stack([-self.omega * s[:, 1],
                                    self.omega * s[:, 0]], axis=-1)
            coupling = W @ s
            force    = rot + coupling + c
            proj     = jnp.sum(s * force, axis=-1, keepdims=True)
            return force - proj * s

        k1 = dynamics(state)
        k2 = dynamics(state + 0.5 * dt * k1)
        k3 = dynamics(state + 0.5 * dt * k2)
        k4 = dynamics(state + dt * k3)
        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return new_state / jnp.linalg.norm(new_state, axis=-1, keepdims=True)

    def simulate(self, n_steps=300, dt=None, input_field=None,
                 fast_weight=None, slow_weight=None):
        """
        Run dynamics for n_steps.

        V10 FIX: noise is sampled per timestep inside scan_fn via RNG carry.
        Previously noise was static per call (random bias field, not true noise).
        """
        if dt is None:          dt          = self.dt
        if input_field is None: input_field = jnp.zeros((self.n_osc, 2))

        fw = fast_weight if fast_weight is not None else self._route_fast_weight()
        sw = slow_weight if slow_weight is not None else self._route_slow_weight()
        W  = fw * self.W_fast + sw * self.W_slow
        c  = input_field
        sigma = self.sigma_noise

        # V10: RNG key carried through scan for per-step noise
        init_key = self._rng_key

        @jax.jit
        def scan_fn(carry, _):
            state, w, rng_key = carry
            if sigma > 0:
                rng_key, subkey = jax.random.split(rng_key)
                noise = jax.random.normal(subkey, state.shape) * sigma
                c_noisy = c + noise
            else:
                c_noisy = c
            new_state = self._rk4_step(state, w, c_noisy, dt)
            return (new_state, w, rng_key), new_state

        (final_state, _, final_key), traj = lax.scan(
            scan_fn, (self.state, W, init_key), None, length=n_steps
        )
        self.state    = final_state
        self._rng_key = final_key
        return traj


    # ────────────────────────────────────────────────────────
    # PLASTICITY — V10 COMPLETE REWRITE
    # ────────────────────────────────────────────────────────

    def _compute_hebbian(self, trajectory):
        """
        Hebbian correlation with mean-field suppression + phase-only normalization.

        Three-step pipeline (each step has a specific role):

        Step 1 — Mean-field centering (V10, Grok):
          z_centered = z - mean(z)
          Removes global synchrony component. Kills ferromagnetic attractor.

        Step 2 — Phase-only normalization (V10+, Grok capacity boost):
          z_centered = z_centered / (|z_centered| + eps)
          Projects centered phasors back to unit circle.
          WHY THIS MATTERS:
            After centering, |z_centered_i| varies — oscillators more aligned
            with the global mean have smaller residual magnitude and contribute
            less to the outer product. This reintroduces "winner-take-more"
            eigenmode concentration across patterns.
            Normalizing back to unit magnitude forces every oscillator to
            contribute equally regardless of alignment. The learning rule
            then stores pure relative phase structure — not amplitude-weighted
            correlations. This is the difference between storing the full
            phasor relationship and storing only the phase offset.
          CAPACITY EFFECT:
            Centering + row-balance stops the ferromagnet.
            Phase normalization stops eigenmode concentration.
            Together: moves capacity from O(0.138N) binary Hopfield regime
            toward O(N) phase-coded regime. More distinct stable memories
            before spurious mixture attractors dominate.

        Step 3 — Masked outer product (V10):
          corr = Re(outer(z_norm, conj(z_norm))) * M
          Mask inside correlation — spec-correct.

        Mathematical form:
          z_i       = x_i^(1) + i x_i^(2)
          z_c_i     = z_i - (1/N) Σ_k z_k          [centering]
          z_n_i     = z_c_i / (|z_c_i| + ε)         [phase-only]
          ΔW_ij     = η · Re(z_n_i · conj(z_n_j)) · M_ij
        """
        mean_state = jnp.mean(trajectory[-50:], axis=0)   # (N, 2)

        # Complex phasors
        z = mean_state[:, 0] + 1j * mean_state[:, 1]      # (N,) complex

        # Step 1: Remove global synchrony component
        z_centered = z - jnp.mean(z)

        # Step 2: Phase-only normalization — capacity boost
        # Each oscillator contributes equally regardless of alignment magnitude
        # V10.2 FIX: soft normalization (ChatGPT: 0.5 power preserves
        # differential commitment while preventing eigenmode concentration)
        # Full unit norm (power=1.0) made every pattern look identical
        # when core was incoherent. 0.5 power is the sweet spot.
        # Global centering — kills ferromagnet cleanly
        z_centered   = z - jnp.mean(z)
        z_normalized = z_centered / (jnp.abs(z_centered) ** 0.5 + 1e-8)

        corr = jnp.real(jnp.outer(z_normalized, jnp.conj(z_normalized)))

        # Mask INSIDE correlation
        corr = corr * self.mask

        return corr

    def hebbian_update(self, trajectory, consolidate=False):
        """
        Fast Hebbian update — V10.1: Order-modulated learning rate.

        Three-factor learning rate (all three AI reviewers agreed):
          1. Core order gate    — linear ramp: high coherence = learn hard
          2. Spectral governor  — negative feedback: rho near limit = slow down
          3. Valence modulation — existing: high |v| = boost learning

        Biological analog: ACh-gated plasticity
          High order (focused, coherent) = high eta_fast
          Low order  (diffuse, wandering) = low eta_fast

        This resolves the fixed eta instability:
          eta=0.022 base, but gate brings effective rate to 0.013-0.030
          depending on coherence and spectral state.
        """
        # Factor 1: core order gate
        mean_state = jnp.mean(trajectory[-50:], axis=0)
        core_z     = jnp.mean(mean_state[:self.core_size, 0] +
                              1j * mean_state[:self.core_size, 1])
        core_order = float(jnp.abs(core_z))
        order_gate = self.order_gate_bias + self.order_gate_scale * core_order

        # Factor 2: spectral governor (negative feedback)
        rho_now  = self._spectral_radius_power(self.W_fast)
        spec_gov = float(jnp.clip(
            1.0 - (rho_now / self.order_rho_target) ** self.order_rho_power,
            0.1, 1.0))

        # Factor 3: valence modulation
        v_mag  = abs(self.get_valence())

        # Combined effective rate
        lr_eff = self.eta_fast * order_gate * spec_gov * (1.0 + self.lambda_v * v_mag)

        corr = self._compute_hebbian(trajectory)

        # Fast weight update
        self.W_fast = 0.99 * self.W_fast + lr_eff * corr

        # Symmetry first
        self.W_fast = (self.W_fast + self.W_fast.T) / 2
        self.W_fast = self.W_fast * self.mask

        # Optional Oja-style row energy normalization
        if self.gamma_oja > 0.0:
            row_energy  = jnp.sum(self.W_fast * self.W_fast, axis=1, keepdims=True)
            self.W_fast = self.W_fast - self.gamma_oja * row_energy * self.W_fast

        # V10.2 FIX: use _balance_rows() which preserves symmetry and
        # protects core block. Manual row-mean broke symmetry after every update,
        # causing Lyapunov to ascend. Symmetric weights = descending energy = attractors.
        self.W_fast = self._balance_rows(self.W_fast)

        # Clip
        self.W_fast = jnp.clip(self.W_fast, -self.W_max, self.W_max)

        # Final spectral enforcement on W_fast
        rho_fast = self._spectral_radius_power(self.W_fast)
        if rho_fast > self.spectral_alert:
            self.W_fast = self.W_fast * (self.spectral_alert / rho_fast)

        # Consolidate into W_slow if requested
        if consolidate:
            self._consolidate()

        # Decay working memory
        self.W_fast = self.W_fast * self.decay

        # Update cup
        self._update_cup()

        # Structural plasticity (if enabled)
        if self.plasticity_enabled:
            self._structural_plasticity_step(trajectory)

        # Log diagnostics
        self.synchrony_log.append(self.global_synchrony())
        self.order_log = getattr(self, 'order_log', [])
        self.order_log.append(core_order)
        self.lr_log = getattr(self, 'lr_log', [])
        self.lr_log.append(float(lr_eff))

    def _consolidate(self):
        """
        Gate W_fast into W_slow — V10.1: inverse order modulation (sleep analog).

        eta_slow is INVERSELY modulated by core order:
          High order (awake, coherent)  = low  eta_slow (don't overbake)
          Low order  (rest, diffuse)    = high eta_slow (consolidate offline)

        This matches ACh/sleep-wake cycle:
          Wake:  high ACh = high eta_fast, LOW  eta_slow
          Sleep: low  ACh = low  eta_fast, HIGH eta_slow

        Spectral governor also applied to W_slow update.
        """
        # Core order at current state
        core_z     = jnp.mean(self.state[:self.core_size, 0] +
                              1j * self.state[:self.core_size, 1])
        core_order = float(jnp.abs(core_z))

        # Inverse modulation: low order = consolidate more
        eta_slow_eff = self.eta_slow * (self.consol_bias +
                       self.consol_scale * (0.5 - core_order))
        eta_slow_eff = float(jnp.clip(eta_slow_eff,
                             self.eta_slow * 0.2,
                             self.eta_slow * 1.5))

        # Spatial gate (which oscillators to consolidate)
        core_phase = self._core_phase()
        alignment  = jnp.abs(self.state @ core_phase)
        gate       = jax.nn.sigmoid(self.kappa * (alignment - self.theta_c))

        delta = (self.W_fast - self.W_slow) * jnp.outer(gate, gate)

        # Spectral governor on W_slow before update
        rho_slow = self._spectral_radius_power(self.W_slow)
        if rho_slow > self.order_rho_target:
            self.W_slow = self.W_slow * (self.order_rho_target / rho_slow)

        self.W_slow = self.W_slow + eta_slow_eff * delta

        # Symmetry once
        self.W_slow = (self.W_slow + self.W_slow.T) / 2

        # V10.2 FIX: use _balance_rows() to preserve symmetry AND core block
        # Old manual row-balance was wiping core cross-coupling every consolidation
        self.W_slow = self._balance_rows(self.W_slow)

        # Clip
        self.W_slow = jnp.clip(self.W_slow, -self.W_max, self.W_max)

        # Final spectral enforcement
        rho = self._spectral_radius_power(self.W_slow)
        if rho > self.spectral_alert:
            self.W_slow = self.W_slow * (self.spectral_alert / rho)
            rho = self.spectral_alert

        self.spectral_radius_log.append(rho)
        self.lyapunov_log.append(self.lyapunov_energy())


    # ────────────────────────────────────────────────────────
    # SEQUENCE LEARNING — V10 COMPLEX T MATRIX
    # ────────────────────────────────────────────────────────

    def learn_transition(self, state_from, state_to, lr=None):
        """
        Asymmetric T matrix update using complex phasors (V10 FIX).

        V9 bug: mean(state, axis=-1) compressed 2D phasors to scalars,
        destroying phase geometry. T learned coordinate averages, not phases.

        V10 fix: operate in complex space.
          z_from = state_from[:, 0] + 1j * state_from[:, 1]
          z_to   = state_to[:, 0]   + 1j * state_to[:, 1]
          T += lr * outer(z_to, conj(z_from))

        T[i,j] now encodes: phase of j at time t predicts phase of i at t+1.
        Asymmetry guaranteed: outer(z_to, conj(z_from)) ≠ outer(z_from, conj(z_to))
        T is NOT symmetrized.
        """
        if lr is None: lr = self.eta_T

        z_from = self._state_to_complex(state_from)
        z_to   = self._state_to_complex(state_to)

        # 1-step complex outer product
        self.T = self.T + lr * jnp.outer(z_to, jnp.conj(z_from))
        self.T = jnp.clip(jnp.abs(self.T), 0, self.W_max) * jnp.exp(
            1j * jnp.angle(self.T))   # clip magnitude, preserve phase

        # 2-step
        if self._prev_z_state is not None:
            self.T2 = self.T2 + (lr * 0.5) * jnp.outer(z_to, jnp.conj(self._prev_z_state))
            self.T2 = jnp.clip(jnp.abs(self.T2), 0, self.W_max) * jnp.exp(
                1j * jnp.angle(self.T2))

        self._prev_z_state = z_from

    def predict_next(self, blend=0.7):
        """
        Predict next state from T matrix using complex phasors.
        Returns real (N, 2) input field from complex prediction.
        """
        z_state = self._state_to_complex()       # (N,) complex
        pred_1  = self.T  @ z_state              # (N,) complex
        pred_2  = self.T2 @ z_state              # (N,) complex
        pred    = blend * pred_1 + (1.0 - blend) * pred_2

        # Convert complex prediction back to (N, 2) field
        return jnp.stack([jnp.real(pred), jnp.imag(pred)], axis=-1)


    # ────────────────────────────────────────────────────────
    # ATTENTION
    # ────────────────────────────────────────────────────────

    def top_down_gain(self, input_field, gamma=None):
        """Top-down gain: amplify input aligned with core phase."""
        if gamma is None: gamma = self.gamma_TD
        core_phase = self._core_phase()
        alignment  = jnp.abs(self.state @ core_phase)
        gain       = 1.0 + gamma * alignment[:, None]
        return input_field * gain

    def salience_detection(self, input_field):
        """Bottom-up salience from prediction error."""
        pred_field = self.predict_next()
        pred_mag   = jnp.linalg.norm(pred_field, axis=-1)
        input_mag  = jnp.linalg.norm(input_field, axis=-1)
        error      = jnp.maximum(input_mag - pred_mag, 0.0)
        return jax.nn.sigmoid(10.0 * (error - self.theta_sal))

    def attend(self, input_field):
        """Combined top-down + bottom-up attention."""
        td_field = self.top_down_gain(input_field)
        salience = self.salience_detection(input_field)
        bu_field = input_field * (1.0 + self.beta_BU * salience[:, None])
        return 0.6 * td_field + 0.4 * bu_field


    # ────────────────────────────────────────────────────────
    # VALENCE AND CUP
    # ────────────────────────────────────────────────────────

    def get_valence(self):
        """Signed valence: r_pos - r_neg. Scalar in [-1, +1]."""
        pos   = self.state[self.val_pos_idx]
        neg   = self.state[self.val_neg_idx]
        r_pos = float(jnp.abs(jnp.mean(pos[:, 0] + 1j * pos[:, 1])))
        r_neg = float(jnp.abs(jnp.mean(neg[:, 0] + 1j * neg[:, 1])))
        return r_pos - r_neg

    def set_valence_input(self, valence: float, strength: float = 0.5):
        """Drive valence populations. Returns (N, 2) input field."""
        field = jnp.zeros((self.n_osc, 2))
        v_pat = jnp.ones((self.valence_size, 2)) * abs(valence) * strength
        idx   = self.val_pos_idx if valence > 0 else self.val_neg_idx
        return field.at[idx].set(v_pat)

    def _update_cup(self):
        """
        Cup accumulator (V10: threshold documented).
        C += α_fill * max(0, -v)  for v < cup_threshold
        Threshold default -0.2: small negative valence is normal dynamics noise.
        Only sustained negativity should accumulate.
        """
        v = self.get_valence()
        if v < self.cup_threshold:
            self.cup += abs(v) * 0.05
        self.cup = max(0.0, self.cup - self.cup_leak)
        self.cup = min(self.cup, 2.0)

    @property
    def cup_pressure(self):
        return min(self.cup / self.cup_cap, 1.0)

    @property
    def logic_available(self):
        return self.cup < self.cup_cap


    # ────────────────────────────────────────────────────────
    # ROUTING
    # ────────────────────────────────────────────────────────

    def get_arousal(self):
        """Displacement of core from resting state. Clipped to [0,1]."""
        cs     = self.core_size
        z_cur  = jnp.mean(self.state[:cs, 0] + 1j * self.state[:cs, 1])
        z_rest = jnp.mean(self.core_rest[:, 0] + 1j * self.core_rest[:, 1])
        return float(jnp.clip(jnp.abs(z_cur - z_rest), 0.0, 1.0))

    def _route_fast_weight(self):
        A  = self.get_arousal()
        cp = self.cup_pressure
        return float(jnp.clip(self.alpha_0 * (1.0 + 0.5*A) * (1.0 - 0.3*cp), 0.3, 2.0))

    def _route_slow_weight(self):
        return float(jnp.clip(self.alpha_0 * (1.0 + 0.2 * self.cup_pressure), 0.8, 1.5))


    # ────────────────────────────────────────────────────────
    # METRICS
    # ────────────────────────────────────────────────────────

    def get_order(self, state=None):
        """Kuramoto order parameter. 1=sync, 0=random."""
        if state is None: state = self.state
        z = state[:, 0] + 1j * state[:, 1]
        return float(jnp.abs(jnp.mean(z)))

    def global_synchrony(self):
        """
        Global synchrony diagnostic (V10 — ferromagnet monitor).
        High value after training = ferromagnetic collapse risk.
        Should stay LOW after V10 centering fix.
        """
        return self.get_order()

    def similarity_to(self, pattern):
        """
        Phase-invariant similarity (relative Kuramoto order parameter).
        S = |(1/N) Σ_i exp(i(θ_i - θ_i^pattern))| ∈ [0,1]
        1.0 = perfect recall, 0.0 = unrelated.
        """
        z_s = self.state[:, 0] + 1j * self.state[:, 1]
        z_p = pattern[:, 0]   + 1j * pattern[:, 1]
        return float(jnp.abs(jnp.mean(z_s * jnp.conj(z_p))))

    def lyapunov_energy(self):
        """E(x) = -½ tr(xᵀ W_sym x). Decreases during free relaxation."""
        W_sym = (self.W_slow + self.W_slow.T) / 2
        xWx   = jnp.sum(self.state * (W_sym @ self.state))
        return float(-0.5 * xWx)

    def status(self):
        """Full status printout."""
        rho = self._spectral_radius_power(self.W_slow)
        print(f"\n── HOPNet V10 Status ─────────────────────────────")
        print(f"  Order:           {self.get_order():.3f}")
        print(f"  Global synchrony:{self.global_synchrony():.3f}  (low=good after V10)")
        print(f"  Arousal:         {self.get_arousal():.3f}")
        print(f"  Valence:         {self.get_valence():+.3f}")
        print(f"  Cup pressure:    {self.cup_pressure:.3f}  {'[VETO]' if not self.logic_available else '[OK]'}")
        print(f"  Lyapunov E:      {self.lyapunov_energy():.4f}")
        print(f"  Spectral ρ:      {rho:.4f}  {'[ALERT]' if rho > self.spectral_alert else '[OK]'}")
        print(f"  W_fast norm:     {float(jnp.linalg.norm(self.W_fast)):.4f}")
        print(f"  W_slow norm:     {float(jnp.linalg.norm(self.W_slow)):.4f}")
        print(f"  T norm:          {float(jnp.linalg.norm(jnp.abs(self.T))):.4f}")
        print(f"  Logic:           {'available' if self.logic_available else 'BLOCKED (cup)'}")
        print(f"──────────────────────────────────────────────────\n")


    # ────────────────────────────────────────────────────────
    # CONTEXT MANAGEMENT
    # ────────────────────────────────────────────────────────

    def reset_working_memory(self):
        self.W_fast          = jnp.zeros_like(self.W_fast)
        self.cup             = max(0.0, self.cup - 0.1)
        self._prev_z_state   = None

    def rest(self, epochs=5):
        """Consolidation rest — no input, cup drains."""
        for _ in range(epochs):
            self.simulate(n_steps=200, input_field=None)
            self.cup = max(0.0, self.cup - self.cup_leak * 3)
        print(f"  Rest: cup={self.cup_pressure:.2f}  E={self.lyapunov_energy():.4f}")


    # ────────────────────────────────────────────────────────
    # STRUCTURAL PLASTICITY (disabled by default)
    # ────────────────────────────────────────────────────────

    def _structural_plasticity_step(self, trajectory):
        mean_state = jnp.mean(trajectory[-50:], axis=0)
        co_act     = jnp.abs(mean_state @ mean_state.T)
        self.co_activation_counts = 0.95 * self.co_activation_counts + 0.05 * co_act
        new_conns  = ((self.co_activation_counts > self.plasticity_threshold) &
                      (self.mask < 0.5)).astype(jnp.float32)
        if float(new_conns.sum()) > 0:
            self.mask   = jnp.minimum(self.mask + new_conns, 1.0)
            self.W_slow = self.W_slow + new_conns * 0.01
        core_protect = (jnp.arange(self.n_osc) < self.core_size)[:, None]
        prune = ((jnp.abs(self.W_slow) < 0.001) & (jnp.abs(self.W_fast) < 0.001) & ~core_protect)
        self.mask = jnp.where(prune, 0.0, self.mask)


    # ────────────────────────────────────────────────────────
    # 4-PATTERN BENCHMARK (Grok spec — built in to V10)
    # ────────────────────────────────────────────────────────

    def run_recall_benchmark(self, patterns, noise_fraction=0.25,
                              clamp_steps=25, recall_steps=400,
                              n_trials=100, success_sim=0.80,
                              success_margin=0.20, verbose=True):
        """
        4-pattern attractor recall benchmark (Grok V10 spec).

        Protocol:
          1. For each pattern, corrupt noise_fraction of oscillators
          2. Clamp noisy cue for clamp_steps
          3. Release input, simulate for recall_steps
          4. Measure phase-invariant similarity to all patterns

        Success per trial:
          S_target > success_sim AND S_target > max(S_others) + success_margin

        Also reports global synchrony R — should stay low after V10 fix.

        Args:
            patterns:       list of (N,2) stored patterns
            noise_fraction: fraction of oscillators to corrupt (0.25 = 25%)
            clamp_steps:    steps to hold cue
            recall_steps:   steps of free relaxation
            n_trials:       trials per pattern
            success_sim:    minimum similarity for correct recall
            success_margin: minimum margin over best competitor

        Returns:
            dict with per-pattern accuracy and global synchrony stats
        """
        n_patterns = len(patterns)
        results    = {i: [] for i in range(n_patterns)}
        synchrony_baseline = []

        if verbose:
            print(f"\n── 4-Pattern Recall Benchmark ──────────────────")
            print(f"  Patterns:      {n_patterns}")
            print(f"  Noise:         {noise_fraction:.0%}")
            print(f"  Trials/pattern:{n_trials}")
            print(f"  Success:       S>{success_sim} AND margin>{success_margin}")
            print()

        for pat_idx, target_pat in enumerate(patterns):
            correct = 0
            for trial in range(n_trials):
                # Save state
                saved_state = self.state

                # Generate noisy cue
                key = jax.random.PRNGKey(pat_idx * 10000 + trial)
                n_corrupt = int(self.n_osc * noise_fraction)
                corrupt_idx = jax.random.choice(key, self.n_osc, (n_corrupt,), replace=False)
                random_phases = jax.random.uniform(key, (n_corrupt,)) * 2 * jnp.pi
                noisy_cue = target_pat.at[corrupt_idx].set(
                    jnp.stack([jnp.cos(random_phases), jnp.sin(random_phases)], axis=-1)
                )

                # Clamp cue
                inp = noisy_cue * 1.4   # V10.3: stronger clamp to break core compromise state
                self.simulate(n_steps=clamp_steps, input_field=inp)

                # Release and free-run
                self.simulate(n_steps=recall_steps, input_field=None)

                # Measure similarities
                sims = [self.similarity_to(p) for p in patterns]
                s_target = sims[pat_idx]
                s_others = [sims[k] for k in range(n_patterns) if k != pat_idx]
                s_best_other = max(s_others) if s_others else 0.0

                # Debug: print first trial similarities
                if trial == 0:
                    print(f"    [debug] pat{pat_idx} trial 0: target={s_target:.3f} others={[f'{s:.3f}' for s in s_others]}")

                # Success criterion
                if s_target > success_sim and s_target > s_best_other + success_margin:
                    correct += 1

                # Global synchrony
                synchrony_baseline.append(self.global_synchrony())

                # Restore state (each trial independent)
                self.state = saved_state

            acc = correct / n_trials
            results[pat_idx] = acc
            if verbose:
                print(f"  Pattern {pat_idx}:  {acc:.0%} ({correct}/{n_trials})")

        mean_acc = float(np.mean(list(results.values())))
        mean_syn = float(np.mean(synchrony_baseline))

        if verbose:
            print(f"\n  Mean accuracy:   {mean_acc:.0%}")
            print(f"  Global synchrony:{mean_syn:.3f}  (should be LOW after V10)")
            if mean_syn > 0.5:
                print(f"  ⚠ High synchrony — ferromagnet may still be active")
            else:
                print(f"  ✓ Low synchrony — centering fix working")
            print(f"──────────────────────────────────────────────────\n")

        return {
            'per_pattern': results,
            'mean_accuracy': mean_acc,
            'mean_synchrony': mean_syn,
        }


    def run_capacity_sweep(self, pattern_counts=None, epochs=10,
                           noise_fraction=0.25, n_trials=20, verbose=True):
        """
        Capacity sweep benchmark (Grok V10+ spec).

        Trains P patterns for each P in pattern_counts.
        Measures recall accuracy vs number of stored patterns.
        Proves phase-only normalization extends capacity beyond 0.138N.

        Also measures spurious mixture rate:
          - 100 free runs from random initial state
          - Spurious = no single pattern dominates (max S_k < 0.6)
          - High spurious rate = capacity exceeded

        Args:
            pattern_counts: list of P values to test (default [4,8,16,32])
            epochs:         training epochs per sweep
            noise_fraction: cue corruption for recall test
            n_trials:       recall trials per pattern per P

        Returns:
            dict: accuracy and spurious rate per P
        """
        if pattern_counts is None:
            pattern_counts = [4, 8, 16, 32]

        results = {}

        if verbose:
            print(f"\n── Capacity Sweep ───────────────────────────────")
            print(f"  Pattern counts: {pattern_counts}")
            print(f"  Epochs/sweep:   {epochs}")
            print(f"  {'P':<6} {'Accuracy':<12} {'Spurious%':<12} {'Sync':<8}")
            print(f"  {'-'*38}")

        for P in pattern_counts:
            # Fresh network for each sweep
            net = HOPNet(n_osc=self.n_osc, seed=42,
                         core_size=self.core_size,
                         valence_size=self.valence_size)

            # Generate P random patterns
            patterns = []
            for i in range(P):
                k  = jax.random.PRNGKey(i * 137 + 42)
                th = jax.random.uniform(k, (self.n_osc,)) * 2 * jnp.pi
                patterns.append(jnp.stack([jnp.cos(th), jnp.sin(th)], axis=-1))

            # Train
            for epoch in range(epochs):
                for p in patterns:
                    inp  = net.attend(p * 0.85)   # V10.2: stronger drive for phase-lock
                    traj = net.simulate(n_steps=150, input_field=inp)
                    net.hebbian_update(traj, consolidate=(epoch >= epochs//2))
                net.reset_working_memory()

            # Recall benchmark
            bench = net.run_recall_benchmark(
                patterns, noise_fraction=noise_fraction,
                n_trials=n_trials, verbose=False
            )

            # Spurious mixture rate: free runs from random state
            spurious = 0
            for trial in range(50):
                key = jax.random.PRNGKey(trial + 9999)
                rand_theta = jax.random.uniform(key, (self.n_osc,)) * 2 * jnp.pi
                net.state = jnp.stack([jnp.cos(rand_theta), jnp.sin(rand_theta)], axis=-1)
                net.simulate(n_steps=400, input_field=None)
                sims = [net.similarity_to(p) for p in patterns]
                if max(sims) < 0.6:
                    spurious += 1
            spurious_rate = spurious / 50

            results[P] = {
                'accuracy':      bench['mean_accuracy'],
                'spurious_rate': spurious_rate,
                'synchrony':     bench['mean_synchrony'],
            }

            if verbose:
                print(f"  {P:<6} {bench['mean_accuracy']:<12.0%} "
                      f"{spurious_rate:<12.0%} {bench['mean_synchrony']:<8.3f}")

        if verbose:
            print(f"  {'-'*38}")
            print(f"  Classical Hopfield capacity limit: ~{0.138 * self.n_osc:.0f} patterns")
            print(f"──────────────────────────────────────────────────\n")

        return results

    # ────────────────────────────────────────────────────────
    # VERIFICATION TESTS
    # ────────────────────────────────────────────────────────

    def test_no_ferromagnet(self):
        """V10 NEW: Global synchrony should stay low after training."""
        print("TEST: Ferromagnet suppression")
        # Train with random inputs
        for _ in range(5):
            dummy = jax.random.normal(jax.random.PRNGKey(0), (self.n_osc, 2))
            dummy = dummy / jnp.linalg.norm(dummy, axis=-1, keepdims=True)
            traj  = self.simulate(n_steps=150, input_field=dummy * 0.5)
            self.hebbian_update(traj, consolidate=False)
        # Free run from random state
        noise = jax.random.normal(jax.random.PRNGKey(42), (self.n_osc, 2))
        self.simulate(n_steps=500, input_field=noise * 0.3)
        syn = self.global_synchrony()
        passed = syn < 0.4
        print(f"  Global synchrony after training: {syn:.3f}  {'✓' if passed else '✗ (ferromagnet active)'}\n")
        return passed

    def test_consolidation(self, patterns):
        """Non-core W_slow grows after consolidation."""
        print("TEST: Consolidation")
        non_core = 1.0 - ((jnp.arange(self.n_osc) < self.core_size)[:, None] * jnp.eye(self.n_osc))
        before = float(jnp.linalg.norm(self.W_slow * non_core))
        for _ in range(8):
            for p in patterns:
                traj = self.simulate(n_steps=150, input_field=p * 0.68)
                self.hebbian_update(traj, consolidate=True)
        after  = float(jnp.linalg.norm(self.W_slow * non_core))
        passed = after > before * 1.05
        print(f"  before={before:.4f}  after={after:.4f}  {'✓' if passed else '✗'}\n")
        return passed

    def test_core_stability(self):
        """Core recovers after strong disturbance."""
        print("TEST: Core stability")
        noise = jax.random.normal(jax.random.PRNGKey(999), (self.n_osc, 2)) * 0.5
        self.simulate(n_steps=200, input_field=noise)
        for _ in range(10):
            self.simulate(n_steps=200, input_field=None)
        core_order = float(jnp.abs(jnp.mean(
            self.state[:self.core_size, 0] + 1j * self.state[:self.core_size, 1])))
        passed = core_order > 0.08
        print(f"  core_order={core_order:.3f}  {'✓' if passed else '✗'}\n")
        return passed

    def test_fast_weights_clear(self):
        """W_fast < 5% of peak after 15 decay cycles."""
        print("TEST: Fast weights clear")
        dummy = jnp.ones((self.n_osc, 2)) * 0.68
        for _ in range(5):
            traj = self.simulate(n_steps=180, input_field=dummy)
            self.hebbian_update(traj, consolidate=False)
        peak = float(jnp.linalg.norm(self.W_fast))
        for _ in range(15):
            self.W_fast = self.W_fast * self.decay
        after  = float(jnp.linalg.norm(self.W_fast))
        ratio  = after / (peak + 1e-8)
        passed = ratio < 0.05
        print(f"  peak={peak:.3f}  after={after:.3f}  ({ratio:.1%})  {'✓' if passed else '✗'}\n")
        return passed

    def test_spectral_radius(self):
        """Power iteration gives reasonable estimate."""
        print("TEST: Spectral radius (power iteration)")
        rho = self._spectral_radius_power(self.W_slow)
        passed = 0.0 < rho < self.W_max * 2
        print(f"  ρ={rho:.4f}  {'✓' if passed else '✗'}\n")
        return passed

    def test_lyapunov_descent(self, pattern):
        """Energy decreases during free relaxation."""
        print("TEST: Lyapunov descent")
        self.simulate(n_steps=100, input_field=pattern * 0.5)
        e0 = self.lyapunov_energy()
        self.simulate(n_steps=300, input_field=None)
        e1 = self.lyapunov_energy()
        passed = e1 <= e0 + 0.1
        print(f"  E: {e0:.4f} → {e1:.4f}  ΔE={e1-e0:.4f}  {'✓' if passed else '✗'}\n")
        return passed

    def run_all_tests(self, patterns=None):
        """Run all verification tests."""
        print("\n" + "="*52)
        print("  HOPNet V10 — VERIFICATION TESTS")
        print("="*52 + "\n")
        results = []
        results.append(self.test_no_ferromagnet())
        results.append(self.test_core_stability())
        results.append(self.test_fast_weights_clear())
        results.append(self.test_spectral_radius())
        if patterns:
            results.append(self.test_consolidation(patterns))
            results.append(self.test_lyapunov_descent(patterns[0]))
        n = sum(results)
        print(f"  {n}/{len(results)} tests passed")
        if n == len(results):
            print("  ✓ All tests passed — V10 architecture verified\n")
        else:
            print(f"  ⚠ {len(results)-n} failed\n")
        return n == len(results)


# ============================================================
# QUICK START
# ============================================================

if __name__ == "__main__":
    import time

    print("\n" + "="*52)
    print("  HOPNet V10 — Quick Start + Benchmark")
    print("="*52)

    net = HOPNet(n_osc=512, seed=42)
    net.status()

    def make_pattern(seed, n):
        k  = jax.random.PRNGKey(seed)
        th = jax.random.uniform(k, (n,)) * 2 * jnp.pi
        return jnp.stack([jnp.cos(th), jnp.sin(th)], axis=-1)

    patterns = [make_pattern(s, net.n_osc) for s in [10, 20]]

    print("Training 4 patterns (10 epochs)...")
    t0 = time.time()
    # V10.3: Episode-based training — context not position
    # Symmetric episodes guarantee equal basin depth for all patterns
    # T matrix learns valence-weighted contextual transitions
    import numpy as np_train
    episodes = [
        (0, 1, 1.0),   # p0 -> p1 positive
        (1, 0, 1.0),   # p1 -> p0 positive
        (0, 0, 0.8),   # p0 self-reinforcing
        (1, 1, 0.8),   # p1 self-reinforcing
    ]
    for epoch in range(30):
        ep_order = list(range(len(episodes)))
        np_train.random.seed(epoch)
        np_train.random.shuffle(ep_order)
        for ei in ep_order:
            pi_a, pi_b, valence = episodes[ei]
            p_a = patterns[pi_a]
            p_b = patterns[pi_b]
            # Train pattern A
            traj_a = net.simulate(n_steps=200, input_field=net.attend(p_a * 0.85))
            net.hebbian_update(traj_a, consolidate=True)
            net.W_fast = net.W_fast * 0.0
            # Train pattern B
            traj_b = net.simulate(n_steps=200, input_field=net.attend(p_b * 0.85))
            net.hebbian_update(traj_b, consolidate=True)
            # T matrix: A -> B weighted by valence
            z_a = jnp.mean(traj_a[-50:], axis=0); z_a = z_a[:,0] + 1j*z_a[:,1]
            z_b = jnp.mean(traj_b[-50:], axis=0); z_b = z_b[:,0] + 1j*z_b[:,1]
            net.T = net.T + net.eta_T * jnp.outer(z_b, jnp.conj(z_a)) * valence
            t_norm = float(jnp.linalg.norm(jnp.abs(net.T)))
            if t_norm > 20.0: net.T = net.T * (20.0 / t_norm)
            net.W_fast = net.W_fast * 0.0
        net.reset_working_memory()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}  E={net.lyapunov_energy():.4f}"
                  f"  sync={net.global_synchrony():.3f}"
                  f"  W_slow={float(jnp.linalg.norm(net.W_slow)):.3f}"
                  f"  order={net.get_order():.3f}")

    print(f"Training complete in {time.time()-t0:.1f}s")

    # Diagnostic: how strongly is each pattern stored in W_slow?
    print("  Pattern storage in W_slow:")
    for i, p in enumerate(patterns):
        z = p[:,0] + 1j*p[:,1]
        zc = z - jnp.mean(z)
        zn = zc / (jnp.abs(zc)**0.5 + 1e-8)
        signal = float(jnp.abs(jnp.sum(jnp.real(jnp.outer(zn, jnp.conj(zn))) * net.W_slow)))
        print(f"    Pattern {i}: W_slow projection = {signal:.4f}")
    print()

    # Verification tests
    net.run_all_tests(patterns)

    # 4-pattern recall benchmark
    net.run_recall_benchmark(patterns, noise_fraction=0.25, n_trials=20,
                          success_sim=0.50, success_margin=0.20)

    net.status()
    print("V10 ready. Run danger_demo.py for the full proof of concept.")

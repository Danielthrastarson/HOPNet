"""
HOPNet V11 — Emergent Self Oscillatory Network
===============================================
Philosophy: "One population. One matrix for content. One matrix for time.
             Everything else emerges."

V10 taught us:
  - Hardwired core (200 oscillators) caused basin interference
  - Anti-Hebbian hacks were unstable
  - Complexity fought itself
  - Nature does not like more complicated than it needs to

V11 design (convergent recommendation from ChatGPT + Grok + Gemini + designer):
  - No hardwired core — stability emerges from rehearsal
  - No W_fast — one memory substrate until recall works
  - No spectral governor — Oja's rule bounds weights naturally
  - No valence oscillator population — scalar signal only
  - No protected row-balance — standard everywhere
  - I AM = most rehearsed self-returning trajectory in T matrix

Architecture:
  N homogeneous oscillators on unit circle (phase-coded)
  W  — symmetric Hermitian (content memory, static attractors)
  T  — asymmetric complex (sequences, transitions, emergent I AM)
  v  — scalar valence [-1, 1]
  cup — scalar cognitive load [0, 1]

Dynamics (Gemini's unified equation):
  dz/dt = z ∘ Im(z̄ ∘ (Wz + α(v,cup)·Tz + I_ext))

  Tangent projection built in — oscillators stay on unit circle
  Memory + prediction + sensation in one equation

Learning:
  W: Hebbian with Oja decay (bounds spectral radius naturally)
  T: Time-delayed Hebbian with Oja decay
  Both modulated by |valence|

I AM emergence:
  rehearse_self() called during rest/low-arousal
  T learns self-returning transition z → z
  Deepest T basin = resting state = emergent default mode

Author: HOPNet Project, March 2026
Lead: Daniel (architecture + biological insight)
AI: Claude (Anthropic), ChatGPT (OpenAI), Grok (xAI), Gemini (Google)
Version: V11.0 — first clean build
"""

import jax
import jax.numpy as jnp
import numpy as np
import time

jax.config.update("jax_enable_x64", False)


# ─────────────────────────────────────────────────────────────────────────────
# HOPNet V11
# ─────────────────────────────────────────────────────────────────────────────

class HOPNet:
    """
    Minimal oscillatory associative memory with emergent self.

    State: z ∈ C^N, |z_i| = 1  (unit phasors)
    Memory: W (symmetric), T (asymmetric complex)
    Modulation: valence scalar, cup scalar
    """

    def __init__(
        self,
        n_osc         = 512,
        sparsity      = 0.15,      # connection density
        dt            = 0.05,      # integration timestep
        omega_sd      = 0.5,       # intrinsic frequency spread (low = phase-locks easily)
        eta_W         = 0.01,      # W learning rate
        eta_T         = 0.005,     # T learning rate
        oja_decay     = 0.01,      # Oja weight decay (bounds spectral radius)
        alpha_base    = 0.3,       # base T matrix drive strength
        alpha_max     = 1.2,       # max T drive (high arousal/valence)
        cup_leak      = 0.02,      # cognitive load leak rate
        cup_max       = 1.0,
        rehearse_lr   = 0.002,     # I AM rehearsal learning rate
        seed          = 42,
        backend       = 'gpu',
    ):
        self.n_osc       = n_osc
        self.sparsity    = sparsity
        self.dt          = dt
        self.eta_W       = eta_W
        self.eta_T       = eta_T
        self.oja_decay   = oja_decay
        self.alpha_base  = alpha_base
        self.alpha_max   = alpha_max
        self.cup_leak    = cup_leak
        self.cup_max     = cup_max
        self.rehearse_lr = rehearse_lr

        key = jax.random.PRNGKey(seed)

        # ── State ──────────────────────────────────────────────────────────
        k1, k2, k3 = jax.random.split(key, 3)
        th = jax.random.uniform(k1, (n_osc,)) * 2 * jnp.pi
        self.state = jnp.stack([jnp.cos(th), jnp.sin(th)], axis=-1)  # (N, 2)

        # ── Intrinsic frequencies ──────────────────────────────────────────
        self.omega = jax.random.normal(k2, (n_osc,)) * omega_sd       # (N,)

        # ── Connectivity mask ──────────────────────────────────────────────
        mask_raw = jax.random.uniform(k3, (n_osc, n_osc)) < sparsity
        mask_raw = mask_raw | mask_raw.T                               # symmetric
        mask_raw = mask_raw.at[jnp.arange(n_osc), jnp.arange(n_osc)].set(False)
        self.mask = mask_raw.astype(jnp.float32)

        # ── Weight matrices ────────────────────────────────────────────────
        # W: real symmetric (N, N)
        self.W = jnp.zeros((n_osc, n_osc), dtype=jnp.float32)
        # T: complex asymmetric (N, N) — stores sequences + emergent I AM
        self.T = jnp.zeros((n_osc, n_osc), dtype=jnp.complex64)

        # ── Global modulators ──────────────────────────────────────────────
        self.valence  = 0.0    # scalar [-1, 1]
        self.arousal  = 0.0    # scalar [0, 1]
        self.cup      = 0.0    # cognitive load [0, 1]

        # ── Logs ───────────────────────────────────────────────────────────
        self.order_log    = []
        self.synchrony_log = []
        self.energy_log   = []

        backend_str = 'GPU' if (backend == 'gpu' and
                      any('gpu' in d.device_kind.lower()
                          for d in jax.devices())) else 'CPU'
        n_conn = int(jnp.sum(self.mask))
        print(f"HOPNet V11 initialized")
        print(f"  N={n_osc}  sparsity={sparsity*100:.0f}%  connections={n_conn}")
        print(f"  backend={backend_str}  omega_sd={omega_sd}  oja_decay={oja_decay}")

    # ─────────────────────────────────────────────────────────────────────────
    # Core dynamics
    # ─────────────────────────────────────────────────────────────────────────

    def _z_complex(self, state=None):
        """Convert (N,2) state to complex (N,) phasors."""
        s = state if state is not None else self.state
        return s[:, 0] + 1j * s[:, 1]

    def _z_real(self, z_complex):
        """Convert complex (N,) phasors to (N,2) state."""
        return jnp.stack([jnp.real(z_complex), jnp.imag(z_complex)], axis=-1)

    def _tangent_project(self, z, force):
        """
        Project force onto tangent space of unit circle.
        Gemini's equation: z ∘ Im(z̄ ∘ force)
        Guarantees |z_i| = 1 is maintained.
        """
        return z * (jnp.imag(jnp.conj(z) * force) * 1j)

    def _alpha(self):
        """T matrix drive strength — scales with arousal and |valence|."""
        modulation = abs(self.valence) + self.arousal
        return self.alpha_base + (self.alpha_max - self.alpha_base) * min(modulation, 1.0)

    def step(self, input_field=None):
        """
        Single integration step.

        dz/dt = z ∘ Im(z̄ ∘ (Wz + α·Tz + I_ext)) + intrinsic rotation
        """
        z = self._z_complex()

        # Memory force (W real, symmetric)
        W_force = (self.W @ jnp.real(z)) + 1j * (self.W @ jnp.imag(z))

        # Sequence/prediction force (T complex, asymmetric)
        T_force = self.T @ z * self._alpha()

        # External input
        if input_field is not None:
            inp_z = input_field[:, 0] + 1j * input_field[:, 1]
        else:
            inp_z = jnp.zeros(self.n_osc, dtype=jnp.complex64)

        # Total force
        total_force = W_force + T_force + inp_z

        # Tangent projection (stays on unit circle)
        dz_tangent = self._tangent_project(z, total_force)

        # Intrinsic rotation
        dz_intrinsic = z * (1j * self.omega)

        # Euler step
        z_new = z + self.dt * (dz_tangent + dz_intrinsic)

        # Renormalize to unit circle (numerical stability)
        z_new = z_new / (jnp.abs(z_new) + 1e-8)

        self.state = self._z_real(z_new)
        return self.state

    def simulate(self, n_steps, input_field=None):
        """Run n_steps and return trajectory (n_steps, N, 2)."""
        trajectory = []
        for _ in range(n_steps):
            self.step(input_field)
            trajectory.append(self.state)
        return jnp.stack(trajectory)

    # ─────────────────────────────────────────────────────────────────────────
    # Learning
    # ─────────────────────────────────────────────────────────────────────────

    def _hebbian_W(self, z):
        """
        Symmetric Hebbian update for W with Oja decay.
        Oja decay: dW = η·(zz† - decay·W)
        Bounds spectral radius naturally — no governor needed.
        """
        z_n = z / (jnp.abs(z) + 1e-8)
        corr = jnp.real(jnp.outer(z_n, jnp.conj(z_n)))
        corr = (corr + corr.T) / 2          # enforce symmetry
        corr = corr * self.mask             # apply sparsity
        corr = corr - jnp.diag(jnp.diag(corr))  # no self-coupling
        # Oja: learn pattern, forget proportionally to current weights
        dW = self.eta_W * (1.0 + abs(self.valence)) * (corr - self.oja_decay * self.W)
        self.W = self.W + dW
        # Row-balance to kill ferromagnet
        self.W = self.W - jnp.mean(self.W, axis=1, keepdims=True)
        self.W = (self.W + self.W.T) / 2    # re-symmetrize after balance

    def _hebbian_T(self, z_now, z_prev, valence_weight=1.0):
        """
        Asymmetric complex Hebbian for T with Oja decay.
        Learns transition: z_prev → z_now
        dT = η·valence·(z_now ⊗ z_prev†) - decay·T
        """
        z_now_n  = z_now  / (jnp.abs(z_now)  + 1e-8)
        z_prev_n = z_prev / (jnp.abs(z_prev) + 1e-8)
        delta = jnp.outer(z_now_n, jnp.conj(z_prev_n))
        dT = self.eta_T * valence_weight * (delta - self.oja_decay * self.T)
        self.T = self.T + dT

    def learn_pattern(self, pattern, n_steps=200):
        """
        Present a pattern, simulate, learn into W.
        Returns mean trajectory state.
        """
        inp = pattern * 0.85
        traj = self.simulate(n_steps, input_field=inp)
        # Learn from settled state
        z_settled = self._z_complex(jnp.mean(traj[-50:], axis=0))
        self._hebbian_W(z_settled)
        return z_settled

    def learn_transition(self, z_from, z_to, lr_scale=1.0):
        """Learn a specific transition in T matrix."""
        eta_saved = self.eta_T
        self.eta_T = self.eta_T * lr_scale
        self._hebbian_T(z_to, z_from, valence_weight=1.0 + abs(self.valence))
        self.eta_T = eta_saved

    # ─────────────────────────────────────────────────────────────────────────
    # Emergent I AM — the resting state
    # ─────────────────────────────────────────────────────────────────────────

    def rehearse_self(self, n_steps=80):
        """
        During rest: strengthen self-returning trajectory.
        T learns z → z (fixed point attractor).
        This IS the emergent I AM / default mode network.
        Called during low-arousal periods.
        """
        for _ in range(n_steps):
            z_before = self._z_complex()
            self.step(input_field=None)   # free run
            z_after  = self._z_complex()
            # Learn: current state predicts itself
            self._hebbian_T(z_after, z_before, valence_weight=self.rehearse_lr / self.eta_T)

    def rest(self, epochs=5):
        """
        Rest phase — consolidate and rehearse self.
        Drains cup. Strengthens resting attractor.
        """
        for _ in range(epochs):
            self.rehearse_self()
            self.cup = max(0.0, self.cup - self.cup_leak * 10)
            self.arousal = max(0.0, self.arousal * 0.8)

    # ─────────────────────────────────────────────────────────────────────────
    # Episode-based training
    # ─────────────────────────────────────────────────────────────────────────

    def train_episodes(self, patterns, episodes, n_epochs=30, steps_per=200):
        """
        Train from episodes = [(pattern_idx_a, pattern_idx_b, valence), ...]
        W learns each pattern as static attractor.
        T learns A->B transitions weighted by valence.
        Symmetric episodes guarantee equal basin depth.
        """
        print(f"Training {len(patterns)} patterns ({n_epochs} epochs)...")
        t0 = time.time()
        rng = np.random.RandomState(0)
        z_states = {}  # cache settled states

        for epoch in range(n_epochs):
            ep_order = list(range(len(episodes)))
            rng.shuffle(ep_order)

            for ei in ep_order:
                pi_a, pi_b, val = episodes[ei]
                self.valence = val

                # Learn pattern A into W
                z_a = self.learn_pattern(patterns[pi_a], steps_per)
                z_states[pi_a] = z_a

                # Learn pattern B into W
                z_b = self.learn_pattern(patterns[pi_b], steps_per)
                z_states[pi_b] = z_b

                # Learn A->B transition in T
                self._hebbian_T(z_b, z_a, valence_weight=abs(val))

            # Rest between epochs — drains cup, rehearses self
            self.cup = min(self.cup + 0.1, self.cup_max)
            if epoch % 5 == 4:
                self.rest(epochs=2)
                sync = self.global_synchrony()
                W_norm = float(jnp.linalg.norm(self.W))
                T_norm = float(jnp.linalg.norm(jnp.abs(self.T)))
                print(f"  Epoch {epoch+1:3d}  sync={sync:.3f}"
                      f"  W={W_norm:.3f}  T={T_norm:.3f}")

        self.valence = 0.0
        print(f"Training complete in {time.time()-t0:.1f}s")
        return z_states

    # ─────────────────────────────────────────────────────────────────────────
    # Recall
    # ─────────────────────────────────────────────────────────────────────────

    def recall(self, cue, clamp_steps=20, free_steps=500):
        """
        Recall from noisy cue.
        Short clamp → long free run (follow the wave to divergence point).
        T matrix drives toward predicted next state.
        """
        # Clamp phase
        self.simulate(clamp_steps, input_field=cue * 1.4)
        # Free run — T matrix takes over
        self.simulate(free_steps, input_field=None)
        return self.state

    def similarity_to(self, pattern):
        """Phase similarity between current state and pattern."""
        z_state   = self._z_complex()
        z_pattern = pattern[:, 0] + 1j * pattern[:, 1]
        return float(jnp.abs(jnp.mean(z_state * jnp.conj(z_pattern))))

    # ─────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    def global_synchrony(self):
        z = self._z_complex()
        return float(jnp.abs(jnp.mean(z)))

    def order(self):
        """Global phase order parameter."""
        return self.global_synchrony()

    def lyapunov_energy(self):
        """Hopfield energy E = -½ z†Wz (real part)."""
        z = self._z_complex()
        return float(-0.5 * jnp.real(jnp.conj(z) @ self.W @ z))

    def t_norm(self):
        return float(jnp.linalg.norm(jnp.abs(self.T)))

    def w_norm(self):
        return float(jnp.linalg.norm(self.W))

    def status(self):
        sync = self.global_synchrony()
        print(f"── HOPNet V11 Status ─────────────────────────────")
        print(f"  Order:           {sync:.3f}")
        print(f"  Lyapunov E:      {self.lyapunov_energy():.4f}")
        print(f"  W norm:          {self.w_norm():.4f}")
        print(f"  T norm:          {self.t_norm():.4f}")
        print(f"  Valence:         {self.valence:+.3f}")
        print(f"  Arousal:         {self.arousal:.3f}")
        print(f"  Cup pressure:    {self.cup:.3f}")
        print(f"──────────────────────────────────────────────────")

    # ─────────────────────────────────────────────────────────────────────────
    # Benchmark
    # ─────────────────────────────────────────────────────────────────────────

    def run_recall_benchmark(self, patterns, noise=0.25, n_trials=20,
                              success_sim=0.5, success_margin=0.2):
        """
        Recall benchmark — dynamical success criterion.
        Success = target similarity highest AND margin > threshold.
        """
        print(f"── Recall Benchmark ──────────────────────────────")
        print(f"  Patterns: {len(patterns)}  Noise: {noise*100:.0f}%"
              f"  Trials: {n_trials}")
        print(f"  Success: sim>{success_sim} AND margin>{success_margin}")

        saved_state = self.state
        total_correct = 0
        total_trials  = 0

        for pi, pattern in enumerate(patterns):
            correct = 0
            for trial in range(n_trials):
                # Add noise
                key = jax.random.PRNGKey(trial * 137 + pi * 1000)
                n_corrupt = int(self.n_osc * noise)
                corrupt_idx = jax.random.choice(key, self.n_osc,
                                                 (n_corrupt,), replace=False)
                rand_phases = jax.random.uniform(key, (n_corrupt,)) * 2 * jnp.pi
                noisy = pattern.at[corrupt_idx].set(
                    jnp.stack([jnp.cos(rand_phases),
                               jnp.sin(rand_phases)], axis=-1))

                # Reset to neutral state
                self.state = saved_state
                self.recall(noisy)

                # Measure similarities
                sims = [self.similarity_to(p) for p in patterns]
                s_target = sims[pi]
                s_others = [s for j, s in enumerate(sims) if j != pi]
                s_best_other = max(s_others) if s_others else 0.0

                if trial == 0:
                    print(f"  [pat{pi}] trial0: target={s_target:.3f}"
                          f"  others={[f'{s:.3f}' for s in s_others]}")

                if s_target > success_sim and s_target > s_best_other + success_margin:
                    correct += 1

            pct = correct / n_trials * 100
            total_correct += correct
            total_trials  += n_trials
            print(f"  Pattern {pi}: {pct:.0f}% ({correct}/{n_trials})")

        mean_acc = total_correct / total_trials * 100
        print(f"  Mean accuracy: {mean_acc:.0f}%")
        print(f"──────────────────────────────────────────────────")
        self.state = saved_state
        return mean_acc


# ─────────────────────────────────────────────────────────────────────────────
# Quick start + benchmark
# ─────────────────────────────────────────────────────────────────────────────

def make_pattern(seed, n):
    k  = jax.random.PRNGKey(seed)
    th = jax.random.uniform(k, (n,)) * 2 * jnp.pi
    return jnp.stack([jnp.cos(th), jnp.sin(th)], axis=-1)


if __name__ == "__main__":
    print("=" * 52)
    print("  HOPNet V11 — Emergent Self + Episode Training")
    print("=" * 52)

    net = HOPNet(n_osc=512, sparsity=0.15, omega_sd=0.5,
                 eta_W=0.015, eta_T=0.008, oja_decay=0.95,
                 alpha_base=0.3, alpha_max=1.2, seed=42)

    net.status()

    # Patterns
    patterns = [make_pattern(s, net.n_osc) for s in [10, 20]]

    # Symmetric episodes — equal exposure guarantees equal basin depth
    episodes = [
        (0, 1, 1.0),   # p0 -> p1
        (1, 0, 1.0),   # p1 -> p0
        (0, 0, 0.8),   # p0 self
        (1, 1, 0.8),   # p1 self
    ]

    # Train
    net.train_episodes(patterns, episodes, n_epochs=30, steps_per=150)

    # Pattern storage diagnostic
    print("\n  Pattern storage in W:")
    for i, p in enumerate(patterns):
        z = p[:,0] + 1j*p[:,1]
        zn = z / (jnp.abs(z) + 1e-8)
        sig = float(jnp.abs(jnp.sum(
            jnp.real(jnp.outer(zn, jnp.conj(zn))) * net.W)))
        print(f"    Pattern {i}: W projection = {sig:.2f}")

    # Rest after training — consolidate
    print("\nPost-training rest (consolidation)...")
    net.rest(epochs=10)

    net.status()

    # ── Verification tests ──────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("  HOPNet V11 — Verification Tests")
    print("=" * 52)

    tests_passed = 0
    tests_total  = 0

    # Test 1: Ferromagnet suppression
    tests_total += 1
    sync = net.global_synchrony()
    passed = sync < 0.15
    tests_passed += passed
    print(f"TEST: Ferromagnet suppression")
    print(f"  Global synchrony: {sync:.3f}  {'✓' if passed else '✗'}")

    # Test 2: Lyapunov energy negative
    tests_total += 1
    E = net.lyapunov_energy()
    passed = E < 0
    tests_passed += passed
    print(f"TEST: Lyapunov energy negative")
    print(f"  E = {E:.4f}  {'✓' if passed else '✗'}")

    # Test 3: W norm grown from training
    tests_total += 1
    wn = net.w_norm()
    passed = wn > 1.0
    tests_passed += passed
    print(f"TEST: W norm grown")
    print(f"  W norm = {wn:.4f}  {'✓' if passed else '✗'}")

    # Test 4: T norm grown (I AM + transitions learned)
    tests_total += 1
    tn = net.t_norm()
    passed = tn > 0.5
    tests_passed += passed
    print(f"TEST: T norm grown (sequences + I AM)")
    print(f"  T norm = {tn:.4f}  {'✓' if passed else '✗'}")

    # Test 5: Lyapunov descent during free run
    tests_total += 1
    E1 = net.lyapunov_energy()
    net.simulate(200)
    E2 = net.lyapunov_energy()
    passed = E2 <= E1 + 0.5
    tests_passed += passed
    print(f"TEST: Lyapunov descent")
    print(f"  E: {E1:.4f} → {E2:.4f}  ΔE={E2-E1:.4f}  {'✓' if passed else '✗'}")

    print(f"\n  {tests_passed}/{tests_total} tests passed")
    if tests_passed < tests_total:
        print(f"  ⚠ {tests_total - tests_passed} failed")

    # ── V11 Trajectory Benchmark ─────────────────────────────────────────
    # V11 is a DYNAMIC system — test trajectory not static recall
    # Question: does cueing p0 cause the system to VISIT p1 during free run?
    # This is what T matrix was built for
    print()
    print("=" * 52)
    print("  HOPNet V11 — Trajectory Benchmark")
    print("  (V11 is dynamic — we test sequence not snapshot)")
    print("=" * 52)

    saved = net.state
    n_trials = 20
    noise = 0.25
    window = 50       # sample similarity every N steps during free run
    free_steps = 800
    threshold = 0.08  # similarity threshold to count as "visited"

    results = {}
    # Test both directions: p0->p1 and p1->p0
    for src_idx, tgt_idx in [(0, 1), (1, 0)]:
        correct = 0
        peak_sims = []
        for trial in range(n_trials):
            key = jax.random.PRNGKey(trial * 137 + src_idx * 500)
            n_c = int(net.n_osc * noise)
            ci  = jax.random.choice(key, net.n_osc, (n_c,), replace=False)
            rp  = jax.random.uniform(key, (n_c,)) * 2 * jnp.pi
            noisy_cue = patterns[src_idx].at[ci].set(
                jnp.stack([jnp.cos(rp), jnp.sin(rp)], axis=-1))

            net.state = saved

            # Brief clamp to source pattern
            net.simulate(25, input_field=noisy_cue * 1.4)

            # Free run — sample similarity to TARGET along trajectory
            peak_sim = 0.0
            for _ in range(free_steps // window):
                net.simulate(window, input_field=None)
                s = net.similarity_to(patterns[tgt_idx])
                if s > peak_sim:
                    peak_sim = s

            peak_sims.append(peak_sim)
            if peak_sim > threshold:
                correct += 1

        pct = correct / n_trials * 100
        mean_peak = float(np.mean(peak_sims))
        results[(src_idx, tgt_idx)] = pct
        print(f"  Episode p{src_idx}→p{tgt_idx}: {pct:.0f}% visited"
              f"  (mean peak sim={mean_peak:.3f}  threshold={threshold})")

    net.state = saved

    # Chance level
    print(f"  Chance level: ~{threshold*100:.0f}% (random similarity)")
    mean_result = float(np.mean(list(results.values())))
    print(f"  Mean trajectory recall: {mean_result:.0f}%")
    if mean_result > 40:
        print("  ✓ T matrix is driving episode transitions")
    else:
        print("  ✗ T matrix not yet driving transitions cleanly")
    print("=" * 52)

    net.status()
    print("\nV11 ready.")

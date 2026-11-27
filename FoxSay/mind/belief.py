"""
belief.py -- the mathematics of FUNCTIONAL BEHAVIOR: belief states and
scheduling that make the model a functional agent across turns.

Neuroscience analogy:
  * BeliefState  = parietal association cortex: the brain tracks a hidden
                   world through noisy observations with Bayesian filtering.
  * DialogueState = working-memory conversation model (angular gyrus / DST):
                   the whole dialogue is compressed into a state estimate,
                   not re-read from scratch every turn.
  * HTNPlanner   = prefrontal hierarchical planning: intent -> sub-intents
                   -> tool calls, a decomposition tree like motor plans.
  * KalmanStream = streaming/online inference (like the cerebellum's
                   incremental state estimation): update, don't recompute.

Mathematics implemented:

  * POMDP belief filter:
        b'(s') = O(o | s') sum_s T(s' | s) b(s)      (then normalize)
    predict() and update() split the Bayes filter into its two exact parts.

  * Dialogue state tracking (DST):
        p(state | u_1..u_t) = p(u_t | state) p(state | u_1..u_{t-1}) / Z
    exactly the same Bayes filter, over dialogue states; summarize() gives
    MAP state + entropy (the "compression" of the conversation).

  * HTN planning: methods map a task to alternative decompositions; plan()
    expands depth-first until atomic actions (leaves), i.e. intent ->
    sub-intent -> tool call.

  * Kalman streaming: scalar steady-state filter
        predict: x^- = a x,  P^- = a^2 P + q
        update:  K = P^- / (P^- + r),  x = x^- + K (y - x^-),  P = (1-K) P^-
    steady_gain() solves the discrete algebraic Riccati equation
        P_inf = a^2 P_inf + q - (a^2 P_inf + q)^2 / (a^2 P_inf + q + r)
    by fixed-point iteration -- the exact stationary Kalman gain.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["BeliefState", "DialogueState", "HTNPlanner", "KalmanStream"]


# ---------------------------------------------------------------------------
# 1. POMDP belief filtering
# ---------------------------------------------------------------------------

class BeliefState:
    """Exact Bayes filter for a finite POMDP.

    T: (S, S) transition matrix  T[s', s] = P(s' | s)   (columns sum to 1)
    O: (S,) or (K, S) observation model for the current observation:
       O[s'] = P(o | s'). Normalization happens after every update.
    """

    def predict(self, b: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """b: (S,) -> (S,): b' = T @ b (Chapman-Kolmogorov)."""
        return T @ b

    def update(self, b: torch.Tensor, obs_model: torch.Tensor) -> torch.Tensor:
        """Bayes: b' ∝ O ⊙ b. Returns the normalized posterior."""
        post = obs_model * b
        Z = post.sum()
        return post / (Z + 1e-12)

    def filter(self, b: torch.Tensor, T: torch.Tensor,
               obs_model: torch.Tensor) -> torch.Tensor:
        """One full step: predict then update."""
        return self.update(self.predict(b, T), obs_model)


# ---------------------------------------------------------------------------
# 2. dialogue state tracking (DST)
# ---------------------------------------------------------------------------

class DialogueState:
    """The whole dialogue compressed into a state estimate (DST Bayes filter).

    States are discrete dialogue states (intents, slots); each utterance
    supplies a likelihood vector likelihood[s] = P(u_t | state). The prior is
    the previous compressed state -- NO full-history re-reading, exactly the
    "cross-turn belief update" the functional layer needs.
    """

    def __init__(self, n_states: int, uniform: bool = True):
        self.n_states = n_states
        b = torch.ones(n_states) / n_states if uniform else torch.zeros(n_states)
        self.b = b
        self.history_len = 0

    def update(self, likelihood: torch.Tensor) -> torch.Tensor:
        """likelihood: (S,) P(u_t | state). Returns the new compressed state."""
        self.b = self.b * likelihood
        Z = self.b.sum()
        self.b = self.b / (Z + 1e-12)
        self.history_len += 1
        return self.b

    def summarize(self):
        """MAP state + entropy of the compressed belief (uncertainty metric)."""
        H = -(self.b * (self.b + 1e-12).log()).sum()
        return {"map_state": int(self.b.argmax()), "entropy": float(H),
                "turns": self.history_len}


# ---------------------------------------------------------------------------
# 3. HTN hierarchical planning
# ---------------------------------------------------------------------------

class HTNPlanner:
    """Hierarchical Task Network: intent -> sub-intents -> tool calls.

    methods: dict task -> list of decompositions; each decomposition is a
    list of subtasks (strings). A task without a method entry is atomic
    (a tool call). plan(root) expands depth-first, leftmost; cyclic methods
    are cut by a depth cap (guaranteed termination).
    """

    def __init__(self, methods: dict, max_depth: int = 8):
        self.methods = methods
        self.max_depth = max_depth

    def plan(self, root: str) -> list:
        """Expand root into a flat list of atomic actions (leaves)."""
        plan = []
        stack = [(root, 0)]
        while stack:
            task, depth = stack.pop()
            if task not in self.methods or depth >= self.max_depth:
                plan.append(task)                # atomic (tool call)
                continue
            # leftmost decomposition first (reverse to keep order on the stack)
            decomps = self.methods[task]
            stack.extend((sub, depth + 1) for sub in reversed(decomps[0]))
        return plan

    def depth_of(self, root: str) -> int:
        """Maximum decomposition depth under root (tree height)."""
        if root not in self.methods:
            return 0
        return 1 + max((self.depth_of(sub) for sub in self.methods[root][0]),
                       default=0)


# ---------------------------------------------------------------------------
# 4. Kalman streaming (online inference, update-don't-recompute)
# ---------------------------------------------------------------------------

class KalmanStream:
    """Scalar steady-state Kalman filter for streaming belief updates.

    state: x, covariance: P; dynamics a, process noise q, sensor noise r.
    """

    def __init__(self, a: float = 1.0, q: float = 1e-2, r: float = 1.0):
        self.a, self.q, self.r = a, q, r
        self.x = torch.zeros(())
        self.P = torch.ones(())

    def step(self, y) -> torch.Tensor:
        """One filter step with observation y. Returns the updated estimate."""
        x_pred = self.a * self.x
        P_pred = self.a * self.a * self.P + self.q
        K = P_pred / (P_pred + self.r)
        self.x = x_pred + K * (y - x_pred)
        self.P = (1 - K) * P_pred
        return self.x

    def steady_gain(self, iters: int = 200, tol: float = 1e-10) -> float:
        """Solve the discrete algebraic Riccati equation by fixed-point
        iteration; returns the stationary Kalman gain K_inf."""
        P = torch.ones(())
        for _ in range(iters):
            P_pred = self.a * self.a * P + self.q
            K = P_pred / (P_pred + self.r)
            P_new = (1 - K) * P_pred
            if abs(float(P_new - P)) < tol:
                break
            P = P_new
        P_pred = self.a * self.a * P + self.q
        return float(P_pred / (P_pred + self.r))

    def reset(self):
        self.x = torch.zeros(())
        self.P = torch.ones(())

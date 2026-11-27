"""
communication.py -- the mathematics of the URGE TO SPEAK (expression drive).

Neuroscience analogy: this is the frontal speech-motor circuit (Broca's area,
prefrontal VOI decision loops, the dopaminergic novelty signal, and the
locus-coeruleus-like arousal rhythm that decides WHEN a message fires).

What a passive (response-only) model lacks, made explicit as mathematics:

  * VOIGate -- the value of information criterion (Howard 1966 / RL):
        VOI(m) = E_{o ~ p(o|m)}[ V(b') ] - V(b) - c(m)
    A speaker exists only if it owns a quantity for "is it worth saying".
    Neuroscience: prefrontal expected-value computation before initiating
    an utterance (Broca's area gating by prefrontal VOI).

  * ExpectedFreeEnergy -- Friston active inference, the complete form of the
    expression drive.  Saying a sentence = choosing a policy pi that lowers
    expected surprise:
        G(pi) = epistemic value + pragmatic value
              = E_q[ ln q(s) - ln p(s, o) ] + E_q[ -ln p(o | C) ]
    Decomposed exactly:
        G = -H[q(s)] + E_q[-ln p(o|s)] + KL(q(s) || p(s)) - E_q[ln p(o|C)]
    where the KL term is the "expected information gain" (curiosity), the
    second term is expected ambiguity, and the last is goal preference.
    Neuroscience: the distinction between PASSIVE emotion (free energy of the
    current state, already in emotion.py) and ACTIVE expression (expected
    free energy AFTER acting).

  * IntrinsicReward -- curiosity as information gain:
        r_int = KL( p(theta | x) || p(theta) )
    (closed form for diagonal Gaussians; 0 when the observation teaches
    nothing). Neuroscience: dopaminergic reward-prediction-error novelty.

  * ListenerModel -- the social dimension (Lewis signaling game): speaking is
    an act chosen to change the listener's ACTION. The verifier plays the
    listener; message value = expected change in the listener's score.
        value(m) = E[ score(o | m) ] - E[ score(o | no message) ]
    Grice relevance = value / cost.
    Neuroscience: mirror-neuron / theory-of-mind loop (superior temporal
    sulcus + temporo-parietal junction).

  * HawkesSpeaker -- the temporal structure of the urge (self-exciting point
    process):  lambda(t) = mu + sum_{t_i < t} alpha exp(-beta (t - t_i)).
    Speaking once EXCITES the urge to speak again (alpha), decaying at rate
    beta; a message fires when lambda crosses the threshold. O(1) recurrence:
        lambda(t + dt) = mu + (lambda(t) - mu) exp(-beta dt)  (+ alpha on fire)
    Neuroscience: noradrenergic arousal / locus coeruleus burstiness: the
    model owns "this moment I especially want to speak".

  * OptionIntent -- intent as a temporally-extended option (Sutton's Options
    framework): intent = (policy pi, termination beta), runs until its own
    termination condition, not a fixed cycle count.
    Neuroscience: basal-ganglia chunking of goal-directed behavior.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = [
    "VOIGate", "ExpectedFreeEnergy", "IntrinsicReward",
    "ListenerModel", "HawkesSpeaker", "OptionIntent",
]


# ---------------------------------------------------------------------------
# 1. VOI: the worth-of-speaking criterion
# ---------------------------------------------------------------------------

class VOIGate:
    """VOI(m) = E[V(b_post)] - V(b_prior) - cost(m).

    Works on ANY value provider (typically the TD-critic from emotion.py):
    V_prior is the value of the current belief, V_post samples are the values
    of the beliefs reachable AFTER the message. Differentiable when V is.
    """

    def __init__(self, message_cost: float = 0.0, threshold: float = 0.0):
        self.message_cost = message_cost
        self.threshold = threshold

    def vois(self, V_prior, V_post) -> torch.Tensor:
        """V_prior: scalar; V_post: (K,) samples or scalar. Returns VOI."""
        if torch.is_tensor(V_post) and V_post.dim() > 0:
            return V_post.mean() - V_prior - self.message_cost
        return V_post - V_prior - self.message_cost

    def should_speak(self, V_prior, V_post) -> bool:
        return bool(float(self.vois(V_prior, V_post)) > self.threshold)

    def batch(self, V_prior, V_post_matrix, costs=None) -> torch.Tensor:
        """V_post_matrix: (M, K) value samples for M candidate messages.
        Returns (M,) VOI per candidate message."""
        costs = costs if costs is not None else torch.full(
            (V_post_matrix.size(0),), self.message_cost, device=V_post_matrix.device)
        return V_post_matrix.mean(dim=-1) - V_prior - costs


# ---------------------------------------------------------------------------
# 2. Expected Free Energy (active inference)
# ---------------------------------------------------------------------------

class ExpectedFreeEnergy:
    """G(pi) = -H[q(s)] + E_q[-ln p(o|s)] + KL(q(s)||p(s)) - E_q[ln p(o|C)].

    Call with the four terms computed by any inference machinery; the class
    guarantees the decomposition is exact and exposes each component so the
    ACTIVE (expected surprise after acting) term stays separate from the
    PASSIVE free energy that emotion.py already tracks.
    """

    def __call__(self, neg_entropy, expected_neg_loglik, prior_kl,
                 pragmatic) -> torch.Tensor:
        """All inputs scalar tensors; returns G = epistemic + pragmatic."""
        epistemic = neg_entropy + expected_neg_loglik + prior_kl
        return epistemic + pragmatic

    def decompose(self, neg_entropy, expected_neg_loglik, prior_kl,
                  pragmatic) -> dict:
        epistemic = neg_entropy + expected_neg_loglik + prior_kl
        return {
            "epistemic": epistemic,          # the curiosity / info-seeking part
            "pragmatic": pragmatic,          # the goal-preference part
            "G": epistemic + pragmatic,
        }


# ---------------------------------------------------------------------------
# 3. intrinsic reward = information gain
# ---------------------------------------------------------------------------

class IntrinsicReward:
    """r_int = KL( p(theta|x) || p(theta) ), diagonal-Gaussian closed form.

    KL = 0.5 * sum( log(s0^2/s1^2) + (s1^2 + (m1-m0)^2)/s0^2 - 1 )
    """

    def kl_diag(self, m0, s0, m1, s1) -> torch.Tensor:
        var0, var1 = s0 ** 2, s1 ** 2
        return 0.5 * (torch.log(var0 / var1).sum() +
                      ((var1 + (m1 - m0) ** 2) / var0).sum() - m0.numel())

    def __call__(self, prior_mu, prior_sigma, post_mu, post_sigma) -> torch.Tensor:
        return self.kl_diag(prior_mu, prior_sigma, post_mu, post_sigma)


# ---------------------------------------------------------------------------
# 4. listener model (Lewis signaling)
# ---------------------------------------------------------------------------

class ListenerModel:
    """Speaking is an act aimed at changing the listener's action.

    The verifier (or any scorer) plays the listener. Message value is the
    EXPECTED CHANGE in the listener's score caused by the message; Grice
    relevance normalizes by cost.

        value(m) = E[ score(o | m) ] - E[ score(o | silence) ]
    """

    def message_value(self, listener, message=None, baseline_score=None):
        """listener: callable(message) -> scalar score (tensor OK)."""
        if baseline_score is None:
            baseline_score = listener(None)
        post_score = listener(message)
        return post_score - baseline_score

    def grice_relevance(self, value_gain, cost):
        """Relevance = effect / cost (Grice's maxim of relation)."""
        return value_gain / (cost + 1e-12)


# ---------------------------------------------------------------------------
# 5. Hawkes self-exciting speaking schedule
# ---------------------------------------------------------------------------

class HawkesSpeaker:
    """lambda(t) = mu + sum_{t_i < t} alpha exp(-beta (t - t_i)).

    O(1) recurrence between events:
        lambda(t+dt) = mu + (lambda(t) - mu) exp(-beta dt)   [no event in dt]
        lambda -> lambda + alpha                             [event fires]
    """

    def __init__(self, mu: float = 0.05, alpha: float = 0.4,
                 beta: float = 0.8, threshold: float = 1.0):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.lam = mu           # current intensity
        self.t = 0.0            # current time

    def advance(self, dt: float, fire: bool = False) -> float:
        """Evolve time by dt; fire=True means a message was emitted now."""
        self.lam = self.mu + (self.lam - self.mu) * math.exp(-self.beta * dt)
        if fire:
            self.lam += self.alpha        # self-excitation: said -> want more
        self.t += dt
        return self.lam

    def should_speak(self, dt: float) -> bool:
        return self.advance(dt) >= self.threshold

    @staticmethod
    def closed_form(t: float, events, mu: float, alpha: float, beta: float) -> float:
        """Exact lambda(t) from event times (for verification).

        Right-continuous convention: an event AT time t contributes alpha
        (lambda jumps instantly, matching advance(..., fire=True)).
        """
        lam = mu
        for ti in events:
            if ti <= t:
                lam += alpha * math.exp(-beta * (t - ti))
        return lam


# ---------------------------------------------------------------------------
# 6. intent = option (temporal abstraction with its own termination)
# ---------------------------------------------------------------------------

class OptionIntent:
    """An intent is an option <I, pi, beta>: a policy that runs until ITS OWN
    termination condition, not a fixed number of cycles. The expression drive
    is "keep expressing this intent until it is satisfied"."""

    def __init__(self, name: str, termination, max_steps: int = 16):
        self.name = name
        self.termination = termination   # callable(state) -> bool
        self.max_steps = max_steps
        self.steps = 0
        self.active = True

    def step(self, state):
        """Run one intent step: returns (done, terminated)."""
        self.steps += 1
        done = bool(self.termination(state)) or self.steps >= self.max_steps
        if done:
            self.active = False
        return done

    def is_active(self):
        return self.active

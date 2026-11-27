"""
FoxSay.mind -- the mathematical substrate of the model's mind.

One aggregate `Mind` object couples every subsystem from the math manifest:

  spectral      per-subsystem spectral regularization (RMT, sigma-reparam,
                contraction budgets) -- MULTIPLE frameworks, one per
                trunk / sub-GAN / verifier
  manifold      Riemannian nets deployed at different positions of the chain
                of thought (euclidean / fisher / poincare / frobenius);
                every position samples its own tangent-space "thought"
  emotion       TD-lambda critic (dopamine), OU valence/arousal complex
                state, variational free energy, PID homeostasis
  intuition     MCTS-UCT thinking tree, TPR combinator, modular number
                embeddings, proof-DAG message passing, equivariant layers
  memory        modern Hopfield attractors (attention as retrieval)
  causal        differentiable SCM (do / counterfactuals / chain reward)
  info          InfoNCE cross-modal mutual information
  uncertainty   snapshot ensemble + temperature calibration + MC shadow

Mind.observe(...) is called once per loop cycle; it advances the emotion
state, samples a fresh thought, and returns the modulators (temperature
scale, stop-threshold shift, attention gain).  Mind.aux_losses() returns
the differentiable regularizers to add to the training loss.

Everything is opt-in: GPTConfig(mind=True) turns it on.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import causal, emotion, info, intuition, manifold, memory, spectral, uncertainty
from . import communication, rationality, belief
from .emotion import Critic, EmotionState, FreeEnergy, Homeostat
from .intuition import Combinator, Equivariant, NumEmbed, ProofDAG, ThinkTree
from .manifold import ManifoldSchedule, ThoughtSampler
from .memory import HopfieldMemory
from .spectral import SpectralEnsemble, SpectralSystem
from .uncertainty import MCShadow, SnapshotEnsemble, TemperatureCalibration
from .communication import VOIGate, ExpectedFreeEnergy, IntrinsicReward, \
    ListenerModel, HawkesSpeaker, OptionIntent
from .rationality import FastWeights, TypeChecker, MartingaleMonitor, \
    SubmodularBound, SavageDecision
from .belief import BeliefState, DialogueState, HTNPlanner, KalmanStream

__all__ = ["Mind"]


class Mind(nn.Module):
    def __init__(self, n_embd: int, n_cycles: int = 4, subgan_ids=(),
                 aux_weights: dict = None):
        super().__init__()
        self.n_embd = n_embd
        self.n_cycles = n_cycles
        self.subgan_ids = tuple(subgan_ids)
        self.aux_weights = aux_weights or {
            "spectral": 1e-3, "info": 0.01, "attractor": 1e-4, "causal": 0.1,
        }
        # manifold nets per position
        self.schedule = ManifoldSchedule(n_cycles)
        self.sampler = ThoughtSampler(self.schedule)
        # emotion
        self.critic = Critic(n_embd)
        self.emotion = EmotionState(n_embd)
        self.free_energy = FreeEnergy()
        self.homeostat = Homeostat()
        # intuition
        self.tree = ThinkTree()
        self.combinator = Combinator(n_embd)
        self.numbers = NumEmbed()
        self.dag = ProofDAG(n_embd)
        self.equiv = Equivariant(n_embd)
        # memory / causal / info / uncertainty
        self.hopfield = HopfieldMemory(n_embd)
        self.causal = causal.CausalChain(n_embd)
        self.contrast = info.ModalityContrast()
        self.calib = TemperatureCalibration()
        self.mc = MCShadow(n_embd)
        # multiple spectral frameworks: one per sub-GAN + one for the trunk
        self.spectral = SpectralEnsemble()
        self._snapshotter = None  # bound lazily to the GPT model
        # -- expression drive (communication decision theory) ---------------
        self.voi = VOIGate()
        self.efe = ExpectedFreeEnergy()
        self.intrinsic = IntrinsicReward()
        self.listener = ListenerModel()
        self.speaker = HawkesSpeaker()          # locus-coeruleus-like rhythm
        self.intent: OptionIntent = None
        # -- strong rationality (probabilistic-logic completeness) ----------
        self.fast = FastWeights(n_embd)         # synaptic short-term plasticity
        self.typecheck = TypeChecker()
        self.martingale = MartingaleMonitor()
        self.submod = SubmodularBound()
        self.savage = SavageDecision()
        # -- functional layer (belief states & scheduling) ------------------
        self.belief = BeliefState()
        self.dialogue = DialogueState(n_states=8)
        self.htn = HTNPlanner({})
        self.kalman = KalmanStream()
        # per-cycle bookkeeping
        self._last_delta = torch.zeros(())
        self._prev_state = None

    # -- wiring ------------------------------------------------------------
    def bind_spectral(self, model, subgan_cluster=None):
        """One spectral framework per subsystem (the multi-framework design)."""
        self.spectral.add(
            SpectralSystem("trunk", rho=0.95, sigma=1.1).register(model))
        if subgan_cluster is not None:
            for name, sg in subgan_cluster.subgans.items():
                self.spectral.add(
                    SpectralSystem(name, rho=0.5, sigma=0.6).register(sg))
        return self

    def bind_snapshots(self, model):
        self._snapshotter = SnapshotEnsemble(model)

    # -- per-cycle interface ------------------------------------------------
    def observe(self, x, cycle: int, reward=None, subgan_id=None,
                depth_fraction: float = 1.0, r_process=None):
        """One mind step per loop cycle.

        Returns modulators dict:
          temperature_scale, stop_shift, attention_gain, thought
        where `thought` is a fresh tangent-space sample (different every
        call -- every position of the chain produces its own thought).
        """
        s = x.detach().float().mean(dim=(0, 1))  # pooled stream state
        # dopamine: TD error (use process reward as r if no task reward)
        r = 0.0 if reward is None else float(reward)
        delta = self.critic.update(s.unsqueeze(0), torch.tensor(r),
                                   s.unsqueeze(0))
        self._last_delta = delta.detach()
        self.emotion.update(delta)
        # Hebbian fast weights: intuition temporarily GROWN on the slow weights
        # (synaptic short-term plasticity) from the previous to current state
        if self._prev_state is not None:
            self.fast.hebbian_update(self._prev_state.unsqueeze(0), s.unsqueeze(0))
        self._prev_state = s
        # the urge rhythm: each cycle advances the self-exciting Hawkes clock
        urge = self.speaker.advance(1.0)
        # homeostat modulates process-reward weight by arousal error
        w_mod = 1.0 + self.homeostat(measurement=float(self.emotion.arousal))
        # a fresh thought, sampled in this position's manifold
        thought = self.sampler(s.unsqueeze(0), cycle, depth_fraction,
                               subgan_id=subgan_id).squeeze(0)
        return {
            "temperature_scale": float(self.emotion.temperature_scale()),
            "stop_shift": float(self.emotion.stop_threshold_shift()),
            "attention_gain": float(self.emotion.attention_gain()),
            "thought": thought,
            "delta": float(delta.detach().mean()),
            "w_process_mod": w_mod,
            "urge": float(urge),      # Hawkes intensity: the wish-to-speak clock
        }

    def modulators(self):
        return {
            "temperature_scale": float(self.emotion.temperature_scale()),
            "stop_shift": float(self.emotion.stop_threshold_shift()),
            "attention_gain": float(self.emotion.attention_gain()),
        }

    def speak_decision(self, V_prior, V_post, dt=None):
        """Active-speaking gate (functional entry point).

        Combines BOTH drives from communication.py:
          * VOI (prefrontal worth-of-speaking): speak iff value gain > cost
          * Hawkes rhythm (arousal clock): the urge must be hot right now
        Returns (voi, urge, speak).
        """
        V_prior_t = V_prior if torch.is_tensor(V_prior) else torch.tensor(float(V_prior))
        V_post_t = V_post if torch.is_tensor(V_post) else torch.tensor(float(V_post))
        voi = float(self.voi.vois(V_prior_t, V_post_t))
        urge = self.speaker.advance(0.0 if dt is None else dt)
        speak = voi > self.voi.threshold and urge >= self.speaker.threshold
        return voi, urge, speak

    def stop_threshold_shift(self):
        return float(self.emotion.stop_threshold_shift())

    # -- losses --------------------------------------------------------------
    def aux_losses(self):
        """Differentiable regularizers to add to the main training loss."""
        losses = {}
        if self.spectral.systems:
            l = self.spectral.spectral_loss()
            if l is not None:
                losses["spectral"] = self.aux_weights["spectral"] * l
        if self._prev_state is not None:
            losses["attractor"] = self.aux_weights["attractor"] * \
                self.hopfield.attractor_loss(self._prev_state.unsqueeze(0))
        c = self.contrast.contrast()
        if c is not None:
            losses["info"] = self.aux_weights["info"] * c
        return losses

    def spectral_project(self):
        self.spectral.project()

    def snapshot(self):
        if self._snapshotter is not None:
            self._snapshotter.snapshot()

    def init_weights(self):
        """Call from GPT.init_weights (meta-safe: real tensors by then)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.hopfield.init_weights()
        self.emotion.rho = 0.0
        self.emotion.phi = 0.0

"""A small deterministic state-space testbed with addressable message gates.

This is an executable Stage-A reference model, not the proposal's large model
or a trained result.  It exposes the same conceptual routing contract: current
peripheral information reaches neural state only through the ascending matrix;
current neural information reaches the peripheral state only through the
descending matrix; shared context has no direct current-peripheral bypass
(P0089-P0094).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .errors import ContractError, FutureLeakageError
from .interventions import (
    VirtualIntervention,
    apply_message_intervention,
    require_safe_rescue_source,
    validate_gate_gain,
    validate_future_messages,
)


@dataclass(frozen=True)
class State:
    neural: np.ndarray
    peripheral: np.ndarray
    shared: np.ndarray


@dataclass(frozen=True)
class Simulation:
    neural: np.ndarray
    peripheral: np.ndarray
    shared: np.ndarray


Controller = Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]


class CoupledLinearStateSpace:
    """Compact CPU reference dynamics for gate and rescue tests."""

    def __init__(
        self,
        neural_dim: int,
        peripheral_dim: int,
        shared_dim: int = 1,
        *,
        seed: int = 0,
        matrices: dict[str, np.ndarray] | None = None,
    ) -> None:
        if min(neural_dim, peripheral_dim, shared_dim) <= 0:
            raise ContractError("state dimensions must be positive")
        self.neural_dim = int(neural_dim)
        self.peripheral_dim = int(peripheral_dim)
        self.shared_dim = int(shared_dim)
        self.seed = int(seed)
        self.matrices = matrices or self._random_matrices(self.seed)
        self._validate_matrices()

    def _random_matrices(self, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)

        def stable(rows: int, columns: int, scale: float = 0.15) -> np.ndarray:
            return rng.normal(0.0, scale, size=(rows, columns))

        identity_n = np.eye(self.neural_dim) * 0.75
        identity_p = np.eye(self.peripheral_dim) * 0.70
        identity_c = np.eye(self.shared_dim) * 0.65
        return {
            "a_neural": identity_n + stable(self.neural_dim, self.neural_dim, 0.03),
            "a_peripheral": identity_p + stable(self.peripheral_dim, self.peripheral_dim, 0.03),
            "a_shared": identity_c + stable(self.shared_dim, self.shared_dim, 0.03),
            "b_ascending": stable(self.neural_dim, self.peripheral_dim, 0.08),
            "b_descending": stable(self.peripheral_dim, self.neural_dim, 0.08),
            "b_neural_shared": stable(self.shared_dim, self.neural_dim, 0.06),
            "b_shared_neural": stable(self.neural_dim, self.shared_dim, 0.06),
            "b_external_neural": stable(self.neural_dim, 1, 0.04),
            "b_external_shared": stable(self.shared_dim, 1, 0.04),
            "b_external_peripheral": stable(self.peripheral_dim, 1, 0.04),
        }

    def _validate_matrices(self) -> None:
        shapes = {
            "a_neural": (self.neural_dim, self.neural_dim),
            "a_peripheral": (self.peripheral_dim, self.peripheral_dim),
            "a_shared": (self.shared_dim, self.shared_dim),
            "b_ascending": (self.neural_dim, self.peripheral_dim),
            "b_descending": (self.peripheral_dim, self.neural_dim),
            "b_neural_shared": (self.shared_dim, self.neural_dim),
            "b_shared_neural": (self.neural_dim, self.shared_dim),
            "b_external_neural": (self.neural_dim, 1),
            "b_external_shared": (self.shared_dim, 1),
            "b_external_peripheral": (self.peripheral_dim, 1),
        }
        for name, expected in shapes.items():
            if name not in self.matrices or np.asarray(self.matrices[name]).shape != expected:
                raise ContractError(f"matrix {name} must have shape {expected}")
            if not np.isfinite(np.asarray(self.matrices[name], dtype=float)).all():
                raise ContractError(f"matrix {name} must contain only finite values")

    def encode_history(
        self,
        neural_history: np.ndarray,
        peripheral_history: np.ndarray,
        *,
        history_times_sec: np.ndarray | None = None,
        observation_cutoff_sec: float | None = None,
        shared_history: np.ndarray | None = None,
    ) -> State:
        neural = np.asarray(neural_history, dtype=float)
        peripheral = np.asarray(peripheral_history, dtype=float)
        if neural.ndim != 2 or neural.shape[1] != self.neural_dim or neural.shape[0] == 0:
            raise ContractError("neural_history must have shape (time, neural_dim)")
        if peripheral.ndim != 2 or peripheral.shape[1] != self.peripheral_dim or peripheral.shape[0] != neural.shape[0]:
            raise ContractError("peripheral_history must align with neural_history")
        if not np.isfinite(neural).all() or not np.isfinite(peripheral).all():
            raise ContractError("history values must be finite")
        if history_times_sec is not None:
            times = np.asarray(history_times_sec, dtype=float)
            if times.shape != (neural.shape[0],):
                raise ContractError("history_times_sec does not align with history arrays")
            if not np.isfinite(times).all():
                raise ContractError("history_times_sec must contain only finite values")
            if observation_cutoff_sec is not None and not np.isfinite(observation_cutoff_sec):
                raise ContractError("observation_cutoff_sec must be finite")
            if observation_cutoff_sec is not None and np.any(times >= observation_cutoff_sec):
                raise FutureLeakageError("history contains data at or after observation_cutoff_sec")
        if shared_history is None:
            shared = np.zeros((neural.shape[0], self.shared_dim), dtype=float)
        else:
            shared = np.asarray(shared_history, dtype=float)
            if shared.ndim != 2 or shared.shape != (neural.shape[0], self.shared_dim):
                raise ContractError("shared_history does not align with state dimensions")
            if not np.isfinite(shared).all():
                raise ContractError("shared_history must contain only finite values")
        return State(neural[-1].copy(), peripheral[-1].copy(), shared[-1].copy())

    def restore(self, state: State) -> State:
        """Return a copy of a pre-intervention state for paired conditions."""

        self._validate_state(state)
        return State(state.neural.copy(), state.peripheral.copy(), state.shared.copy())

    def _validate_state(self, state: State) -> None:
        if state.neural.shape != (self.neural_dim,):
            raise ContractError("neural state dimension mismatch")
        if state.peripheral.shape != (self.peripheral_dim,):
            raise ContractError("peripheral state dimension mismatch")
        if state.shared.shape != (self.shared_dim,):
            raise ContractError("shared state dimension mismatch")
        if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in (state.neural, state.peripheral, state.shared)):
            raise ContractError("state values must be finite")

    def simulate(
        self,
        initial_state: State,
        peripheral_forcing: np.ndarray,
        *,
        gain_ascending: float = 1.0,
        gain_descending: float = 1.0,
        external_input: np.ndarray | None = None,
        innovations: np.ndarray | None = None,
        controller: Controller | None = None,
    ) -> Simulation:
        """Roll out dynamics using only declared future forcing inputs."""

        self._validate_state(initial_state)
        forcing = np.asarray(peripheral_forcing, dtype=float)
        if forcing.ndim != 2 or forcing.shape[1] != self.peripheral_dim:
            raise ContractError("peripheral_forcing must have shape (time, peripheral_dim)")
        if not np.isfinite(forcing).all():
            raise ContractError("peripheral_forcing must contain only finite values")
        gain_ascending = validate_gate_gain(gain_ascending, "gain_ascending")
        gain_descending = validate_gate_gain(gain_descending, "gain_descending")
        steps = forcing.shape[0]
        if external_input is None:
            external = np.zeros((steps, 1), dtype=float)
        else:
            external = np.asarray(external_input, dtype=float)
            if external.shape != (steps, 1):
                raise ContractError("external_input must have shape (time, 1)")
            if not np.isfinite(external).all():
                raise ContractError("external_input must contain only finite values")
        if innovations is None:
            noise = np.zeros((steps, self.neural_dim), dtype=float)
        else:
            noise = np.asarray(innovations, dtype=float)
            if noise.shape != (steps, self.neural_dim):
                raise ContractError("innovations must have shape (time, neural_dim)")
            if not np.isfinite(noise).all():
                raise ContractError("innovations must contain only finite values")

        matrix = self.matrices
        neural = np.zeros((steps, self.neural_dim), dtype=float)
        peripheral = np.zeros((steps, self.peripheral_dim), dtype=float)
        shared = np.zeros((steps, self.shared_dim), dtype=float)
        n = initial_state.neural.copy()
        p = initial_state.peripheral.copy()
        c = initial_state.shared.copy()
        for step in range(steps):
            message = forcing[step].copy()
            if controller is not None:
                message = np.asarray(controller(n.copy(), p.copy(), c.copy(), step), dtype=float)
                if message.shape != (self.peripheral_dim,):
                    raise ContractError("synthetic controller returned the wrong peripheral dimension")
                if not np.isfinite(message).all():
                    raise ContractError("synthetic controller returned non-finite values")
            p = np.tanh(matrix["a_peripheral"] @ p + gain_descending * matrix["b_descending"] @ n + matrix["b_external_peripheral"] @ external[step] + message)
            # Shared context does not receive current peripheral observations.
            c = np.tanh(matrix["a_shared"] @ c + matrix["b_neural_shared"] @ n + matrix["b_external_shared"] @ external[step])
            n = np.tanh(
                matrix["a_neural"] @ n
                + gain_ascending * matrix["b_ascending"] @ p
                + matrix["b_shared_neural"] @ c
                + matrix["b_external_neural"] @ external[step]
                + noise[step]
            )
            peripheral[step] = p
            shared[step] = c
            neural[step] = n
        return Simulation(neural=neural, peripheral=peripheral, shared=shared)

    def intervene(self, messages: np.ndarray, intervention: VirtualIntervention) -> tuple[np.ndarray, float, float]:
        transformed = apply_message_intervention(messages, intervention)
        return transformed, float(intervention.ascending_gain), float(intervention.descending_gain)

    def damage_control(
        self,
        state: State,
        *,
        neural_mask: np.ndarray,
        shared_mask: np.ndarray | None = None,
        neural_scale: float = 0.0,
        shared_scale: float = 0.0,
    ) -> State:
        """Apply an explicit state-space damage control with caller-supplied masks.

        The masks and scale are kept outside the protected test.  This is a
        control operation, not a claim that generic damage is mechanistically
        equivalent to closing a body-feedback gate (P0118/P0165).
        """

        self._validate_state(state)
        n_mask = np.asarray(neural_mask, dtype=bool)
        c_mask = np.zeros(self.shared_dim, dtype=bool) if shared_mask is None else np.asarray(shared_mask, dtype=bool)
        if n_mask.shape != (self.neural_dim,) or c_mask.shape != (self.shared_dim,):
            raise ContractError("damage masks do not match state dimensions")
        if not np.isfinite([neural_scale, shared_scale]).all():
            raise ContractError("damage scales must be finite")
        neural = state.neural.copy()
        shared = state.shared.copy()
        neural[n_mask] *= neural_scale
        shared[c_mask] *= shared_scale
        return State(neural, state.peripheral.copy(), shared)

    def rescue(
        self,
        initial_state: State,
        future_messages: np.ndarray,
        *,
        message_times_sec: np.ndarray,
        observation_cutoff_sec: float,
        mode: str,
        source: str,
        controller: Controller | None = None,
        phase_shift_steps: int = 0,
        allow_oracle: bool = False,
    ) -> Simulation:
        """Run an explicitly labelled rescue while guarding future-body leakage."""

        require_safe_rescue_source(source, allow_oracle=allow_oracle)
        validate_future_messages(message_times_sec, observation_cutoff_sec)
        messages = np.asarray(future_messages, dtype=float)
        if messages.ndim != 2 or messages.shape[0] != len(message_times_sec):
            raise ContractError("future_messages and message_times_sec do not align")
        if mode == "original_interface":
            return self.simulate(initial_state, messages)
        if mode == "phase_shifted":
            shifted = apply_message_intervention(
                messages,
                VirtualIntervention(name="causal_phase_shift", phase_shift_steps=phase_shift_steps),
            )
            return self.simulate(initial_state, shifted)
        if mode in {"open_loop_replay", "participant_swapped"}:
            return self.simulate(initial_state, messages)
        if mode == "closed_loop_synthetic":
            if controller is None:
                raise ContractError("closed_loop_synthetic rescue requires a controller")
            return self.simulate(initial_state, messages, controller=controller)
        if mode == "oracle_assimilation":
            if not allow_oracle or source != "actual_future_observation":
                raise FutureLeakageError("oracle assimilation must be explicitly labelled and opt-in")
            return self.simulate(initial_state, messages)
        raise ContractError(f"unknown rescue mode: {mode}")


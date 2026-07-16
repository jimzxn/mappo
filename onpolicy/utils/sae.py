"""SAE estimator used by the source MAPPO buffers.

This module is Python 3.6 compatible.  Axis 0 is always time; every remaining
axis is an independent rollout/agent lane.
"""

from collections import namedtuple

import numpy as np


SAEResult = namedtuple(
    "SAEResult",
    [
        "base_advantages",
        "selective_advantages",
        "actor_advantages",
        "value_targets",
        "td_errors",
        "selection_gate",
        "effective_lambda",
    ],
)


class SAEEstimator(object):
    """Bounded single-trace SAE with configurable actor-side output."""

    def __init__(
        self,
        gamma,
        gae_lambda,
        omega=1.0,
        alpha=0.1,
        gate="hard",
        temperature=1.0,
        blend_mode="legacy_add",
    ):
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.omega = float(omega)
        self.alpha = float(alpha)
        self.gate = gate
        self.temperature = float(temperature)
        self.blend_mode = blend_mode
        self._validate_config()

    def _validate_config(self):
        scalars = (
            self.gamma,
            self.gae_lambda,
            self.omega,
            self.alpha,
            self.temperature,
        )
        if not all(np.isfinite(value) for value in scalars):
            raise ValueError("SAE configuration must be finite")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if not 0.0 <= self.omega <= 1.0:
            raise ValueError("sae_omega must be in [0, 1]")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("sae_alpha must be in [0, 1]")
        if self.gate not in ("hard", "soft"):
            raise ValueError("sae_gate must be 'hard' or 'soft'")
        if self.temperature <= 0.0:
            raise ValueError("sae_temperature must be positive")
        if self.blend_mode not in ("legacy_add", "residual", "pure"):
            raise ValueError(
                "sae_blend_mode must be 'legacy_add', 'residual', or 'pure'"
            )

    def compute(
        self,
        rewards,
        values,
        trace_masks,
        bootstrap_masks=None,
        bad_masks=None,
        active_masks=None,
    ):
        """Return separate actor advantages and critic value targets.

        ``values`` must already be denormalized and have shape ``[T+1, ...]``.
        All transition masks have shape ``[T, ...]``.
        """
        self._validate_config()
        dtype = _result_dtype(rewards, values)
        rewards = _float_copy("rewards", rewards, dtype)
        values = _float_copy("values", values, dtype)
        if rewards.ndim < 1 or rewards.shape[0] < 1:
            raise ValueError("rewards must contain a time axis")
        expected_values = (rewards.shape[0] + 1,) + rewards.shape[1:]
        if values.shape != expected_values:
            raise ValueError(
                "values must have shape %r, got %r"
                % (expected_values, values.shape)
            )

        trace_masks = _mask_copy(
            "trace_masks", trace_masks, rewards.shape, dtype
        )
        if bootstrap_masks is None:
            bootstrap_masks = trace_masks.copy()
        else:
            bootstrap_masks = _mask_copy(
                "bootstrap_masks", bootstrap_masks, rewards.shape, dtype
            )
        if bad_masks is not None:
            bad_masks = _mask_copy(
                "bad_masks", bad_masks, rewards.shape, dtype
            )
        if active_masks is not None:
            active_masks = _mask_copy(
                "active_masks", active_masks, rewards.shape, dtype
            )

        td_errors = (
            rewards
            + self.gamma * bootstrap_masks * values[1:]
            - values[:-1]
        )
        base_lambda = np.full(
            rewards.shape, self.gae_lambda, dtype=dtype
        )
        base_advantages = _reverse_trace(
            td_errors,
            trace_masks,
            base_lambda,
            self.gamma,
            bad_masks,
        )
        selection_gate = self._make_gate(base_advantages, dtype)
        effective_lambda = (
            self.gae_lambda
            + (1.0 - self.gae_lambda) * self.omega * selection_gate
        ).astype(dtype, copy=False)
        selective_advantages = _reverse_trace(
            td_errors,
            trace_masks,
            effective_lambda,
            self.gamma,
            bad_masks,
        )

        if self.blend_mode == "legacy_add":
            actor_advantages = (
                base_advantages + self.alpha * selective_advantages
            )
        elif self.blend_mode == "residual":
            actor_advantages = base_advantages + self.alpha * (
                selective_advantages - base_advantages
            )
        else:
            # Pure SAE intentionally ignores the historical blend coefficient.
            actor_advantages = selective_advantages.copy()
        if active_masks is not None:
            actor_advantages = actor_advantages * active_masks
            selection_gate = selection_gate * active_masks

        # SAE never changes the critic target.
        value_targets = values[:-1] + base_advantages
        return SAEResult(
            base_advantages=base_advantages.astype(dtype, copy=False),
            selective_advantages=selective_advantages.astype(
                dtype, copy=False
            ),
            actor_advantages=actor_advantages.astype(dtype, copy=False),
            value_targets=value_targets.astype(dtype, copy=False),
            td_errors=td_errors.astype(dtype, copy=False),
            selection_gate=selection_gate.astype(dtype, copy=False),
            effective_lambda=effective_lambda.astype(dtype, copy=False),
        )

    def _make_gate(self, advantages, dtype):
        if self.gate == "hard":
            return (advantages > 0.0).astype(dtype)
        positive = np.maximum(advantages, 0.0)
        return np.tanh(positive / self.temperature).astype(dtype)


def apply_sae_to_buffer(buffer, next_value, value_normalizer=None):
    """Compute SAE and atomically update a shared or separated MAPPO buffer."""
    next_value = np.asarray(next_value)
    expected_shape = buffer.value_preds[-1].shape
    if next_value.shape != expected_shape:
        raise ValueError(
            "next_value must have shape %r, got %r"
            % (expected_shape, next_value.shape)
        )
    if not np.all(np.isfinite(next_value)):
        raise ValueError("next_value contains NaN or Inf")

    raw_values = buffer.value_preds.copy()
    raw_values[-1] = next_value
    uses_normalization = bool(
        getattr(buffer, "_use_popart", False)
        or getattr(buffer, "_use_valuenorm", False)
    )
    if uses_normalization:
        if value_normalizer is None:
            raise ValueError(
                "value_normalizer is required when PopArt/ValueNorm is enabled"
            )
        values = np.asarray(value_normalizer.denormalize(raw_values))
    else:
        values = raw_values

    bad_masks = None
    if getattr(buffer, "_use_proper_time_limits", False):
        bad_masks = buffer.bad_masks[1:]
    active_masks = None
    if hasattr(buffer, "active_masks"):
        active_masks = buffer.active_masks[:-1]

    result = buffer._sae_estimator.compute(
        rewards=buffer.rewards,
        values=values,
        trace_masks=buffer.masks[1:],
        bootstrap_masks=buffer.masks[1:],
        bad_masks=bad_masks,
        active_masks=active_masks,
    )

    # Do not partially mutate a buffer if validation/computation fails.
    buffer.value_preds[-1] = next_value
    buffer.returns[:-1] = result.value_targets
    buffer.returns[-1] = values[-1]
    buffer.sae_advantages = result.actor_advantages.copy()
    buffer.sae_diagnostics = {
        "base_advantages": result.base_advantages.copy(),
        "selective_advantages": result.selective_advantages.copy(),
        "selection_gate": result.selection_gate.copy(),
        "effective_lambda": result.effective_lambda.copy(),
        "td_errors": result.td_errors.copy(),
    }
    return result


def _reverse_trace(td_errors, trace_masks, lambdas, gamma, bad_masks):
    advantages = np.zeros_like(td_errors)
    running = np.zeros_like(td_errors[0])
    for step in reversed(range(td_errors.shape[0])):
        running = (
            td_errors[step]
            + gamma * trace_masks[step] * lambdas[step] * running
        )
        if bad_masks is not None:
            running = running * bad_masks[step]
        advantages[step] = running
    return advantages


def _result_dtype(first, second):
    if np.asarray(first).dtype == np.float64:
        return np.float64
    if np.asarray(second).dtype == np.float64:
        return np.float64
    return np.float32


def _float_copy(name, value, dtype):
    try:
        array = np.array(value, dtype=dtype, copy=True)
    except (TypeError, ValueError) as error:
        raise TypeError("%s must be numeric" % name) from error
    if not np.all(np.isfinite(array)):
        raise ValueError("%s contains NaN or Inf" % name)
    return array


def _mask_copy(name, value, expected_shape, dtype):
    array = _float_copy(name, value, dtype)
    if array.shape != expected_shape:
        raise ValueError(
            "%s must have shape %r, got %r"
            % (name, expected_shape, array.shape)
        )
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("%s must be in [0, 1]" % name)
    return array

from dataclasses import dataclass
from typing import Optional


@dataclass
class MembraneReset:

    reset_value: float = 0.0

    def __call__(self, spikes, state, spike_threshold):
        new_state = state.copy()
        new_state["v_mem"] = (
            new_state["v_mem"] * (spikes == 0).float() + self.reset_value
        )
        return new_state


@dataclass
class MembraneSubtract:

    subtract_value: Optional[float] = None

    def __call__(self, spikes, state, threshold):
        new_state = state.copy()
        if self.subtract_value is not None:
            new_state["v_mem"] = new_state["v_mem"] - spikes * self.subtract_value
        else:
            new_state["v_mem"] = new_state["v_mem"] - spikes * threshold
        return new_state

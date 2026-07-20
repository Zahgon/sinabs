import math
from dataclasses import dataclass

import torch


@dataclass
class Heaviside:

    window: float = 1.0

    def __call__(self, v_mem, spike_threshold):
        return ((v_mem >= (spike_threshold - self.window)).float()) / spike_threshold


def gaussian(x: torch.Tensor, mu: float, sigma: float):
    pass


@dataclass
class Gaussian:

    mu: float = 0.0
    sigma: float = 0.5
    grad_scale: float = 1.0

    def __call__(self, v_mem, spike_threshold):
        return (
            gaussian(x=v_mem - spike_threshold, mu=self.mu, sigma=self.sigma)
            * self.grad_scale
        )


@dataclass
class MultiGaussian:

    mu: float = 0.0
    sigma: float = 0.5
    h: float = 0.15
    s: float = 6
    grad_scale: float = 1.0

    def __call__(self, v_mem, spike_threshold):
        return (
            (1 + self.h)
            * gaussian(x=v_mem - spike_threshold, mu=self.mu, sigma=self.sigma)
            - self.h
            * gaussian(
                x=v_mem - spike_threshold, mu=self.sigma, sigma=self.s * self.sigma
            )
            - self.h
            * gaussian(
                x=v_mem - spike_threshold, mu=-self.sigma, sigma=self.s * self.sigma
            )
        ) * self.grad_scale


@dataclass
class SingleExponential:

    grad_width: float = 0.5
    grad_scale: float = 1.0

    def __call__(self, v_mem, spike_threshold):
        abs_width = spike_threshold * self.grad_width
        return (
            self.grad_scale
            / abs_width
            * torch.exp(-torch.abs(v_mem - spike_threshold) / abs_width)
        )


@dataclass
class PeriodicExponential:

    grad_width: float = 0.5
    grad_scale: float = 1.0

    def __call__(self, v_mem, spike_threshold):
        vmem_normalized = v_mem / spike_threshold - 0.5

        vmem_periodic = vmem_normalized - torch.floor(vmem_normalized)
        vmem_periodic = spike_threshold * (2 * vmem_periodic - 1)

        vmem_below = (v_mem - spike_threshold) * (v_mem < spike_threshold)
        vmem_above = vmem_periodic * (v_mem >= spike_threshold)
        vmem_new = vmem_above + vmem_below

        surrogate = torch.exp(-torch.abs(vmem_new) / self.grad_width)

        return self.grad_scale * surrogate

import torch.autograd


class Quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


class StochasticRounding(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass

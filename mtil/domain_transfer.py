"""Tools for domain transfer/domain invariance losses."""

import torch
from torch import nn
import torch.nn.functional as F


class ReverseGrad(torch.autograd.Function):
    """This layer acts like the identity function on the forward pass, but
    reverses gradients on the backwards pass."""
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, dydx):
        return -dydx


reverse_grad = ReverseGrad.apply


def make_binary_domain_classifier(in_chans,
                                  hidden_chans=256,
                                  ActivationCls=nn.ReLU):
    """Simple MLP for domain classification (no gradient reversal)."""
    return nn.Sequential(
        nn.Linear(in_chans, hidden_chans),
        ActivationCls(),
        nn.Linear(hidden_chans, 1),
    )


class BinaryDomainLossModule(nn.Module):
    """Combines gradient reversal -> domain classifier -> (xent loss, acc.)."""
    def __init__(self, in_chans, **kwargs):
        super().__init__()
        self.classifier = make_binary_domain_classifier(in_chans, **kwargs)

    def forward(self, x, labels_ints):
        assert labels_ints.shape == (x.shape[0], )
        assert ((labels_ints == 0) | (labels_ints == 1)).all()
        rev_x = reverse_grad(x)
        logits = self.classifier(rev_x)
        logits = logits.squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits,
                                                  labels_ints,
                                                  reduction='mean')
        pred_labels = (logits >= 0).to(torch.long)
        acc = torch.mean((pred_labels == labels_ints).to(torch.float))
        return loss, acc


def test_grad_rev():
    def fn(u):
        return 0.5 * u**2 - u

    def fn_deriv(u):
        return u - 1

    for val in [-1.0, 0.0, 4.0]:
        # test normal forward pass
        x = torch.tensor(val, requires_grad=True)
        y = fn(x)
        y.backward()
        real_grad = fn_deriv(x.detach())
        assert torch.allclose(x.grad, real_grad), (x.grad, real_grad)

        # test reversed forward pass
        x_rev = torch.tensor(val, requires_grad=True)
        y_rev = fn(reverse_grad(x_rev))
        y_rev.backward()
        assert torch.allclose(x_rev.grad, -real_grad), (x_rev.grad, -real_grad)

        # also sanity check
        assert torch.allclose(x_rev.grad, -x.grad), (x_rev.grad, x.grad)

        # test adding the two
        x_joint = torch.tensor(val, requires_grad=True)
        y_joint = fn(x_joint)
        y_joint_rev = fn(reverse_grad(x_joint))
        (y_joint + y_joint_rev).backward()
        assert torch.allclose(x_joint.grad, 0.0), x_joint.grad

    print("Done, tests pass!")


if __name__ == '__main__':
    test_grad_rev()

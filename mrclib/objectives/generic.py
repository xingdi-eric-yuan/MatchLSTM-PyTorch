# encoding: utf-8
from __future__ import absolute_import
from __future__ import print_function
import logging
import torch

logger = logging.getLogger(__name__)
_eps = 1e-6


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


class StandardNLL(torch.nn.modules.loss._Loss):
    """
    Shape:
        - Input:    batch x time x class
        - Target:   batch x time x class
    """
    def forward(self, y_pred, y_true):
        _assert_no_grad(y_true)
        P = y_true.float() * y_pred  # batch x time x class
        P = torch.sum(P, dim=1)  # batch x class
        epsilon = (torch.lt(P, 0.0).float() + torch.eq(P, 0.0).float()) * _eps  # batch x class
        log_P = torch.log(P + epsilon)  # batch x class
        sum_log_P = torch.sum(log_P, dim=1)  # n_b
        return -sum_log_P

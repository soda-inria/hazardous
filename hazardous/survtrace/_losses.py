import torch
import torch.nn.functional as F
from torch import Tensor

from ._utils import pad_col


class _Loss(torch.nn.Module):
    """Generic loss function.

    Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction


#### LogisticHazard Loss ####
class NLLLogistiHazardLoss(_Loss):
    """Negative log-likelihood of the hazard parametrization model.
    See `loss.nll_logistic_hazard` for details.

    Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.

    Returns:
        torch.tensor -- The negative log-likelihood.
    """

    def forward(
        self,
        phi: Tensor,
        idx_durations: Tensor,
        events: Tensor,
        reduction: str = "mean",
    ) -> Tensor:
        return nll_logistic_hazard(phi, idx_durations, events, reduction)


def _reduction(loss: Tensor, reduction: str = "mean") -> Tensor:
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    raise ValueError(
        f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'."
    )


def nll_logistic_hazard(
    phi: Tensor, idx_durations: Tensor, events: Tensor, reduction: str = "mean"
) -> Tensor:
    """Negative log-likelihood of the discrete time hazard parametrized \
        model LogisticHazard [1].

    Arguments:
        phi {torch.tensor} -- Estimates in (-inf, inf), where hazard = sigmoid(phi).
        idx_durations {torch.tensor} -- Event times represented as indices.
        events {torch.tensor} -- Indicator of event (1.) or censoring (0.).
            Same length as 'idx_durations'.
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.

    Returns:
        torch.tensor -- The negative log-likelihood.

    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time \
        Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(
            "Network output `phi` is too small for `idx_durations`."
            + f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`,"
            + f" but got `phi.shape[1] = {phi.shape[1]}`"
        )
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1, 1)
    idx_durations = idx_durations.view(-1, 1)

    y_bce = torch.zeros_like(phi).scatter(1, idx_durations, events)

    bce = F.binary_cross_entropy_with_logits(phi, y_bce, reduction="none")
    loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
    return _reduction(loss, reduction)


#### PCH Loss ####
class NLLPCHazardLoss(_Loss):
    def forward(self, y_pred, y_true, reduction="mean"):
        """Negative log-likelihood of the PC-Hazard parametrization model.
        See `loss.nll_pc_hazard_loss` for details.

        Arguments:
            reduction {string} -- How to reduce the loss.
                'none': No reduction.
                'mean': Mean of tensor.
                'sum: sum.

        Returns:
            torch.tensor -- The negative log-likelihood loss.
        """
        return nll_pc_hazard_loss(
            phi=y_pred,
            idx_durations=y_true["duration"],
            events=y_true["event"],
            interval_frac=y_true["frac_duration"],
            reduction=reduction,
        )


def log_softplus(input, threshold=-15.0):
    """Equivalent to 'F.softplus(input).log()', but for 'input < threshold',
    we return 'input', as this is approximately the same.
    Arguments:
        input {torch.tensor} -- Input tensor

    Keyword Arguments:
        threshold {float} -- Treshold for when to just return input (default: {-15.})

    Returns:
        torch.tensor -- return log(softplus(input)).
    """
    output = input.clone()
    above = input >= threshold
    output[above] = F.softplus(input[above]).log()
    return output


def nll_pc_hazard_loss(
    phi: Tensor,
    idx_durations: Tensor,
    events: Tensor,
    interval_frac: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Negative log-likelihood of the PC-Hazard parametrization model [1].

    Arguments:
        phi {torch.tensor} -- Estimates in (-inf, inf), where hazard = sigmoid(phi).
        idx_durations {torch.tensor} -- Event times represented as indices.
        events {torch.tensor} -- Indicator of event (1.) or censoring (0.).
            Same length as 'idx_durations'.
        interval_frac {torch.tensor} -- Fraction of last interval before \
            event/censoring.
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.

    Returns:
        torch.tensor -- The negative log-likelihood.
    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time \
        Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    if events.dtype is torch.bool:
        events = events.float()
    idx_durations = idx_durations.view(-1, 1)
    events = events.view(-1)
    interval_frac = interval_frac.view(-1)

    keep = idx_durations.view(-1) >= 0
    phi = phi[keep, :]
    idx_durations = idx_durations[keep, :]
    events = events[keep]
    interval_frac = interval_frac[keep]

    log_h_e = log_softplus(phi.gather(1, idx_durations).view(-1)).mul(events)
    haz = F.softplus(phi)
    scaled_h_e = haz.gather(1, idx_durations).view(-1).mul(interval_frac)
    haz = pad_col(haz, where="start")
    sum_haz = haz.cumsum(1).gather(1, idx_durations).view(-1)
    loss = -log_h_e.sub(scaled_h_e).sub(sum_haz)
    return _reduction(loss, reduction)

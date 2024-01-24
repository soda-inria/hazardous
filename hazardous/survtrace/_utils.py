import torch


def pad_col_2d(input, val=0, where="end"):
    """Pad a 2d tensor column-wise.

    Parameters
    ----------
    input : torch.tensor of shape (n_samples, n_time_steps)
        Input to pad.

    val : int, default=0
        Padding value.

    where : {'start', 'end'}, default='end'
        * If set to start, the padding is added on the left.
        * If set to end, the padding is added to the right.
    """
    if input.ndim != 2:
        raise ValueError("Only works for `phi` tensor that is 2D.")
    pad = torch.zeros_like(input[:, :1])
    if val != 0:
        pad = pad + val
    if where == "end":
        return torch.cat([input, pad], dim=1)
    elif where == "start":
        return torch.cat([pad, input], dim=1)
    raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")


def pad_col_3d(input, val=0, where="end"):
    """Pad a 3d tensor second-axis-wise.

    Parameters
    ----------
    input : torch.tensor of shape (n_samples, n_time_steps, n_events)
        Input to pad on the second axis (axis=1).

    val : int, default=0
        Padding value.

    where : {'start', 'end'}, default='end'
        * If set to start, the padding is added on the left.
        * If set to end, the padding is added to the right.
    """
    if input.ndim != 3:
        raise ValueError("Only works for `phi` tensor that is 3D.")
    pad = torch.zeros_like(input[:, :, :1])
    if val != 0:
        pad = pad + val
    if where == "end":
        return torch.cat([input, pad], dim=2)
    elif where == "start":
        return torch.cat([pad, input], dim=2)
    raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")

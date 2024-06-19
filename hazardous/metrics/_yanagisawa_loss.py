import numpy as np


class CensoredNegativeLogLikelihoodSimple:
    def __init__(self, epsilon=0.000001):
        self.espilon = epsilon

    def loss(self, pred, duration, event, time_grid):
        loss = 0
        for idx_time in range(len(time_grid) - 1):
            lower_time = time_grid[idx_time]
            upper_time = time_grid[idx_time + 1]
            mask = (lower_time < duration) & (duration <= upper_time)
            mask = mask.values
            f = pred[1, :, idx_time + 1] - pred[1, :, idx_time]
            loss -= (event * np.log(f + self.espilon))[mask].sum()
            loss -= ((1 - event) * np.log(1 - pred[1, :, idx_time + 1] + self.espilon))[
                mask
            ].sum()
        return loss / pred.shape[1]

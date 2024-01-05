import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from hazardous._comp_risks_loss import _MultinomialBinaryLoss, sum_exp_minus


def test_multinomial_binary_loss():
    """Test the multinomial binary loss function."""
    # Test the loss function with a single observation.
    y_true = np.array([1])
    y_proba = np.array([[0.1, 0.8, 0.1]])
    loss_out = np.empty(1)
    sample_weight = np.array([1])

    loss = _MultinomialBinaryLoss().loss(y_true, y_proba, sample_weight, loss_out)

    exp_p, max_value, sum_exps = sum_exp_minus(0, y_proba)

    assert max_value == 0.8
    assert sum_exps == pytest.approx(1 + 2 * np.exp(-0.7))

    exp_p = np.exp(y_proba)
    exp_p /= exp_p.sum()
    expected_loss = -np.log(np.abs(np.array([1, 0, 1]) - exp_p)).sum()

    assert loss == pytest.approx(expected_loss)


def test_multinomial_binary_gradient_hessian():
    """Test the multinomial binary gradient function."""
    # Test the gradient function with a single observation.
    y_true = np.array([1])
    y_proba = np.array([[0.1, 0.8, 0.1]])
    gradient_out = np.empty((1, 3))
    hessian_out = np.empty((1, 3))
    sample_weight = np.array([1])

    grad, hessian = _MultinomialBinaryLoss().gradient_hessian(
        y_true, y_proba, sample_weight, gradient_out, hessian_out
    )

    exp_p = np.exp(y_proba[0] - 0.8)
    exp_p /= exp_p.sum()

    neg_g1 = exp_p[1] / (1 - exp_p[0])
    neg_g2 = exp_p[1] / (1 - exp_p[2])
    pos_g = 2 * exp_p[1] - 1
    expected_grad = np.array([[neg_g1, pos_g, neg_g2]])

    neg_h1 = exp_p[1] * (1 - exp_p[0] - exp_p[1])
    neg_h2 = exp_p[1] * (1 - exp_p[2] - exp_p[1])
    pos_h = 2 * exp_p[1] * (1 - exp_p[1])
    expected_hessian = np.array([[neg_h1, pos_h, neg_h2]])

    assert_almost_equal(grad, expected_grad)
    assert_almost_equal(hessian, expected_hessian)

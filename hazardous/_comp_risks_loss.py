import numpy as np
from sklearn._loss.link import Interval, MultinomialLogit
from sklearn._loss.loss import BaseLoss


def sum_exp_minus(i, raw_prediction):
    p = raw_prediction[i]
    max_value = np.max(p)
    exp_p = np.exp(p - max_value)
    sum_exps = np.sum(exp_p)

    return exp_p, max_value, sum_exps


class _MultinomialBinaryLoss:
    """Half Multinomial deviance loss with multinomial logit link.

    Domain:
    y_true in {0, 1, 2, 3, .., n_classes - 1}
    y_pred in (0, 1)**n_classes, i.e. interval with boundaries excluded

    Link:
    y_pred = softmax(raw_prediction)

    Note: Label encoding is built-in, i.e. {0, 1, 2, 3, .., n_classes - 1} is
    mapped to (y_true == k) for k = 0 .. n_classes - 1 which is either 0 or 1.
    """

    # Note that we do not assume memory alignment/contiguity of 2d arrays.
    # There seems to be little benefit in doing so. Benchmarks proofing the
    # opposite are welcome.
    def loss(
        self,
        y_true,  # IN
        raw_prediction,  # IN
        sample_weight,  # IN
        loss_out,  # OUT
        n_threads=1,
    ):
        y_true = y_true.astype(int)
        n_samples = y_true.shape[0]
        n_classes = raw_prediction.shape[1]

        # We assume n_samples > n_classes. In this case having the inner loop
        # over n_classes is a good default.
        # TODO: If every memoryview is contiguous and raw_prediction is
        #       f-contiguous, can we write a better algo (loops) to improve
        #       performance?

        for i in range(n_samples):
            p, max_value, sum_exps = sum_exp_minus(i, raw_prediction)
            log_sum_exps = np.log(sum_exps)
            loss_out[i] = n_classes * log_sum_exps + max_value

            for k in range(1, n_classes):
                # label decode y_true
                if y_true[i] == k:
                    loss_out[i] -= raw_prediction[i, k]
                else:
                    loss_out[i] -= np.log(sum_exps - p[k])

            loss_out[i] *= sample_weight[i]

        return np.asarray(loss_out)

    def loss_gradient(
        self,
        y_true,  # IN
        raw_prediction,  # IN
        sample_weight,  # IN
        loss_out,  # OUT
        gradient_out,  # OUT
        n_threads=1,
    ):
        y_true = y_true.astype(int)
        n_samples = y_true.shape[0]
        n_classes = raw_prediction.shape[1]

        for i in range(n_samples):
            p, max_value, sum_exps = sum_exp_minus(i, raw_prediction)
            log_sum_exps = np.log(sum_exps)
            loss_out[i] = log_sum_exps + max_value
            p /= sum_exps
            for k in range(1, n_classes):
                # label decode y_true
                if y_true[i] == k:
                    loss_out[i] -= raw_prediction[i, k]
                else:
                    loss_out[i] += log_sum_exps - np.log(sum_exps - p[k])
                # p_k = y_pred_k = prob of class k
                # gradient_k = (p_k - (y_true == k)) * sw

                gradient_out[i, k] = (p[k] - 1 * (y_true[i] == k)) * sample_weight[i]
            loss_out[i] *= sample_weight[i]
        gradient_out[:, 0] = 0.0
        return np.asarray(loss_out), np.asarray(gradient_out)

    def gradient(
        self,
        y_true,  # IN
        raw_prediction,  # IN
        sample_weight,  # IN
        gradient_out,  # OUT
        n_threads=1,
    ):
        n_samples = y_true.shape[0]
        n_classes = raw_prediction.shape[1]

        for i in range(n_samples):
            p, _, sum_exps = sum_exp_minus(i, raw_prediction)
            p /= sum_exps  # p_k = y_pred_k = prob of class k
            for k in range(1, n_classes):
                gradient_out[i, k] = (p[k] - 1 * (y_true[i] == k)) * sample_weight[i]

        gradient_out[:, 0] = 0.0
        return np.asarray(gradient_out)

    def gradient_hessian(
        self,
        y_true,  # IN
        raw_prediction,  # IN
        sample_weight,  # IN
        gradient_out,  # OUT
        hessian_out,  # OUT
        n_threads=1,
    ):
        y_true = y_true.astype(int)
        n_samples = y_true.shape[0]
        n_classes = raw_prediction.shape[1]
        for i in range(n_samples):
            p, _, sum_exps = sum_exp_minus(i, raw_prediction)
            p /= sum_exps

            for k in range(1, n_classes):
                # p_k = y_pred_k = prob of class k
                # gradient_k = (p_k - (y_true == k)) * sw
                # hessian_k = p_k * (1 - p_k) * sw
                gradient_out[i, k] = (p[k] - 1 * (y_true[i] == k)) * sample_weight[i]
                hessian_out[i, k] = p[k] * (1 - p[k]) * sample_weight[i]

        gradient_out[:, 0] = 0.0
        hessian_out[:, 0] = 0.0
        return np.asarray(gradient_out), np.asarray(hessian_out)

    # This method simplifies the implementation of hessp in linear models,
    # i.e. the matrix-vector product of the full hessian, not only of the
    # diagonal (in the classes) approximation as implemented above.
    def gradient_proba(
        self,
        y_true,  # IN
        raw_prediction,  # IN
        sample_weight,  # IN
        gradient_out,  # OUT
        proba_out,  # OUT
        n_threads=1,
    ):
        y_true = y_true.astype(int)
        n_samples = y_true.shape[0]
        n_classes = raw_prediction.shape[1]

        for i in range(n_samples):
            p, _, sum_exps = sum_exp_minus(i, raw_prediction)
            p /= sum_exps
            for k in range(1, n_classes):
                proba_out[i, k] = p[k]  # y_pred_k = prob of class k
                # gradient_k = (p_k - (y_true == k)) * sw
                gradient_out[i, k] = (p[k] - 1 * (y_true[i] == k)) * sample_weight[i]
        gradient_out[:, 0] = 0.0
        return np.asarray(gradient_out), np.asarray(proba_out)


class MultinomialBinaryLoss(BaseLoss):
    """Categorical cross-entropy loss, for multiclass classification.

    Domain:
    y_true in {0, 1, 2, 3, .., n_classes - 1}
    y_pred has n_classes elements, each element in (0, 1)

    Link:
    y_pred = softmax(raw_prediction)

    Note: We assume y_true to be already label encoded. The inverse link is
    softmax. But the full link function is the symmetric multinomial logit
    function.

    For a given sample x_i, the categorical cross-entropy loss is defined as
    the negative log-likelihood of the multinomial distribution, it
    generalizes the binary cross-entropy to more than 2 classes::

        loss_i = log(sum(exp(raw_pred_{i, k}), k=0..n_classes-1))
                - sum(y_true_{i, k} * raw_pred_{i, k}, k=0..n_classes-1)

    See [1].

    Note that for the hessian, we calculate only the diagonal part in the
    classes: If the full hessian for classes k and l and sample i is H_i_k_l,
    we calculate H_i_k_k, i.e. k=l.

    Reference
    ---------
    .. [1] :arxiv:`Simon, Noah, J. Friedman and T. Hastie.
        "A Blockwise Descent Algorithm for Group-penalized Multiresponse and
        Multinomial Regression".
        <1311.6529>`
    """

    is_multiclass = True

    def __init__(self, sample_weight=None, n_classes=3):
        super().__init__(
            closs=_MultinomialBinaryLoss(),
            link=MultinomialLogit(),
            n_classes=n_classes,
        )
        self.interval_y_true = Interval(0, np.inf, True, False)
        self.interval_y_pred = Interval(0, 1, False, False)

    def in_y_true_range(self, y):
        """Return True if y is in the valid range of y_true.

        Parameters
        ----------
        y : ndarray
        """
        return self.interval_y_true.includes(y) and np.all(y.astype(int) == y)

    def fit_intercept_only(self, y_true, sample_weight=None):
        """Compute raw_prediction of an intercept-only model.

        This is the softmax of the weighted average of the target, i.e. over
        the samples axis=0.
        """
        out = np.zeros(self.n_classes, dtype=y_true.dtype)
        eps = np.finfo(y_true.dtype).eps
        for k in range(self.n_classes):
            out[k] = np.average(y_true == k, weights=sample_weight, axis=0)
            out[k] = np.clip(out[k], eps, 1 - eps)
        return self.link.link(out[None, :]).reshape(-1)

    def predict_proba(self, raw_prediction):
        """Predict probabilities.

        Parameters
        ----------
        raw_prediction : array of shape (n_samples, n_classes)
            Raw prediction values (in link space).

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Element-wise class probabilities.
        """
        return self.link.inverse(raw_prediction)

    def gradient_proba(
        self,
        y_true,
        raw_prediction,
        sample_weight=None,
        gradient_out=None,
        proba_out=None,
        n_threads=1,
    ):
        """Compute gradient and class probabilities fow raw_prediction.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : array of shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        gradient_out : None or array of shape (n_samples, n_classes)
            A location into which the gradient is stored. If None, a new array
            might be created.
        proba_out : None or array of shape (n_samples, n_classes)
            A location into which the class probabilities are stored. If None,
            a new array might be created.
        n_threads : int, default=1
            Might use openmp thread parallelism.

        Returns
        -------
        gradient : array of shape (n_samples, n_classes)
            Element-wise gradients.

        proba : array of shape (n_samples, n_classes)
            Element-wise class probabilities.
        """
        if gradient_out is None:
            if proba_out is None:
                gradient_out = np.empty_like(raw_prediction)
                proba_out = np.empty_like(raw_prediction)
            else:
                gradient_out = np.empty_like(proba_out)
        elif proba_out is None:
            proba_out = np.empty_like(gradient_out)

        return self.closs.gradient_proba(
            y_true=y_true,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            gradient_out=gradient_out,
            proba_out=proba_out,
            n_threads=n_threads,
        )

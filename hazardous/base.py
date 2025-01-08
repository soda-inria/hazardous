from .metrics._brier_score import mean_integrated_brier_score


class SurvivalMixin:
    _estimator_type = "survival"

    def score(self, X, y):
        """Return the mean of IBS for each event of interest and survival.

        This returns the negative of the mean of the Integrated Brier Score
        (IBS, a proper scoring rule) of each competing event as well as the IBS
        of the survival to any event. So, the higher the value, the better the
        model to be consistent with the scoring convention of scikit-learn to
        make it possible to use this class with scikit-learn model selection
        utilities such as ``GridSearchCV`` and ``RandomizedSearchCV``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : dict with keys "event" and "duration"
            The target values. "event" is a boolean array of shape (n_samples,)
            indicating whether the event was observed or not. "duration" is a
            float array of shape (n_samples,) indicating the time of the event
            or the time of censoring.

        Returns
        -------
        score : float
            The negative of time-integrated Brier score (IBS).

        TODO: implement time integrated NLL and use as the default for the
        .score method to match the objective function used at fit time.
        """
        y_pred = self.predict_cumulative_incidence(X)
        return -mean_integrated_brier_score(
            y_train=self.y_train_,
            y_test=y,
            y_pred=y_pred,
            time_grid=self.time_grid_,
        )

    def __sklearn_tags__(self):
        return {"requires_y": True}

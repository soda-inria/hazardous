from collections import defaultdict

from sklearn.pipeline import Pipeline


class SurvivalPipeline(Pipeline):
    """A scikit-learn pipeline for survival analysis.

    This class inherits from the scikit-learn :class:`~sklearn.pipeline.Pipeline`
    directly, and introduces prediction methods implemented in
    :class:`hazardous.SurvivalBoost`:

    - ``predict_cumulative_incidence``
    - ``predict_survival_func``

    Parameters
    ----------
    steps : list of tuples
        List of (name of step, estimator) tuples that are to be chained in
        sequential order. To be compatible with the scikit-learn API, all steps
        must define fit. All non-last steps must also define transform.
        See Combining Estimators for more details.

    transform_input : list of str, default=None
        The names of the metadata parameters that should be transformed by the
        pipeline before passing it to the step consuming it.
        This enables transforming some input arguments to fit (other than X)
        to be transformed by the steps of the pipeline up to the step which requires
        them. Requirement is defined via metadata routing. For instance,
        this can be used to pass a validation set through the pipeline.
        You can only set this if metadata routing is enabled, which you can enable
        using sklearn.set_config(enable_metadata_routing=True).

    memory : str or object with the :class:`joblib.Memory` interface, default=None
        Used to cache the fitted transformers of the pipeline.
        The last step will never be cached, even if it is a transformer.
        By default, no caching is performed. If a string is given, it is
        the path to the caching directory. Enabling caching triggers a clone of the
        transformers before fitting. Therefore, the transformer instance given
        to the pipeline cannot be inspected directly.
        Use the attribute named_steps or steps to inspect estimators within the
        pipeline. Caching the transformers is advantageous when fitting is
        time consuming. See Caching nearest neighbors for an example on how to
        enable caching.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as
        it is completed.

    See also
    --------
    :func:`~hazardous.pipeline.make_survival_pipeline`
    :func:`~sklearn.pipeline.make_pipeline`
    :class:`~sklearn.pipeline.Pipeline`
    """

    def predict_cumulative_incidence(self, X, times=None):
        Xt = self._transform(X)
        return self[-1].predict_cumulative_incidence(Xt, times)

    def predict_survival_func(self, X, times=None):
        Xt = self._transform(X)
        return self[-1].predict_survival_func(Xt, times)

    def _transform(self, X):
        for _, _, transformer in self._iter(with_final=False):
            X = transformer.transform(X)
        return X

    @property
    def time_grid_(self):
        return self[-1].time_grid_


# Private sklearn function vendored at version 1.6.0
def _name_estimators(estimators):
    """Generate names for estimators."""

    names = [
        estimator if isinstance(estimator, str) else type(estimator).__name__.lower()
        for estimator in estimators
    ]
    namecount = defaultdict(int)
    for est, name in zip(estimators, names):
        namecount[name] += 1

    for k, v in list(namecount.items()):
        if v == 1:
            del namecount[k]

    for i in reversed(range(len(estimators))):
        name = names[i]
        if name in namecount:
            names[i] += "-%d" % namecount[name]
            namecount[name] -= 1

    return list(zip(names, estimators))


def make_survival_pipeline(*steps, **kwargs):
    """Construct a :class:`~hazardous.pipeline.SurvivalPipeline` from the \
        given estimators.

    Parameters
    ----------
    *steps : list of Estimator objects
        List of the scikit-learn estimators that are chained together.

    memory : str or object with the :class:`joblib.Memory` interface, default=None
        Used to cache the fitted transformers of the pipeline.
        The last step will never be cached, even if it is a transformer.
        By default, no caching is performed. If a string is given, it is the path
        to the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer instance given
        to the pipeline cannot be inspected directly. Use the attribute named_steps
        or steps to inspect estimators within the pipeline. Caching the transformers
        is advantageous when fitting is time consuming.

    transform_input : list of str, default=None
        This enables transforming some input arguments to fit (other than X)
        to be transformed by the steps of the pipeline up to the step
        which requires them. Requirement is defined via metadata routing.
        This can be used to pass a validation set through the pipeline for instance.

        You can only set this if metadata routing is enabled, which you can enable
        using ``sklearn.set_config(enable_metadata_routing=True)``.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as
        it is completed.

    Returns
    -------
    p : SurvivalPipeline
        Returns a hazardous SurvivalPipeline object.

    See also
    --------
    :class:`~hazardous.pipeline.SurvivalPipeline`
    :func:`~sklearn.pipeline.make_pipeline`
    :class:`~sklearn.pipeline.Pipeline`
    """
    return SurvivalPipeline(_name_estimators(steps), **kwargs)

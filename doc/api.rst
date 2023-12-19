API
===

This page lists all the public functions and classes of the `hazardous`
package:

.. currentmodule:: hazardous


Estimators
----------

.. autosummary::
    :toctree: generated/
    :template: class.rst
    :nosignatures:

    GradientBoostingIncidence


Metrics
-------

.. autosummary::
    :toctree: generated/
    :template: function.rst
    :nosignatures:

    metrics.brier_score_survival
    metrics.brier_score_incidence
    metrics.integrated_brier_score_survival
    metrics.integrated_brier_score_incidence

Datasets
--------

.. autosummary::
    :toctree: generated/
    :template: function.rst
    :nosignatures:

    data.make_synthetic_competing_weibull
    data.load_seer


Inverse Probability Censoring Weight
------------------------------------

.. autosummary::
    :toctree: generated/
    :template: class.rst
    :nosignatures:

    IPCWEstimator

API
===

This page lists all the public functions and classes of the ``hazardous``
package:

.. currentmodule:: hazardous


Estimators
----------

.. autosummary::
    :toctree: generated/
    :template: class.rst
    :nosignatures:

    SurvivalBoost


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
    metrics.concordance_index_incidence
    metrics.accuracy_in_time
    metrics.km_calibration
    metrics.aj_calibration

Calibration
-----------

.. autosummary::
    :toctree: generated/
    :template: class.rst
    :nosignatures:

    metrics.KMCalibration
    metrics.AJCalibration

Datasets
--------

.. autosummary::
    :toctree: generated/
    :template: function.rst
    :nosignatures:

    data.make_synthetic_competing_weibull
    data.load_seer


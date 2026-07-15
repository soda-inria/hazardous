.. _changes:

===============
Release history
===============

.. currentmodule:: hazardous

Ongoing development
===================

Release 0.2.0
=============

* Added the AJ-calibration metrics:
    * :func:`aj_calibration`
    * :func:`aj_calibration_at_t`
    * :func:`aj_calibration_per_event`

* Updating Python version to support 3.13, 3.14 and dropping support for 3.9.
* Updating code to support the new versions of scikit-learn.


Release 0.1.0
=============

* :class:`SurvivalBoost`: Added the Survival Boost estimator.
* Added the following metrics:
    * :func:`brier_score_survival`
    * :func:`brier_score_incidence`
    * :func:`integrated_brier_score_survival`
    * :func:`integrated_brier_score_incidence`
    * :func:`concordance_index_incidence`
    * :func:`accuracy_in_time`

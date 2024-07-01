.. hazardous documentation master file, created by
   sphinx-quickstart on Fri Jun  2 10:47:19 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HΛZΛRDOUS
=========

Gradient-boosting survival analysis
-----------------------------------------------

.. container:: index-features

   * survival and competing risks

   * scikit-learn compatible

   * scalable gradient boosting

A scalable **time-to-event and competing risk prediction model**
implemented in Python.

.. container:: index-box sd-card

   **Competing risk settings**

   Predicting which event will occur, and when, from data where some of
   the events have not yet been observed:

   .. image:: competing_risk_diagram.svg

The model is **a gradient-boosting variant**, that offers prediction for
survival and competing risks settings, fully compatible with
`scikit-learn <https://scikit-learn.org>`_. It can be used with
scikit-learn tools such as pipelines, column transformers,
cross-validation, hyper-parameter search tools, etc.

.. This package will also offer neural network based estimators by leveraging
   `PyTorch <https://pytorch.org>`_ and `skorch
   <https://skorch.readthedocs.io/>`_.

This library puts a focus on predictive accuracy rather than on inference.
Quantifying the statistical association or causal effect of covariates with/on
the cumulated event incidence or instantaneous hazard rate is not in the scope
of this library at this time.

- License: MIT
- GitHub repository: https://github.com/soda-inria/hazardous
- Changelog: https://github.com/soda-inria/hazardous/blob/main/CHANGES.rst
- Status: under development, API is subject to change without notice.

.. currentmodule:: hazardous

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
   auto_examples/index
   downloading_seer

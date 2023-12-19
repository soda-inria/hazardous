.. hazardous documentation master file, created by
   sphinx-quickstart on Fri Jun  2 10:47:19 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HΛZΛRDOUS
=========

*Predictive survival and competing risks analysis in Python*

The objective of this library is to provide a Python implementation of
**time-to-event prediction models** in the presence of right-censored data.

The estimators of this library build on top of `scikit-learn
<https://scikit-learn.org>`_ components and extend the scikit-learn API to
offer dedicated prediction methods for survival and competing risks analysis.

They should be interoperable with scikit-learn tools such as pipelines, column
transformers, cross-validation, hyper-parameter seach tools, etc.

This package will also offer neural network based estimators by leveraging
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

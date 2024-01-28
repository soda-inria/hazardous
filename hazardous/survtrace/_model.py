"""
This code is a refactoring derived from RyanWangZf/SurvTRACE
also published under the MIT license with the following copyright:

    Copyright (c) 2021 Maximum
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNet
from skorch.callbacks import Callback, EarlyStopping, ProgressBar
from skorch.dataset import ValidSplit, unpack_data
from torch.optim import Adam

from hazardous.utils import get_n_events

from ._bert_layers import (
    DEFAULT_QUANTILE_HORIZONS,
    BertCLSMulti,
    BertEmbeddings,
    BertEncoder,
)
from ._encoder import SurvFeatureEncoder, SurvTargetEncoder
from ._losses import NLLPCHazardLoss
from ._utils import pad_col_3d


# Skorch hack 1: this will properly set parameters at runtime using callbacks.
# See: https://stackoverflow.com/a/60170023
class ShapeSetter(Callback):
    def on_train_begin(self, net, X=None, y=None):
        net.check_data(X, y)

        enc = SurvFeatureEncoder(
            categorical_columns=net.categorical_columns,
            numeric_columns=net.numeric_columns,
        )
        X_trans = enc.fit_transform(X)
        n_numerical_features = len(enc.numeric_columns_)
        vocab_size = enc.vocab_size_

        n_events = get_n_events(y["event"])
        n_features_in = X_trans.shape[1]

        if net.quantile_horizons is None:
            n_features_out = len(DEFAULT_QUANTILE_HORIZONS) + 1
        else:
            n_features_out = len(net.quantile_horizons) + 1

        net.set_params(
            embeddings__n_numerical_features=n_numerical_features,
            embeddings__vocab_size=vocab_size,
            cls__n_events=n_events,
            cls__n_features_in=n_features_in,
            cls__n_features_out=n_features_out,
        )


class SurvTRACE(NeuralNet):
    def __init__(
        self,
        module=None,
        criterion=None,
        optimizer=None,
        train_split=None,
        callbacks=None,
        categorical_columns=None,
        numeric_columns=None,
        quantile_horizons=None,
        batch_size=1024,
        lr=1e-3,
        optimizer__weight_decay=0,
        device="cpu",
        max_epochs=100,
        iterator_valid__batch_size=10_000,
        **kwargs,
    ):
        if module is None:
            module = _SurvTRACEModule()

        if criterion is None:
            criterion = NLLPCHazardLoss

        if optimizer is None:
            optimizer = Adam

        if train_split is None:
            # 10% of the dataset is used for validation.
            train_split = ValidSplit(0.1, stratified=True)

        # Skorch hack 2: this allows to use ShapeSetter on nested modules
        # in initialize_module().
        self._modules = ["module", "embeddings", "cls"]
        self.embeddings_ = module.embeddings
        self.cls_ = module.cls

        if callbacks is None:
            callbacks = [
                ShapeSetter(),
                ProgressBar(detect_notebook=False),
                EarlyStopping(monitor="valid_loss", patience=3, threshold=0.001),
            ]

        super().__init__(
            module=module,
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            optimizer__weight_decay=optimizer__weight_decay,
            callbacks=callbacks,
            batch_size=batch_size,
            device=device,
            max_epochs=max_epochs,
            train_split=train_split,
            # superseed batch_size
            iterator_valid__batch_size=iterator_valid__batch_size,
            **kwargs,
        )
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.quantile_horizons = quantile_horizons

    def initialize_module(self):
        """Skorch hack 3: set changed modules."""
        for module_name in self._modules:
            if module_name in ["cls", "embeddings"]:
                kwargs = self.get_params_for(module_name)
                sub_module = getattr(self, module_name + "_", None)
                sub_module = self.initialized_instance(sub_module, kwargs)
                # pylint: disable=attribute-defined-outside-init
                setattr(self.module, module_name, sub_module)
        return super().initialize_module()

    def check_data(self, X, y=None):
        if not hasattr(X, "__dataframe__"):
            raise TypeError(f"X must be a dataframe. Got {type(X)}")

        if y is None or not hasattr(y, "__dataframe__"):
            raise TypeError(f"y must be a dataframe. Got {type(y)}")

        return super().check_data(X, y)

    def get_dataset(self, X, y=None):
        if hasattr(self, "feature_encoder_"):
            X = self.feature_encoder_.transform(X)
        else:
            self.feature_encoder_ = SurvFeatureEncoder(
                categorical_columns=self.categorical_columns,
                numeric_columns=self.numeric_columns,
            )
            X = self.feature_encoder_.fit_transform(X)

        categorical_columns = self.feature_encoder_.categorical_columns_
        numeric_columns = self.feature_encoder_.numeric_columns_

        X_numerical = torch.as_tensor(
            X[numeric_columns].to_numpy(),
            dtype=torch.float,
        )
        X_categorical = torch.as_tensor(
            X[categorical_columns].to_numpy(),
            dtype=torch.long,
        )
        X = {
            "X_numerical": X_numerical,
            "X_categorical": X_categorical,
        }

        if y is not None:
            if self.quantile_horizons is None:
                quantile_horizons = DEFAULT_QUANTILE_HORIZONS
            else:
                quantile_horizons = self.quantile_horizons

            if hasattr(self, "target_encoder_"):
                # XXX: Is this even reached?
                y = self.target_encoder_.transform(y)
            else:
                self.target_encoder_ = SurvTargetEncoder(
                    quantile_horizons=quantile_horizons
                )
                y = self.target_encoder_.fit_transform(y)

            y = y.to_dict(orient="list")  # A dict of list
            y["duration"] = torch.as_tensor(y["duration"], dtype=torch.long)
            y["event"] = torch.as_tensor(y["event"], dtype=torch.long)
            y["frac_duration"] = torch.as_tensor(y["frac_duration"], dtype=torch.float)

            self.n_events = get_n_events(y["event"])

        return super().get_dataset(X, y)

    def train_step_single(self, batch, **fit_params):
        """Compute y_pred, loss value, and update net's gradients.

        The module is set to be in train mode (e.g. dropout is
        applied).

        Parameters
        ----------
        batch
          A single batch returned by the data loader.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        Returns
        -------
        step : dict
          A dictionary ``{'loss': loss, 'y_pred': y_pred}``, where the
          float ``loss`` is the result of the loss function and
          ``y_pred`` the prediction generated by the PyTorch module.

        """
        self._set_training(True)
        Xi, yi = unpack_data(batch)
        loss = 0
        all_y_pred = []
        event_multiclass = yi["event"].clone()
        for event_of_interest in range(1, self.n_events + 1):
            yi["event"] = (event_multiclass == event_of_interest).long()
            fit_params["event_of_interest"] = event_of_interest
            y_pred = self.infer(Xi, **fit_params)

            loss += self.get_loss(y_pred, yi, X=Xi, training=True)
            all_y_pred.append(y_pred[:, :, None])

        all_y_pred = torch.concatenate(
            all_y_pred, axis=2
        )  # (n_samples, n_time_steps, n_events)
        loss.backward()

        return {
            "loss": loss,
            "y_pred": all_y_pred,
        }

    def validation_step(self, batch, **fit_params):
        """Perform a forward step using batched data and return the
        resulting loss.

        The module is set to be in evaluation mode (e.g. dropout is
        not applied).

        Parameters
        ----------
        batch
          A single batch returned by the data loader.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        """
        self._set_training(False)
        Xi, yi = unpack_data(batch)
        loss = 0
        all_y_pred = []
        event_multiclass = yi["event"].clone()
        with torch.no_grad():
            for event_of_interest in range(1, self.n_events + 1):
                yi["event"] = (event_multiclass == event_of_interest).long()
                fit_params["event_of_interest"] = event_of_interest
                y_pred = self.infer(Xi, **fit_params)

                loss += self.get_loss(y_pred, yi, X=Xi, training=False)
                all_y_pred.append(y_pred[:, :, None])

        all_y_pred = torch.concatenate(
            all_y_pred, axis=2
        )  # (n_samples, n_time_steps, n_events)
        return {
            "loss": loss,
            "y_pred": all_y_pred,
        }

    def evaluation_step(self, batch, training=False):
        """Perform a forward step to produce the output used for
        prediction and scoring.

        Therefore, the module is set to evaluation mode by default
        beforehand which can be overridden to re-enable features
        like dropout by setting ``training=True``.

        Parameters
        ----------
        batch
          A single batch returned by the data loader.

        training : bool (default=False)
          Whether to set the module to train mode or not.

        Returns
        -------
        y_infer
          The prediction generated by the module.

        """
        self.check_is_fitted()
        Xi, _ = unpack_data(batch)
        all_y_pred = []
        with torch.set_grad_enabled(training):
            self._set_training(training)
            for event_of_interest in range(1, self.n_events + 1):
                y_pred = self.infer(Xi, event_of_interest=event_of_interest)
                all_y_pred.append(y_pred[:, :, None])
            all_y_pred = torch.concatenate(
                all_y_pred, axis=2
            )  # (n_samples, n_time_steps, n_events)
            return all_y_pred

    def predict_hazard(self, X):
        y_pred = self.predict(X)  # (n_samples, n_time_steps, n_events)
        y_pred = torch.from_numpy(y_pred)
        y_pred = y_pred.permute(2, 0, 1)  # (n_events, n_samples, n_time_steps)
        hazard = F.softplus(y_pred)
        hazard = pad_col_3d(hazard, where="start")
        return hazard.numpy()

    def predict_survival_function(self, X):
        hazard = self.predict_hazard(X)
        surv = np.exp(-hazard.cumsum(axis=2))
        return surv

    def predict_cumulative_incidence(self, X):
        risks = 1 - self.predict_survival_function(X)
        surv = (1 - risks.sum(axis=0))[None, :, :]
        return np.concatenate([surv, risks], axis=0)

    def score(self, X, y):
        y_pred = self.predict(X)
        y_pred = torch.from_numpy(y_pred)

        dataset = self.get_dataset(X, y)
        y_true = dataset.y
        event_multiclass = y_true["event"].clone()

        loss = 0
        with torch.no_grad():
            for event_of_interest in range(1, self.n_events + 1):
                y_true["event"] = (event_multiclass == event_of_interest).long()
                y_pred_event = y_pred[:, :, event_of_interest - 1]
                loss += self.get_loss(y_pred_event, y_true, X=X, training=False)

            return -loss


class _SurvTRACEModule(nn.Module):
    """
    Parameters
    ----------
    numerical_features : array-like of {bool, int, str} of shape (n_features) \
            or shape (n_numerical_features,), default=None
        Indicates the numerical features.

        - None : no feature will be considered numerical.
        - boolean array-like : boolean mask indicating numerical features.
        - integer array-like : integer indices indicating numerical
          features.
        - str array-like: names of numerical features (assuming the training
          data has feature names).

    Beware:

    * n_numerical_features
    * vocab_size
    * n_events
    * n_features_in
    * n_features_out

    are set during fit() using the SurvTRACE class, derived from skorch.NeuralNet.

    TODO
    """

    def __init__(
        self,
        init_range=0.02,
        # BertEmbedding
        n_numerical_features=1,  # *
        vocab_size=8,  # *
        hidden_size=16,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.0,
        initializer_range=0.02,
        # BertEncoder
        num_hidden_layers=3,
        # BertCLS
        intermediate_size=64,
        n_events=1,  # *
        n_features_in=1,  # *
        n_features_out=1,  # *
    ):
        super().__init__()
        self.init_range = init_range
        self.embeddings = BertEmbeddings(
            n_numerical_features=n_numerical_features,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob,
            initializer_range=initializer_range,
        )
        self.encoder = BertEncoder(num_hidden_layers=num_hidden_layers)
        self.cls = BertCLSMulti(
            n_events=n_events,
            n_features_in=n_features_in,
            n_features_out=n_features_out,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
        )

        self.apply(self._init_weights)  # TODO try to remove it

    def forward(self, X_numerical, X_categorical, event_of_interest):
        X_trans = self.embeddings(X_numerical, X_categorical)
        X_trans, _, _ = self.encoder(X_trans)
        y_pred = self.cls(X_trans, event_of_interest=event_of_interest)
        return y_pred

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses \
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

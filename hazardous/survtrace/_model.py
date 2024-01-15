"""
This code is a refactoring derived from RyanWangZf/SurvTRACE
also published under the MIT license with the following copyright:

    Copyright (c) 2021 Maximum
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNet
from skorch.callbacks import Callback, ProgressBar
from torch.optim import Adam

from ._bert_layers import (
    DEFAULT_QUANTILE_HORIZONS,
    BertCLSMulti,
    BertEmbeddings,
    BertEncoder,
)
from ._encoder import SurvFeatureEncoder, SurvTargetEncoder
from ._losses import NLLPCHazardLoss
from ._utils import pad_col


# Skorch hack 1: this will properly set parameters at runtime using callbacks.
# See: https://stackoverflow.com/a/60170023
class ShapeSetter(Callback):
    def on_train_begin(self, net, X=None, y=None):
        net.check_data(X, y)

        enc = SurvFeatureEncoder(
            categorical_features=net.categorical_features,
            numerical_features=net.numerical_features,
        )
        X_t = enc.fit_transform(X)
        n_numerical_features = len(enc.numerical_features)
        vocab_size = enc.vocab_size_

        n_events = y["event"].nunique() - 1
        n_features_in = X_t.shape[1]

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
        categorical_features=None,
        numerical_features=None,
        quantile_horizons=None,
        batch_size=1024,
        lr=1e-4,
        weight_decay=0,
        device="cpu",
        max_epochs=100,
    ):
        module = _SurvTRACEModule()
        # Skorch hack 2: this allows to use ShapeSetter on nested modules
        self._modules = ["module", "embeddings", "cls"]
        self.embeddings_ = module.embeddings
        self.cls_ = module.cls

        criterion = NLLPCHazardLoss
        callbacks = [ShapeSetter(), ProgressBar(detect_notebook=False)]
        optimizer = Adam
        super().__init__(
            module=module,
            criterion=criterion,
            optimizer=optimizer,
            optimizer__lr=lr,
            optimizer__weight_decay=weight_decay,
            callbacks=callbacks,
            batch_size=batch_size,
            device=device,
            max_epochs=max_epochs,
        )
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.quantile_horizons = quantile_horizons

    def check_data(self, X, y=None):
        if not hasattr(X, "__dataframe__"):
            raise TypeError(f"X must be a dataframe. Got {type(X)}")

        if y is None or not hasattr(y, "__dataframe__"):
            raise TypeError(f"y must be a dataframe. Got {type(y)}")

        return super().check_data(X, y)

    def get_dataset(self, X, y=None):
        if self.quantile_horizons is None:
            quantile_horizons = DEFAULT_QUANTILE_HORIZONS
        else:
            quantile_horizons = self.quantile_horizons

        if hasattr(self, "target_encoder_"):
            y = self.target_encoder_.transform(y)
        else:
            self.target_encoder_ = SurvTargetEncoder(
                quantile_horizons=quantile_horizons
            )
            y = self.target_encoder_.fit_transform(y)

        if hasattr(self, "feature_encoder_"):
            X = self.feature_encoder_.transform(X)
        else:
            self.feature_encoder_ = SurvFeatureEncoder(
                categorical_features=self.categorical_features,
                numerical_features=self.numerical_features,
            )
            X = self.feature_encoder_.fit_transform(X)

        categorical_features = self.feature_encoder_.categorical_features_
        numerical_features = self.feature_encoder_.numerical_features_

        X_numerical = torch.as_tensor(
            X[numerical_features].to_numpy(),
            dtype=torch.float,
        )
        X_categorical = torch.as_tensor(
            X[categorical_features].to_numpy(),
            dtype=torch.long,
        )
        X = {
            "X_numerical": X_numerical,
            "X_categorical": X_categorical,
        }
        y = y.to_dict(orient="list")  # A dict of list
        y["duration"] = torch.as_tensor(y["duration"], dtype=torch.long)
        y["event"] = torch.as_tensor(y["event"], dtype=torch.long)
        y["frac_duration"] = torch.as_tensor(y["frac_duration"], dtype=torch.float)

        return super().get_dataset(X, y)

    def predict_hazard(self, X):
        y_pred = self.predict(X)
        hazard = F.softplus(y_pred)
        hazard = pad_col(hazard, where="start")
        return hazard

    def predict_survival_function(self, X):
        hazard = self.predict_hazard(X)
        surv = torch.exp(-hazard.cumsum(axis=1))
        return surv

    def predict_cumulative_incidence(self, X):
        return 1 - self.predict_survival_function(X)

    def initialize_module(self):
        """Skorch hack 2: set changed modules."""
        for module_name in self._modules:
            if module_name in ["cls", "embeddings"]:
                kwargs = self.get_params_for(module_name)
                sub_module = getattr(self, module_name + "_", None)
                sub_module = self.initialized_instance(sub_module, kwargs)
                # pylint: disable=attribute-defined-outside-init
                setattr(self.module, module_name, sub_module)
        return super().initialize_module()


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
    """

    def __init__(
        self,
        init_range=0.02,
        # BertEmbedding
        n_numerical_features=1,
        vocab_size=8,
        hidden_size=16,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.0,
        # BertEncoder
        num_hidden_layers=3,
        # BertCLS
        intermediate_size=64,
        n_events=1,
        n_features_in=1,
        n_features_out=1,
    ):
        super().__init__()
        self.init_range = init_range
        self.embeddings = BertEmbeddings(
            n_numerical_features=n_numerical_features,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob,
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

    def forward(self, X_numerical, X_categorical):
        X_trans = self.embeddings(X_numerical, X_categorical)
        X_trans, _, _ = self.encoder(X_trans)
        y_pred = self.cls(X_trans)
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

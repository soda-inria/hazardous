import numpy as np


class SurvivalMixin:

    def predict_cumulative_hazard_function(self, X_test, times):
        survival_probs = self.predict_survival_function(X_test, times)
        cumulative_hazards = -np.log(survival_probs + 1e-8)
        return cumulative_hazards

    def predict_risk_estimate(self, X_test, times):
        cumulative_hazards = self.predict_cumulative_hazard_function(X_test, times)
        return cumulative_hazards.sum(axis=1)

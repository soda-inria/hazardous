import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score

from hazardous.data import make_synthetic_competing_weibull


@pytest.mark.parametrize("seed", range(3))
def test_competing_weibull_no_censoring(seed):
    n_samples = 1000
    X, y = make_synthetic_competing_weibull(
        n_events=3,
        n_samples=n_samples,
        return_X_y=True,
        censoring_relative_scale=None,  # no censoring
        random_state=seed,
    )
    assert X.shape == (n_samples, 3 * 2)  # 3 events, 2 parameter features each
    assert X.columns.tolist() == [
        "shape_1",
        "scale_1",
        "shape_2",
        "scale_2",
        "shape_3",
        "scale_3",
    ]
    assert y.shape == (n_samples, 2)
    assert y.columns.tolist() == ["event", "duration"]
    assert sorted(y["event"].unique().tolist()) == [1, 2, 3]
    assert y["duration"].min() >= 0
    assert y["duration"].max() <= 20_000

    # Check that competing events are approximately balanced with the default
    # parameter ranges.
    event_counts = y["event"].value_counts().sort_index()
    assert event_counts.max() < 2 * event_counts.min(), event_counts

    # Check that the features make it possible to separate the events better
    # than a marginal baseline, but that a significant amount of
    # unpredictability remains.
    event_classification_acc = cross_val_score(
        RandomForestClassifier(random_state=seed),
        X,
        y["event"],
        cv=3,
        n_jobs=4,
    ).mean()
    baseline_classification_acc = cross_val_score(
        DummyClassifier(strategy="most_frequent"),
        X,
        y["event"],
        cv=3,
    ).mean()
    assert 0.4 > baseline_classification_acc > 0.3  # approximately balanced
    assert event_classification_acc > 1.2 * baseline_classification_acc
    assert event_classification_acc < 0.6  # still challenging.

    # Check that the features make it possible to predict the durations better
    # than a marginal baseline, but that a significant amount of unpredictability
    # remains.
    median_duration = y["duration"].median()
    duration_regression_relative_error = (
        -cross_val_score(
            RandomForestRegressor(criterion="poisson", random_state=seed),
            X,
            y["duration"],
            cv=3,
            n_jobs=4,
            scoring="neg_mean_absolute_error",
        ).mean()
        / median_duration
    )
    baseline_duration_relative_error = (
        -cross_val_score(
            DummyRegressor(strategy="mean"),
            X,
            y["duration"],
            cv=3,
            scoring="neg_mean_absolute_error",
        ).mean()
        / median_duration
    )
    assert 0.9 < baseline_duration_relative_error < 1.1  # approximately balanced
    assert duration_regression_relative_error < 0.98 * baseline_duration_relative_error
    assert duration_regression_relative_error > 0.8  # still challenging.


@pytest.mark.parametrize("seed", range(3))
def test_competing_weibull_with_censoring(seed):
    n_samples = 1000
    X_low_scale, y_low_scale = make_synthetic_competing_weibull(
        n_events=3,
        n_samples=n_samples,
        return_X_y=True,
        censoring_relative_scale=0.8,
        random_state=seed,
    )
    X_high_scale, y_high_scale = make_synthetic_competing_weibull(
        n_events=3,
        n_samples=n_samples,
        return_X_y=True,
        censoring_relative_scale=1.5,
        random_state=seed,
    )
    # Input features are independent of censoring:
    assert X_low_scale.equals(X_high_scale)

    # Censoring rate is lower for higher scale:
    high_scale_censoring_rate = (y_high_scale["event"] == 0).mean()
    low_scale_censoring_rate = (y_low_scale["event"] == 0).mean()
    assert 0.35 < high_scale_censoring_rate < 0.5
    assert 0.5 < low_scale_censoring_rate < 0.65

    # Uncensored events and durations match:
    commonly_uncensored = (y_low_scale["event"] != 0) & (y_high_scale["event"] != 0)
    y_low_scale_uncensored = y_low_scale[commonly_uncensored]
    y_high_scale_uncensored = y_high_scale[commonly_uncensored]
    assert y_low_scale_uncensored.equals(y_high_scale_uncensored)

    # Check that high scale censoring keeps approximate balance between events:
    event_counts = y_high_scale.query("event != 0")["event"].value_counts().sort_index()
    assert event_counts.max() < 2 * event_counts.min(), event_counts

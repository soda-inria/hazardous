import nox


def _common_test_steps(session):
    session.run("python", "-c", "import sklearn; sklearn.show_versions()")
    session.run("pytest", "-v", "--cov")


# TODO: add 3.12 as soon as numpy and scikit-learn upload 3.12 wheels
# built against the pre-releases.
# https://dev.to/hugovk/help-test-python-312-beta-1508
@nox.session(python=["3.9", "3.10", "3.11"])
def test_latest_from_pypi(session):
    # Test the newest versions of the dependencies.
    session.install(".[test]")
    _common_test_steps(session)


@nox.session(python=["3.11"], venv_backend="mamba")
def test_latest_from_conda_forge(session):
    # Test the newest versions of the dependencies from conda-forge.
    # XXX: hown to do the same with a single mamba install command?
    for package_name in [
        "pytest",
        "pytest-cov",
        "numpy",
        "pandas",
        "scikit-learn",
        "lifelines",
    ]:
        session.conda_install(package_name, channel="conda-forge")
    _common_test_steps(session)


@nox.session(python=["3.9"])
def test_oldest_from_pypi(session):
    # Test the oldest supported version of the dependencies.
    session.install(".[test,oldest_deps]")
    _common_test_steps(session)

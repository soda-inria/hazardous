import nox


def _common_test_steps(session):
    session.run("python", "-c", "import sklearn; sklearn.show_versions()")
    if session.posargs:
        session.run("pytest", *session.posargs)
    else:
        session.run("pytest", "-v", "--cov", "--pyargs", "hazardous")


@nox.session(python=["3.9", "3.10", "3.11", "3.12"])
def test_latest_from_pypi(session):
    # Test the newest versions of the dependencies.
    session.install(".[test]")
    _common_test_steps(session)


@nox.session(python=["3.12"], venv_backend="mamba")
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
        "tqdm",
        "flit",
    ]:
        session.conda_install(package_name, channel="conda-forge")
    session.install("--no-build-isolation", ".")
    _common_test_steps(session)


@nox.session(python=["3.9"])
def test_oldest_from_pypi(session):
    # Test the oldest supported version of the dependencies.
    session.install(".[test,oldest_deps]")
    _common_test_steps(session)


@nox.session
def doc(session):
    session.install(".[doc]")
    session.run(
        # fmt: off
        "python", "-m", "sphinx",
        "-T", "-E",
        "-W", "--keep-going",
        "-b", "html",
        "doc",
        "doc/_build/html",
        # fmt: on
    )

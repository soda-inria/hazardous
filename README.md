# hazardous

Predictive Competing Risks and Survival Analysis.

[![Tests](https://github.com/soda-inria/hazardous/actions/workflows/test.yml/badge.svg)](https://github.com/soda-inria/hazardous/actions/workflows/test.yml)

## Installation and usage

XXX: the following will work only once the 0.1.0 release is out!

```
pip install hazardous
```

or

```
conda install -c conda-forge hazardous
```

Then browse the [online documentation](https://soda-inria.github.io/hazardous/)
and run the `examples/` to get started.

## Development and testing

Install in "editable" mode in your current Python env:

```
pip install -e ".[dev]" --no-build-isolation -v
pre-commit install
```

Run the tests with nox to test in an environment that matches exactly on
specific CI build, for instance:

```
nox -p 3.11 -s test_latest_from_pypi -r
```

The `-r` flag makes it possible to reuse an existing env.

You can also install the test dependencies in the current env and use `pytest`
directly with arbitrary command line arguments:

```
pip install -e ".[test]" --no-build-isolation -v
pytest -vl -x -k test_name_pattern
```

## Building the doc

Using `nox`:

```
nox -s doc -r
```

or manually:

```
pip install -e ".[doc]" --no-build-isolation -v
cd doc
make html
```

The resulting html files are generated under the `doc/_build` folder.

## Building a release

```
pip install build
python -m build
ls dist/
```

TODO: make it possible to automate a release using GitHub Actions for a given tag.

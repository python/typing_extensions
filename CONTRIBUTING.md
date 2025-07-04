Code in this repository should follow CPython's style guidelines and
contributors need to sign the PSF Contributor Agreement.

# typing\_extensions

The `typing_extensions` module provides a way to access new features from the standard
library `typing` module in older versions of Python. For example, Python 3.10 adds
`typing.TypeGuard`, but users of older versions of Python can use `typing_extensions` to
use `TypeGuard` in their code even if they are unable to upgrade to Python 3.10.

If you contribute the runtime implementation of a new `typing` feature to CPython, you
are encouraged to also implement the feature in `typing_extensions`. Because the runtime
implementation of much of the infrastructure in the `typing` module has changed over
time, this may require different code for some older Python versions.

`typing_extensions` may also include experimental features that are not yet part of the
standard library, so that users can experiment with them before they are added to the
standard library. Such features should already be specified in a PEP or merged into
CPython's `main` branch.

# Versioning scheme

Starting with version 4.0.0, `typing_extensions` uses
[Semantic Versioning](https://semver.org/). See the documentation
for more detail.

## Development version
After a release the version is increased once in [pyproject.toml](/pyproject.toml) and
appended with a `.dev` suffix, e.g. `4.0.1.dev`.
Further subsequent updates are not planned between releases.

# Type stubs

A stub file for `typing_extensions` is maintained
[in typeshed](https://github.com/python/typeshed/blob/main/stdlib/typing_extensions.pyi).
Because of the special status that `typing_extensions` holds in the typing ecosystem,
the stubs are placed in the standard library in typeshed and distributed as
part of the stubs bundled with individual type checkers.

# Running tests

Testing `typing_extensions` can be tricky because many development tools depend on
`typing_extensions`, so you may end up testing some installed version of the library,
rather than your local code.

The simplest way to run the tests locally is:

- `cd src/`
- `python test_typing_extensions.py`

Alternatively, you can invoke `unittest` explicitly:

- `python -m unittest test_typing_extensions.py`

Running these commands in the `src/` directory ensures that the local file
`typing_extensions.py` is used, instead of any other version of the library you
may have installed.

# Workflow for PyPI releases

- Make sure you follow the versioning policy in the documentation
  (e.g., release candidates before any feature release, do not release development versions)

- Ensure that GitHub Actions reports no errors.

- Update the version number in `typing_extensions/pyproject.toml` and in
  `typing_extensions/CHANGELOG.md`.

- Create a new GitHub release at https://github.com/python/typing_extensions/releases/new.
  Details:
  - The tag should be just the version number, e.g. `4.1.1`.
  - Copy the release notes from `CHANGELOG.md`.

- Release automation will finish the release. You'll have to manually
  approve the last step before upload.

- After the release has been published on PyPI upgrade the version in number in [pyproject.toml](/pyproject.toml) to a `dev` version of the next planned release. For example, change 4.1.1 to 4.X.X.dev, see also [Development versions](#development-version). # TODO decide on major vs. minor increase.

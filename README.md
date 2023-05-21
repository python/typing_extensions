# Typing Extensions

[![Chat at https://gitter.im/python/typing](https://badges.gitter.im/python/typing.svg)](https://gitter.im/python/typing)

[Documentation](https://typing-extensions.readthedocs.io/en/latest/#) –
[PyPI](https://pypi.org/project/typing-extensions/)

## Overview

The `typing_extensions` module serves two related purposes:

- Enable use of new type system features on older Python versions. For example,
  `typing.TypeGuard` is new in Python 3.10, but `typing_extensions` allows
  users on previous Python versions to use it too.
- Enable experimentation with new type system PEPs before they are accepted and
  added to the `typing` module.

New features may be added to `typing_extensions` as soon as they are specified
in a PEP that has been added to the [python/peps](https://github.com/python/peps)
repository. If the PEP is accepted, the feature will then be added to `typing`
for the next CPython release. No typing PEP has been rejected so far, so we
haven't yet figured out how to deal with that possibility.

Starting with version 4.0.0, `typing_extensions` uses
[Semantic Versioning](https://semver.org/). The
major version is incremented for all backwards-incompatible changes.
Therefore, it's safe to depend
on `typing_extensions` like this: `typing_extensions >=x.y, <(x+1)`,
where `x.y` is the first version that includes all features you need.

`typing_extensions` supports Python versions 3.7 and higher. In the future,
support for older Python versions will be dropped some time after that version
reaches end of life.

## Included items

See [the documentation](https://typing-extensions.readthedocs.io/en/latest/#) for a
complete listing of module contents.

## Running tests

To run tests, navigate into the `src/` directory and run
`test_typing_extensions.py`.

# Unreleased

- Fix a regression introduced in v4.6.0 in the implementation of
  runtime-checkable protocols. The regression meant
  that doing `class Foo(X, typing_extensions.Protocol)`, where `X` was a class that
  had `abc.ABCMeta` as its metaclass, would then cause subsequent
  `isinstance(1, X)` calls to erroneously raise `TypeError`. Patch by
  Alex Waygood (backporting the CPython PR
  https://github.com/python/cpython/pull/105152).
- Sync the repository's LICENSE file with that of CPython.
  `typing_extensions` is distributed under the same license as
  CPython itself.
- Skip a problematic test on Python 3.12.0b1. The test fails on 3.12.0b1 due to
  a bug in CPython, which will be fixed in 3.12.0b2. The
  `typing_extensions` test suite now passes on 3.12.0b1.
- Declare support for Python 3.12. Patch by Jelle Zijlstra.

# Release 4.6.2 (May 25, 2023)

- Fix use of `@deprecated` on classes with `__new__` but no `__init__`.
  Patch by Jelle Zijlstra.
- Fix regression in version 4.6.1 where comparing a generic class against a
  runtime-checkable protocol using `isinstance()` would cause `AttributeError`
  to be raised if using Python 3.7.

# Release 4.6.1 (May 23, 2023)

- Change deprecated `@runtime` to formal API `@runtime_checkable` in the error
  message. Patch by Xuehai Pan.
- Fix regression in 4.6.0 where attempting to define a `Protocol` that was
  generic over a `ParamSpec` or a `TypeVarTuple` would cause `TypeError` to be
  raised. Patch by Alex Waygood.

# Release 4.6.0 (May 22, 2023)

- `typing_extensions` is now documented at
  https://typing-extensions.readthedocs.io/en/latest/. Patch by Jelle Zijlstra.
- Add `typing_extensions.Buffer`, a marker class for buffer types, as proposed
  by PEP 688. Equivalent to `collections.abc.Buffer` in Python 3.12. Patch by
  Jelle Zijlstra.
- Backport two CPython PRs fixing various issues with `typing.Literal`:
  https://github.com/python/cpython/pull/23294 and
  https://github.com/python/cpython/pull/23383. Both CPython PRs were
  originally by Yurii Karabas, and both were backported to Python >=3.9.1, but
  no earlier. Patch by Alex Waygood.

  A side effect of one of the changes is that equality comparisons of `Literal`
  objects will now raise a `TypeError` if one of the `Literal` objects being
  compared has a mutable parameter. (Using mutable parameters with `Literal` is
  not supported by PEP 586 or by any major static type checkers.)
- `Literal` is now reimplemented on all Python versions <= 3.10.0. The
  `typing_extensions` version does not suffer from the bug that was fixed in
  https://github.com/python/cpython/pull/29334. (The CPython bugfix was
  backported to CPython 3.10.1 and 3.9.8, but no earlier.)
- Backport [CPython PR 26067](https://github.com/python/cpython/pull/26067)
  (originally by Yurii Karabas), ensuring that `isinstance()` calls on
  protocols raise `TypeError` when the protocol is not decorated with
  `@runtime_checkable`. Patch by Alex Waygood.
- Backport several significant performance improvements to runtime-checkable
  protocols that have been made in Python 3.12 (see
  https://github.com/python/cpython/issues/74690 for details). Patch by Alex
  Waygood.

  A side effect of one of the performance improvements is that the members of
  a runtime-checkable protocol are now considered “frozen” at runtime as soon
  as the class has been created. Monkey-patching attributes onto a
  runtime-checkable protocol will still work, but will have no impact on
  `isinstance()` checks comparing objects to the protocol. See
  ["What's New in Python 3.12"](https://docs.python.org/3.12/whatsnew/3.12.html#typing)
  for more details.
- `isinstance()` checks against runtime-checkable protocols now use
  `inspect.getattr_static()` rather than `hasattr()` to lookup whether
  attributes exist (backporting https://github.com/python/cpython/pull/103034).
  This means that descriptors and `__getattr__` methods are no longer
  unexpectedly evaluated during `isinstance()` checks against runtime-checkable
  protocols. However, it may also mean that some objects which used to be
  considered instances of a runtime-checkable protocol on older versions of
  `typing_extensions` may no longer be considered instances of that protocol
  using the new release, and vice versa. Most users are unlikely to be affected
  by this change. Patch by Alex Waygood.
- Backport the ability to define `__init__` methods on Protocol classes, a
  change made in Python 3.11 (originally implemented in
  https://github.com/python/cpython/pull/31628 by Adrian Garcia Badaracco).
  Patch by Alex Waygood.
- Speedup `isinstance(3, typing_extensions.SupportsIndex)` by >10x on Python
  <3.12. Patch by Alex Waygood.
- Add `typing_extensions` versions of `SupportsInt`, `SupportsFloat`,
  `SupportsComplex`, `SupportsBytes`, `SupportsAbs` and `SupportsRound`. These
  have the same semantics as the versions from the `typing` module, but
  `isinstance()` checks against the `typing_extensions` versions are >10x faster
  at runtime on Python <3.12. Patch by Alex Waygood.
- Add `__orig_bases__` to non-generic TypedDicts, call-based TypedDicts, and
  call-based NamedTuples. Other TypedDicts and NamedTuples already had the attribute.
  Patch by Adrian Garcia Badaracco.
- Add `typing_extensions.get_original_bases`, a backport of
  [`types.get_original_bases`](https://docs.python.org/3.12/library/types.html#types.get_original_bases),
  introduced in Python 3.12 (CPython PR
  https://github.com/python/cpython/pull/101827, originally by James
  Hilton-Balfe). Patch by Alex Waygood.

  This function should always produce correct results when called on classes
  constructed using features from `typing_extensions`. However, it may
  produce incorrect results when called on some `NamedTuple` or `TypedDict`
  classes that use `typing.{NamedTuple,TypedDict}` on Python <=3.11.
- Constructing a call-based `TypedDict` using keyword arguments for the fields
  now causes a `DeprecationWarning` to be emitted. This matches the behaviour
  of `typing.TypedDict` on 3.11 and 3.12.
- Backport the implementation of `NewType` from 3.10 (where it is implemented
  as a class rather than a function). This allows user-defined `NewType`s to be
  pickled. Patch by Alex Waygood.
- Fix tests and import on Python 3.12, where `typing.TypeVar` can no longer be
  subclassed. Patch by Jelle Zijlstra.
- Add `typing_extensions.TypeAliasType`, a backport of `typing.TypeAliasType`
  from PEP 695. Patch by Jelle Zijlstra.
- Backport changes to the repr of `typing.Unpack` that were made in order to
  implement [PEP 692](https://peps.python.org/pep-0692/) (backport of
  https://github.com/python/cpython/pull/104048). Patch by Alex Waygood.

# Release 4.5.0 (February 14, 2023)

- Runtime support for PEP 702, adding `typing_extensions.deprecated`. Patch
  by Jelle Zijlstra.
- Add better default value for TypeVar `default` parameter, PEP 696. Enables
  runtime check if `None` was passed as default. Patch by Marc Mueller (@cdce8p).
- The `@typing_extensions.override` decorator now sets the `.__override__`
  attribute. Patch by Steven Troxler.
- Fix `get_type_hints()` on cross-module inherited `TypedDict` in 3.9 and 3.10.
  Patch by Carl Meyer.
- Add `frozen_default` parameter on `dataclass_transform`. Patch by Erik De Bonte.

# Release 4.4.0 (October 6, 2022)

- Add `typing_extensions.Any` a backport of python 3.11's Any class which is
  subclassable at runtime. (backport from python/cpython#31841, by Shantanu
  and Jelle Zijlstra). Patch by James Hilton-Balfe (@Gobot1234).
- Add initial support for TypeVarLike `default` parameter, PEP 696.
  Patch by Marc Mueller (@cdce8p).
- Runtime support for PEP 698, adding `typing_extensions.override`. Patch by
  Jelle Zijlstra.
- Add the `infer_variance` parameter to `TypeVar`, as specified in PEP 695.
  Patch by Jelle Zijlstra.

# Release 4.3.0 (July 1, 2022)

- Add `typing_extensions.NamedTuple`, allowing for generic `NamedTuple`s on
  Python <3.11 (backport from python/cpython#92027, by Serhiy Storchaka). Patch
  by Alex Waygood (@AlexWaygood).
- Adjust `typing_extensions.TypedDict` to allow for generic `TypedDict`s on
  Python <3.11 (backport from python/cpython#27663, by Samodya Abey). Patch by
  Alex Waygood (@AlexWaygood).

# Release 4.2.0 (April 17, 2022)

- Re-export `typing.Unpack` and `typing.TypeVarTuple` on Python 3.11.
- Add `ParamSpecArgs` and `ParamSpecKwargs` to `__all__`.
- Improve "accepts only single type" error messages.
- Improve the distributed package. Patch by Marc Mueller (@cdce8p).
- Update `typing_extensions.dataclass_transform` to rename the
  `field_descriptors` parameter to `field_specifiers` and accept
  arbitrary keyword arguments.
- Add `typing_extensions.get_overloads` and
  `typing_extensions.clear_overloads`, and add registry support to
  `typing_extensions.overload`. Backport from python/cpython#89263.
- Add `typing_extensions.assert_type`. Backport from bpo-46480.
- Drop support for Python 3.6. Original patch by Adam Turner (@AA-Turner).

# Release 4.1.1 (February 13, 2022)

- Fix importing `typing_extensions` on Python 3.7.0 and 3.7.1. Original
  patch by Nikita Sobolev (@sobolevn).

# Release 4.1.0 (February 12, 2022)

- Runtime support for PEP 646, adding `typing_extensions.TypeVarTuple`
  and `typing_extensions.Unpack`.
- Add interaction of `Required` and `NotRequired` with `__required_keys__`,
  `__optional_keys__` and `get_type_hints()`. Patch by David Cabot (@d-k-bo).
- Runtime support for PEP 675 and `typing_extensions.LiteralString`.
- Add `Never` and `assert_never`. Backport from bpo-46475.
- `ParamSpec` args and kwargs are now equal to themselves. Backport from
  bpo-46676. Patch by Gregory Beauregard (@GBeauregard).
- Add `reveal_type`. Backport from bpo-46414.
- Runtime support for PEP 681 and `typing_extensions.dataclass_transform`.
- `Annotated` can now wrap `ClassVar` and `Final`. Backport from
  bpo-46491. Patch by Gregory Beauregard (@GBeauregard).
- Add missed `Required` and `NotRequired` to `__all__`. Patch by
  Yuri Karabas (@uriyyo).
- The `@final` decorator now sets the `__final__` attribute on the
  decorated object to allow runtime introspection. Backport from
  bpo-46342.
- Add `is_typeddict`. Patch by Chris Moradi (@chrismoradi) and James
  Hilton-Balfe (@Gobot1234).

# Release 4.0.1 (November 30, 2021)

- Fix broken sdist in release 4.0.0. Patch by Adam Turner (@AA-Turner).
- Fix equality comparison for `Required` and `NotRequired`. Patch by
  Jelle Zijlstra (@jellezijlstra).
- Fix usage of `Self` as a type argument. Patch by Chris Wesseling
  (@CharString) and James Hilton-Balfe (@Gobot1234).

# Release 4.0.0 (November 14, 2021)

- Starting with version 4.0.0, typing_extensions uses Semantic Versioning.
  See the README for more information.
- Dropped support for Python versions 3.5 and older, including Python 2.7.
- Simplified backports for Python 3.6.0 and newer. Patch by Adam Turner (@AA-Turner).

## Added in version 4.0.0

- Runtime support for PEP 673 and `typing_extensions.Self`. Patch by
  James Hilton-Balfe (@Gobot1234).
- Runtime support for PEP 655 and `typing_extensions.Required` and `NotRequired`.
  Patch by David Foster (@davidfstr).

## Removed in version 4.0.0

The following non-exported but non-private names have been removed as they are
unneeded for supporting Python 3.6 and newer.

- TypingMeta
- OLD_GENERICS
- SUBS_TREE
- HAVE_ANNOTATED
- HAVE_PROTOCOLS
- V_co
- VT_co

# Previous releases

Prior to release 4.0.0 we did not provide a changelog. Please check
the Git history for details.

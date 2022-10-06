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

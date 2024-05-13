.. module:: typing_extensions

Welcome to typing_extensions's documentation!
=============================================

``typing_extensions`` complements the standard-library :py:mod:`typing` module,
providing runtime support for type hints as specified by :pep:`484` and subsequent
PEPs. The module serves two related purposes:

- Enable use of new type system features on older Python versions. For example,
  :py:data:`typing.TypeGuard` is new in Python 3.10, but ``typing_extensions`` allows
  users on previous Python versions to use it too.
- Enable experimentation with type system features proposed in new PEPs before they are accepted and
  added to the :py:mod:`typing` module.

New features may be added to ``typing_extensions`` as soon as they are specified
in a PEP that has been added to the `python/peps <https://github.com/python/peps>`_
repository. If the PEP is accepted, the feature will then be added to the
:py:mod:`typing` module for the next CPython release. No typing PEP that
affected ``typing_extensions`` has been rejected so far, so we haven't yet
figured out how to deal with that possibility.

Bugfixes and new typing features that don't require a PEP may be added to
``typing_extensions`` once they are merged into CPython's main branch.

``typing_extensions`` also re-exports all names from the :py:mod:`typing` module,
including those that have always been present in the module. This allows users to
import names from ``typing_extensions`` without having to remember exactly when
each object was added to :py:mod:`typing`. There are a few exceptions:
:py:class:`typing.ByteString`, which is deprecated and due to be removed in Python
3.14, is not re-exported. Similarly, the ``typing.io`` and ``typing.re`` submodules,
which are removed in Python 3.13, are excluded.

Versioning and backwards compatibility
--------------------------------------

Starting with version 4.0.0, ``typing_extensions`` uses
`Semantic Versioning <https://semver.org>`_. A changelog is
maintained `on GitHub <https://github.com/python/typing_extensions/blob/main/CHANGELOG.md>`_.

The major version is incremented for all backwards-incompatible changes.
Therefore, it's safe to depend
on ``typing_extensions`` like this: ``typing_extensions >=x.y, <(x+1)``,
where ``x.y`` is the first version that includes all features you need.
In view of the wide usage of ``typing_extensions`` across the ecosystem,
we are highly hesitant to break backwards compatibility, and we do not
expect to increase the major version number in the foreseeable future.

Feature releases, with version numbers of the form 4.N.0, are made at
irregular intervals when enough new features accumulate. Before a
feature release, at least one release candidate (with a version number
of the form 4.N.0rc1) should be released to give downstream users time
to test. After at least a week of testing, the new feature version
may then be released. If necessary, additional release candidates can
be added.

Bugfix releases, with version numbers of the form 4.N.1 or higher,
may be made if bugs are discovered after a feature release.

We provide no backward compatibility guarantees for prereleases (e.g.,
release candidates) and for unreleased code in our Git repository.

Before version 4.0.0, the versioning scheme loosely followed the Python
version from which features were backported; for example,
``typing_extensions`` 3.10.0.0 was meant to reflect ``typing`` as of
Python 3.10.0. During this period, no changelog was maintained.

Runtime use of types
~~~~~~~~~~~~~~~~~~~~

We aim for complete backwards compatibility in terms of the names we export:
code like ``from typing_extensions import X`` that works on one
typing-extensions release will continue to work on the next.
It is more difficult to maintain compatibility for users that introspect
types at runtime, as almost any detail can potentially break compatibility.
Users who introspect types should follow these guidelines to minimize
the risk of compatibility issues:

- Always check for both the :mod:`typing` and ``typing_extensions`` versions
  of objects, even if they are currently the same on some Python version.
  Future ``typing_extensions`` releases may re-export a separate version of
  the object to backport some new feature or bugfix.
- Use public APIs like :func:`get_origin` and :func:`get_original_bases` to
  access internal information about types, instead of accessing private
  attributes directly. If some information is not available through a public
  attribute, consider opening an issue in CPython to add such an API.

Here is an example recipe for a general-purpose function that could be used for
reasonably performant runtime introspection of typing objects. The function
will be resilient against any potential changes in ``typing_extensions`` that
alter whether an object is reimplemented in ``typing_extensions``, rather than
simply being re-exported from the :mod:`typing` module::

   import functools
   import typing
   import typing_extensions
   from typing import Tuple, Any

   # Use an unbounded cache for this function, for optimal performance
   @functools.lru_cache(maxsize=None)
   def get_typing_objects_by_name_of(name: str) -> Tuple[Any, ...]:
       result = tuple(
           getattr(module, name)
           # You could potentially also include mypy_extensions here,
           # if your library supports mypy_extensions
           for module in (typing, typing_extensions)
           if hasattr(module, name)
       )
       if not result:
           raise ValueError(
               f"Neither typing nor typing_extensions has an object called {name!r}"
           )
       return result


   # Use a cache here as well, but make it a bounded cache
   # (the default cache size is 128)
   @functools.lru_cache()
   def is_typing_name(obj: object, name: str) -> bool:
       return any(obj is thing for thing in get_typing_objects_by_name_of(name))

Example usage::

   >>> import typing, typing_extensions
   >>> from functools import partial
   >>> from typing_extensions import get_origin
   >>> is_literal = partial(is_typing_name, name="Literal")
   >>> is_literal(typing.Literal)
   True
   >>> is_literal(typing_extensions.Literal)
   True
   >>> is_literal(typing.Any)
   False
   >>> is_literal(get_origin(typing.Literal[42]))
   True
   >>> is_literal(get_origin(typing_extensions.Final[42]))
   False

Python version support
----------------------

``typing_extensions`` currently supports Python versions 3.8 and higher. In the future,
support for older Python versions will be dropped some time after that version
reaches end of life.

Module contents
---------------

As most of the features in ``typing_extensions`` exist in :py:mod:`typing`
in newer versions of Python, the documentation here is brief and focuses
on aspects that are specific to ``typing_extensions``, such as limitations
on specific Python versions.

Special typing primitives
~~~~~~~~~~~~~~~~~~~~~~~~~

.. data:: Annotated

   See :py:data:`typing.Annotated` and :pep:`593`. In ``typing`` since 3.9.

   .. versionchanged:: 4.1.0

      ``Annotated`` can now wrap :data:`ClassVar` and :data:`Final`.

.. data:: Any

   See :py:data:`typing.Any`.

   Since Python 3.11, ``typing.Any`` can be used as a base class.
   ``typing_extensions.Any`` supports this feature on older versions.

   .. versionadded:: 4.4.0

      Added to support inheritance from ``Any``.

.. data:: Concatenate

   See :py:data:`typing.Concatenate` and :pep:`612`. In ``typing`` since 3.10.

   The backport does not support certain operations involving ``...`` as
   a parameter; see :issue:`48` and :issue:`110` for details.

.. data:: Final

   See :py:data:`typing.Final` and :pep:`591`. In ``typing`` since 3.8.

.. data:: Literal

   See :py:data:`typing.Literal` and :pep:`586`. In ``typing`` since 3.8.

   :py:data:`typing.Literal` does not flatten or deduplicate parameters on Python <3.9.1, and a
   caching bug was fixed in 3.10.1/3.9.8. The ``typing_extensions`` version
   flattens and deduplicates parameters on all Python versions, and the caching
   bug is also fixed on all versions.

   .. versionchanged:: 4.6.0

      Backported the bug fixes from :pr-cpy:`29334`, :pr-cpy:`23294`, and :pr-cpy:`23383`.

.. data:: LiteralString

   See :py:data:`typing.LiteralString` and :pep:`675`. In ``typing`` since 3.11.

   .. versionadded:: 4.1.0

.. class:: NamedTuple

   See :py:class:`typing.NamedTuple`.

   ``typing_extensions`` backports several changes
   to ``NamedTuple`` on Python 3.11 and lower: in 3.11,
   support for generic ``NamedTuple``\ s was added, and
   in 3.12, the ``__orig_bases__`` attribute was added.

   .. versionadded:: 4.3.0

      Added to provide support for generic ``NamedTuple``\ s.

   .. versionchanged:: 4.6.0

      Support for the ``__orig_bases__`` attribute was added.

   .. versionchanged:: 4.7.0

      The undocumented keyword argument syntax for creating NamedTuple classes
      (``NT = NamedTuple("NT", x=int)``) is deprecated, and will be disallowed
      in Python 3.15. Use the class-based syntax or the functional syntax instead.

   .. versionchanged:: 4.7.0

      When using the functional syntax to create a NamedTuple class, failing to
      pass a value to the 'fields' parameter (``NT = NamedTuple("NT")``) is
      deprecated. Passing ``None`` to the 'fields' parameter
      (``NT = NamedTuple("NT", None)``) is also deprecated. Both will be
      disallowed in Python 3.15. To create a NamedTuple class with zero fields,
      use ``class NT(NamedTuple): pass`` or ``NT = NamedTuple("NT", [])``.


.. data:: Never

   See :py:data:`typing.Never`. In ``typing`` since 3.11.

   .. versionadded:: 4.1.0

.. class:: NewType(name, tp)

   See :py:class:`typing.NewType`. In ``typing`` since 3.5.2.

   Instances of ``NewType`` were made picklable in 3.10 and an error message was
   improved in 3.11; ``typing_extensions`` backports these changes.

   .. versionchanged:: 4.6.0

      The improvements from Python 3.10 and 3.11 were backported.

.. data:: NoDefault

   See :py:class:`typing.NoDefault`. In ``typing`` since 3.13.0.

   .. versionadded:: 4.12.0

.. data:: NotRequired

   See :py:data:`typing.NotRequired` and :pep:`655`. In ``typing`` since 3.11.

   .. versionadded:: 4.0.0

.. class:: ParamSpec(name, *, default=NoDefault)

   See :py:class:`typing.ParamSpec` and :pep:`612`. In ``typing`` since 3.10.

   The ``typing_extensions`` version adds support for the
   ``default=`` argument from :pep:`696`.

   On older Python versions, ``typing_extensions.ParamSpec`` may not work
   correctly with introspection tools like :func:`get_args` and
   :func:`get_origin`. Certain special cases in user-defined
   :py:class:`typing.Generic`\ s are also not available (e.g., see :issue:`126`).

   .. versionchanged:: 4.4.0

      Added support for the ``default=`` argument.

   .. versionchanged:: 4.6.0

      The implementation was changed for compatibility with Python 3.12.

   .. versionchanged:: 4.8.0

      Passing an ellipsis literal (``...``) to *default* now works on Python
      3.10 and lower.

   .. versionchanged:: 4.12.0

      The :attr:`!__default__` attribute is now set to ``None`` if
      ``default=None`` is passed, and to :data:`NoDefault` if no value is passed.

      Previously, passing ``None`` would result in :attr:`!__default__` being set
      to :py:class:`types.NoneType`, and passing no value for the parameter would
      result in :attr:`!__default__` being set to ``None``.

   .. versionchanged:: 4.12.0

      ParamSpecs now have a ``has_default()`` method, for compatibility
      with :py:class:`typing.ParamSpec` on Python 3.13+.

.. class:: ParamSpecArgs

.. class:: ParamSpecKwargs

   See :py:class:`typing.ParamSpecArgs` and :py:class:`typing.ParamSpecKwargs`.
   In ``typing`` since 3.10.

.. class:: Protocol

   See :py:class:`typing.Protocol` and :pep:`544`. In ``typing`` since 3.8.

   Python 3.12 improves the performance of runtime-checkable protocols;
   ``typing_extensions`` backports this improvement.

   .. versionchanged:: 4.6.0

      Backported the ability to define ``__init__`` methods on Protocol classes.

   .. versionchanged:: 4.6.0

      Backported changes to runtime-checkable protocols from Python 3.12,
      including :pr-cpy:`103034` and :pr-cpy:`26067`.

   .. versionchanged:: 4.7.0

      Classes can now inherit from both :py:class:`typing.Protocol` and
      ``typing_extensions.Protocol`` simultaneously. Previously, this led to
      :py:exc:`TypeError` being raised due to a metaclass conflict.

      It is recommended to avoid doing this if possible. Not all features and
      bugfixes that ``typing_extensions.Protocol`` backports from newer Python
      versions are guaranteed to work if :py:class:`typing.Protocol` is also
      present in a protocol class's :py:term:`method resolution order`. See
      :issue:`245` for some examples.

.. data:: ReadOnly

   See :pep:`705`. Indicates that a :class:`TypedDict` item may not be modified.

   .. versionadded:: 4.9.0

.. data:: Required

   See :py:data:`typing.Required` and :pep:`655`. In ``typing`` since 3.11.

   .. versionadded:: 4.0.0

.. data:: Self

   See :py:data:`typing.Self` and :pep:`673`. In ``typing`` since 3.11.

   .. versionadded:: 4.0.0

.. data:: TypeAlias

   See :py:data:`typing.TypeAlias` and :pep:`613`. In ``typing`` since 3.10.

.. class:: TypeAliasType(name, value, *, type_params=())

   See :py:class:`typing.TypeAliasType` and :pep:`695`. In ``typing`` since 3.12.

   .. versionadded:: 4.6.0

.. data:: TypeGuard

   See :py:data:`typing.TypeGuard` and :pep:`647`. In ``typing`` since 3.10.

.. data:: TypeIs

   See :pep:`742`. Similar to :data:`TypeGuard`, but allows more type narrowing.

   .. versionadded:: 4.10.0

.. class:: TypedDict(dict, total=True)

   See :py:class:`typing.TypedDict` and :pep:`589`. In ``typing`` since 3.8.

   ``typing_extensions`` backports various bug fixes and improvements
   to ``TypedDict`` on Python 3.11 and lower.
   :py:class:`TypedDict` does not store runtime information
   about which (if any) keys are non-required in Python 3.8, and does not
   honor the ``total`` keyword with old-style ``TypedDict()`` in Python
   3.9.0 and 3.9.1. :py:class:`typing.TypedDict` also does not support multiple inheritance
   with :py:class:`typing.Generic` on Python <3.11, and :py:class:`typing.TypedDict` classes do not
   consistently have the ``__orig_bases__`` attribute on Python <3.12. The
   ``typing_extensions`` backport provides all of these features and bugfixes on
   all Python versions.

   Historically, ``TypedDict`` has supported an alternative creation syntax
   where the fields are supplied as keyword arguments (e.g.,
   ``TypedDict("TD", a=int, b=str)``). In CPython, this feature was deprecated
   in Python 3.11 and removed in Python 3.13. ``typing_extensions.TypedDict``
   raises a :py:exc:`DeprecationWarning` when this syntax is used in Python 3.12
   or lower and fails with a :py:exc:`TypeError` in Python 3.13 and higher.

   ``typing_extensions`` supports the experimental :data:`ReadOnly` qualifier
   proposed by :pep:`705`. It is reflected in the following attributes:

   .. attribute:: __readonly_keys__

      A :py:class:`frozenset` containing the names of all read-only keys. Keys
      are read-only if they carry the :data:`ReadOnly` qualifier.

      .. versionadded:: 4.9.0

   .. attribute:: __mutable_keys__

      A :py:class:`frozenset` containing the names of all mutable keys. Keys
      are mutable if they do not carry the :data:`ReadOnly` qualifier.

      .. versionadded:: 4.9.0

   The experimental ``closed`` keyword argument and the special key
   ``__extra_items__`` proposed in :pep:`728` are supported.

   When ``closed`` is unspecified or ``closed=False`` is given,
   ``__extra_items__`` behaves like a regular key. Otherwise, this becomes a
   special key that does not show up in ``__readonly_keys__``,
   ``__mutable_keys__``, ``__required_keys__``, ``__optional_keys``, or
   ``__annotations__``.

   For runtime introspection, two attributes can be looked at:

   .. attribute:: __closed__

      A boolean flag indicating whether the current ``TypedDict`` is
      considered closed. This is not inherited by the ``TypedDict``'s
      subclasses.

      .. versionadded:: 4.10.0

   .. attribute:: __extra_items__

      The type annotation of the extra items allowed on the ``TypedDict``.
      This attribute defaults to ``None`` on a TypedDict that has itself and
      all its bases non-closed. This default is different from ``type(None)``
      that represents ``__extra_items__: None`` defined on a closed
      ``TypedDict``.

      If ``__extra_items__`` is not defined or inherited on a closed
      ``TypedDict``, this defaults to ``Never``.

      .. versionadded:: 4.10.0

   .. versionchanged:: 4.3.0

      Added support for generic ``TypedDict``\ s.

   .. versionchanged:: 4.6.0

      A :py:exc:`DeprecationWarning` is now emitted when a call-based
      ``TypedDict`` is constructed using keyword arguments.

   .. versionchanged:: 4.6.0

      Support for the ``__orig_bases__`` attribute was added.

   .. versionchanged:: 4.7.0

      ``TypedDict`` is now a function rather than a class.
      This brings ``typing_extensions.TypedDict`` closer to the implementation
      of :py:mod:`typing.TypedDict` on Python 3.9 and higher.

   .. versionchanged:: 4.7.0

      When using the functional syntax to create a TypedDict class, failing to
      pass a value to the 'fields' parameter (``TD = TypedDict("TD")``) is
      deprecated. Passing ``None`` to the 'fields' parameter
      (``TD = TypedDict("TD", None)``) is also deprecated. Both will be
      disallowed in Python 3.15. To create a TypedDict class with 0 fields,
      use ``class TD(TypedDict): pass`` or ``TD = TypedDict("TD", {})``.

   .. versionchanged:: 4.9.0

      Support for the :data:`ReadOnly` qualifier was added.

   .. versionchanged:: 4.10.0

      The keyword argument ``closed`` and the special key ``__extra_items__``
      when ``closed=True`` is given were supported.

.. class:: TypeVar(name, *constraints, bound=None, covariant=False,
                   contravariant=False, infer_variance=False, default=NoDefault)

   See :py:class:`typing.TypeVar`.

   The ``typing_extensions`` version adds support for the
   ``default=`` argument from :pep:`696`, as well as the
   ``infer_variance=`` argument from :pep:`695` (also available
   in Python 3.12).

   .. versionadded:: 4.4.0

      Added in order to support the new ``default=`` and
      ``infer_variance=`` arguments.

   .. versionchanged:: 4.6.0

      The implementation was changed for compatibility with Python 3.12.

   .. versionchanged:: 4.12.0

      The :attr:`!__default__` attribute is now set to ``None`` if
      ``default=None`` is passed, and to :data:`NoDefault` if no value is passed.

      Previously, passing ``None`` would result in :attr:`!__default__` being set
      to :py:class:`types.NoneType`, and passing no value for the parameter would
      result in :attr:`!__default__` being set to ``None``.

   .. versionchanged:: 4.12.0

      TypeVars now have a ``has_default()`` method, for compatibility
      with :py:class:`typing.TypeVar` on Python 3.13+.

.. class:: TypeVarTuple(name, *, default=NoDefault)

   See :py:class:`typing.TypeVarTuple` and :pep:`646`. In ``typing`` since 3.11.

   The ``typing_extensions`` version adds support for the
   ``default=`` argument from :pep:`696`.

   .. versionadded:: 4.1.0

   .. versionchanged:: 4.4.0

      Added support for the ``default=`` argument.

   .. versionchanged:: 4.6.0

      The implementation was changed for compatibility with Python 3.12.

   .. versionchanged:: 4.12.0

      The :attr:`!__default__` attribute is now set to ``None`` if
      ``default=None`` is passed, and to :data:`NoDefault` if no value is passed.

      Previously, passing ``None`` would result in :attr:`!__default__` being set
      to :py:class:`types.NoneType`, and passing no value for the parameter would
      result in :attr:`!__default__` being set to ``None``.

   .. versionchanged:: 4.12.0

      TypeVarTuples now have a ``has_default()`` method, for compatibility
      with :py:class:`typing.TypeVarTuple` on Python 3.13+.

.. data:: Unpack

   See :py:data:`typing.Unpack` and :pep:`646`. In ``typing`` since 3.11.

   In Python 3.12, the ``repr()`` was changed as a result of :pep:`692`.
   ``typing_extensions`` backports this change.

   Generic type aliases involving ``Unpack`` may not work correctly on
   Python 3.10 and lower; see :issue:`103` for details.

   .. versionadded:: 4.1.0

   .. versionchanged:: 4.6.0

      Backport ``repr()`` changes from Python 3.12.

Abstract Base Classes
~~~~~~~~~~~~~~~~~~~~~

.. class:: Buffer

   See :py:class:`collections.abc.Buffer`. Added to the standard library
   in Python 3.12.

   .. versionadded:: 4.6.0

Protocols
~~~~~~~~~

.. class:: SupportsAbs

   See :py:class:`typing.SupportsAbs`.

   ``typing_extensions`` backports a more performant version of this
   protocol on Python 3.11 and lower.

   .. versionadded:: 4.6.0

.. class:: SupportsBytes

   See :py:class:`typing.SupportsBytes`.

   ``typing_extensions`` backports a more performant version of this
   protocol on Python 3.11 and lower.

   .. versionadded:: 4.6.0

.. class:: SupportsComplex

   See :py:class:`typing.SupportsComplex`.

   ``typing_extensions`` backports a more performant version of this
   protocol on Python 3.11 and lower.

   .. versionadded:: 4.6.0

.. class:: SupportsFloat

   See :py:class:`typing.SupportsFloat`.

   ``typing_extensions`` backports a more performant version of this
   protocol on Python 3.11 and lower.

   .. versionadded:: 4.6.0

.. class:: SupportsIndex

   See :py:class:`typing.SupportsIndex`. In ``typing`` since 3.8.

   ``typing_extensions`` backports a more performant version of this
   protocol on Python 3.11 and lower.

   .. versionchanged:: 4.6.0

      Backported the performance improvements from Python 3.12.

.. class:: SupportsInt

   See :py:class:`typing.SupportsInt`.

   ``typing_extensions`` backports a more performant version of this
   protocol on Python 3.11 and lower.

   .. versionadded:: 4.6.0

.. class:: SupportsRound

   See :py:class:`typing.SupportsRound`.

   ``typing_extensions`` backports a more performant version of this
   protocol on Python 3.11 and lower.

   .. versionadded:: 4.6.0

Decorators
~~~~~~~~~~

.. decorator:: dataclass_transform(*, eq_default=False, order_default=False,
                                   kw_only_default=False, frozen_default=False,
                                   field_specifiers=(), **kwargs)

   See :py:func:`typing.dataclass_transform` and :pep:`681`. In ``typing`` since 3.11.

   Python 3.12 adds the ``frozen_default`` parameter; ``typing_extensions``
   backports this parameter.

   .. versionadded:: 4.1.0

   .. versionchanged:: 4.2.0

      The ``field_descriptors`` parameter was renamed to ``field_specifiers``.
      For compatibility, the decorator now accepts arbitrary keyword arguments.

   .. versionchanged:: 4.5.0

      The ``frozen_default`` parameter was added.

.. decorator:: deprecated(msg, *, category=DeprecationWarning, stacklevel=1)

   See :pep:`702`. In the :mod:`warnings` module since Python 3.13.

   .. versionadded:: 4.5.0

   .. versionchanged:: 4.9.0

      Inheriting from a deprecated class now also raises a runtime
      :py:exc:`DeprecationWarning`.

.. decorator:: final

   See :py:func:`typing.final` and :pep:`591`. In ``typing`` since 3.8.

   Since Python 3.11, this decorator supports runtime introspection
   by setting the ``__final__`` attribute wherever possible; ``typing_extensions.final``
   backports this feature.

   .. versionchanged:: 4.1.0

      The decorator now attempts to set the ``__final__`` attribute on decorated objects.

.. decorator:: overload

   See :py:func:`typing.overload`.

   Since Python 3.11, this decorator supports runtime introspection
   through :func:`get_overloads`; ``typing_extensions.overload``
   backports this feature.

   .. versionchanged:: 4.2.0

      Introspection support via :func:`get_overloads` was added.

.. decorator:: override

   See :py:func:`typing.override` and :pep:`698`. In ``typing`` since 3.12.

   .. versionadded:: 4.4.0

   .. versionchanged:: 4.5.0

      The decorator now attempts to set the ``__override__`` attribute on the decorated
      object.

.. decorator:: runtime_checkable

   See :py:func:`typing.runtime_checkable`. In ``typing`` since 3.8.

   In Python 3.12, the performance of runtime-checkable protocols was
   improved, and ``typing_extensions`` backports these performance
   improvements.

Functions
~~~~~~~~~

.. function:: assert_never(arg)

   See :py:func:`typing.assert_never`. In ``typing`` since 3.11.

   .. versionadded:: 4.1.0

.. function:: assert_type(val, typ)

   See :py:func:`typing.assert_type`. In ``typing`` since 3.11.

   .. versionadded:: 4.2.0

.. function:: clear_overloads()

   See :py:func:`typing.clear_overloads`. In ``typing`` since 3.11.

   .. versionadded:: 4.2.0

.. function:: get_args(tp)

   See :py:func:`typing.get_args`. In ``typing`` since 3.8.

   This function was changed in 3.9 and 3.10 to deal with :data:`Annotated`
   and :class:`ParamSpec` correctly; ``typing_extensions`` backports these
   fixes.

.. function:: get_origin(tp)

   See :py:func:`typing.get_origin`. In ``typing`` since 3.8.

   This function was changed in 3.9 and 3.10 to deal with :data:`Annotated`
   and :class:`ParamSpec` correctly; ``typing_extensions`` backports these
   fixes.

.. function:: get_original_bases(cls)

   See :py:func:`types.get_original_bases`. Added to the standard library
   in Python 3.12.

   This function should always produce correct results when called on classes
   constructed using features from ``typing_extensions``. However, it may
   produce incorrect results when called on some :py:class:`NamedTuple` or
   :py:class:`TypedDict` classes on Python <=3.11.

   .. versionadded:: 4.6.0

.. function:: get_overloads(func)

   See :py:func:`typing.get_overloads`. In ``typing`` since 3.11.

   Before Python 3.11, this works only with overloads created through
   :func:`overload`, not with :py:func:`typing.overload`.

   .. versionadded:: 4.2.0

.. function:: get_protocol_members(tp)

   Return the set of members defined in a :class:`Protocol`. This works with protocols
   defined using either :class:`typing.Protocol` or :class:`typing_extensions.Protocol`.

   ::

      >>> from typing_extensions import Protocol, get_protocol_members
      >>> class P(Protocol):
      ...     def a(self) -> str: ...
      ...     b: int
      >>> get_protocol_members(P)
      frozenset({'a', 'b'})

   Raise :py:exc:`TypeError` for arguments that are not Protocols.

   .. versionadded:: 4.7.0

.. function:: get_type_hints(obj, globalns=None, localns=None, include_extras=False)

   See :py:func:`typing.get_type_hints`.

   In Python 3.11, this function was changed to support the new
   :py:data:`typing.Required` and :py:data:`typing.NotRequired`.
   ``typing_extensions`` backports these fixes.

   .. versionchanged:: 4.1.0

      Interaction with :data:`Required` and :data:`NotRequired`.

   .. versionchanged:: 4.11.0

      When ``include_extra=False``, ``get_type_hints()`` now strips
      :data:`ReadOnly` from the annotation.

.. function:: is_protocol(tp)

   Determine if a type is a :class:`Protocol`. This works with protocols
   defined using either :py:class:`typing.Protocol` or :class:`typing_extensions.Protocol`.

   For example::

      class P(Protocol):
          def a(self) -> str: ...
          b: int

      is_protocol(P)    # => True
      is_protocol(int)  # => False

   .. versionadded:: 4.7.0

.. function:: is_typeddict(tp)

   See :py:func:`typing.is_typeddict`. In ``typing`` since 3.10.

   On versions where :class:`TypedDict` is not the same as
   :py:class:`typing.TypedDict`, this function recognizes
   ``TypedDict`` classes created through either mechanism.

   .. versionadded:: 4.1.0

   .. versionchanged:: 4.7.0

      :func:`is_typeddict` now returns ``False`` when called with
      :data:`TypedDict` itself as the argument, consistent with the
      behavior of :py:func:`typing.is_typeddict`.

.. function:: reveal_type(obj)

   See :py:func:`typing.reveal_type`. In ``typing`` since 3.11.

   .. versionadded:: 4.1.0


Annotation metadata
~~~~~~~~~~~~~~~~~~~

.. class:: Doc(documentation, /)

   Define the documentation of a type annotation using :data:`Annotated`, to be
   used in class attributes, function and method parameters, return values,
   and variables.

   The value should be a positional-only string literal to allow static tools
   like editors and documentation generators to use it.

   This complements docstrings.

   The string value passed is available in the attribute ``documentation``.

   Example::

      >>> from typing_extensions import Annotated, Doc
      >>> def hi(to: Annotated[str, Doc("Who to say hi to")]) -> None: ...

   .. versionadded:: 4.8.0

      See :pep:`727`.

   .. attribute:: documentation

      The documentation string passed to :class:`Doc`.


Pure aliases
~~~~~~~~~~~~

Most of these are simply re-exported from the :mod:`typing` module on all supported
versions of Python, but all are listed here for completeness.

.. class:: AbstractSet

   See :py:class:`typing.AbstractSet`.

   .. versionadded:: 4.7.0

.. data:: AnyStr

   See :py:data:`typing.AnyStr`.

   .. versionadded:: 4.7.0

.. class:: AsyncContextManager

   See :py:class:`typing.AsyncContextManager`. In ``typing`` since 3.5.4 and 3.6.2.

   .. versionchanged:: 4.12.0

      ``AsyncContextManager`` now has an optional second parameter, defaulting to
      ``Optional[bool]``, signifying the return type of the ``__aexit__`` method.

.. class:: AsyncGenerator

   See :py:class:`typing.AsyncGenerator`. In ``typing`` since 3.6.1.

   .. versionchanged:: 4.12.0

      The second type parameter is now optional (it defaults to ``None``).

.. class:: AsyncIterable

   See :py:class:`typing.AsyncIterable`. In ``typing`` since 3.5.2.

.. class:: AsyncIterator

   See :py:class:`typing.AsyncIterator`. In ``typing`` since 3.5.2.

.. class:: Awaitable

   See :py:class:`typing.Awaitable`. In ``typing`` since 3.5.2.

.. class:: BinaryIO

   See :py:class:`typing.BinaryIO`.

   .. versionadded:: 4.7.0

.. data:: Callable

   See :py:data:`typing.Callable`.

   .. versionadded:: 4.7.0

.. class:: ChainMap

   See :py:class:`typing.ChainMap`. In ``typing`` since 3.5.4 and 3.6.1.

.. data:: ClassVar

   See :py:data:`typing.ClassVar` and :pep:`526`. In ``typing`` since 3.5.3.

.. class:: Collection

   See :py:class:`typing.Collection`.

   .. versionadded:: 4.7.0

.. class:: Container

   See :py:class:`typing.Container`.

   .. versionadded:: 4.7.0

.. class:: ContextManager

   See :py:class:`typing.ContextManager`. In ``typing`` since 3.5.4.

   .. versionchanged:: 4.12.0

      ``ContextManager`` now has an optional second parameter, defaulting to
      ``Optional[bool]``, signifying the return type of the ``__exit__`` method.

.. class:: Coroutine

   See :py:class:`typing.Coroutine`. In ``typing`` since 3.5.3.

.. class:: Counter

   See :py:class:`typing.Counter`. In ``typing`` since 3.5.4 and 3.6.1.

.. class:: DefaultDict

   See :py:class:`typing.DefaultDict`. In ``typing`` since 3.5.2.

.. class:: Deque

   See :py:class:`typing.Deque`. In ``typing`` since 3.5.4 and 3.6.1.

.. class:: Dict

   See :py:class:`typing.Dict`.

   .. versionadded:: 4.7.0

.. class:: ForwardRef

   See :py:class:`typing.ForwardRef`.

   .. versionadded:: 4.7.0

.. class:: FrozenSet

   See :py:class:`typing.FrozenSet`.

   .. versionadded:: 4.7.0

.. class:: Generator

   See :py:class:`typing.Generator`.

   .. versionadded:: 4.7.0

   .. versionchanged:: 4.12.0

      The second type and third type parameters are now optional
      (they both default to ``None``).

.. class:: Generic

   See :py:class:`typing.Generic`.

   .. versionadded:: 4.7.0

.. class:: Hashable

   See :py:class:`typing.Hashable`.

   .. versionadded:: 4.7.0

.. class:: IO

   See :py:class:`typing.IO`.

   .. versionadded:: 4.7.0

.. class:: ItemsView

   See :py:class:`typing.ItemsView`.

   .. versionadded:: 4.7.0

.. class:: Iterable

   See :py:class:`typing.Iterable`.

   .. versionadded:: 4.7.0

.. class:: Iterator

   See :py:class:`typing.Iterator`.

   .. versionadded:: 4.7.0

.. class:: KeysView

   See :py:class:`typing.KeysView`.

   .. versionadded:: 4.7.0

.. class:: List

   See :py:class:`typing.List`.

   .. versionadded:: 4.7.0

.. class:: Mapping

   See :py:class:`typing.Mapping`.

   .. versionadded:: 4.7.0

.. class:: MappingView

   See :py:class:`typing.MappingView`.

   .. versionadded:: 4.7.0

.. class:: Match

   See :py:class:`typing.Match`.

   .. versionadded:: 4.7.0

.. class:: MutableMapping

   See :py:class:`typing.MutableMapping`.

   .. versionadded:: 4.7.0

.. class:: MutableSequence

   See :py:class:`typing.MutableSequence`.

   .. versionadded:: 4.7.0

.. class:: MutableSet

   See :py:class:`typing.MutableSet`.

   .. versionadded:: 4.7.0

.. data:: NoReturn

   See :py:data:`typing.NoReturn`. In ``typing`` since 3.5.4 and 3.6.2.

.. data:: Optional

   See :py:data:`typing.Optional`.

   .. versionadded:: 4.7.0

.. class:: OrderedDict

   See :py:class:`typing.OrderedDict`. In ``typing`` since 3.7.2.

.. class:: Pattern

   See :py:class:`typing.Pattern`.

   .. versionadded:: 4.7.0

.. class:: Reversible

   See :py:class:`typing.Reversible`.

   .. versionadded:: 4.7.0

.. class:: Sequence

   See :py:class:`typing.Sequence`.

   .. versionadded:: 4.7.0

.. class:: Set

   See :py:class:`typing.Set`.

   .. versionadded:: 4.7.0

.. class:: Sized

   See :py:class:`typing.Sized`.

   .. versionadded:: 4.7.0

.. class:: Text

   See :py:class:`typing.Text`. In ``typing`` since 3.5.2.

.. class:: TextIO

   See :py:class:`typing.TextIO`.

   .. versionadded:: 4.7.0

.. data:: Tuple

   See :py:data:`typing.Tuple`.

   .. versionadded:: 4.7.0

.. class:: Type

   See :py:class:`typing.Type`. In ``typing`` since 3.5.2.

.. data:: TYPE_CHECKING

   See :py:data:`typing.TYPE_CHECKING`. In ``typing`` since 3.5.2.

.. data:: Union

   See :py:data:`typing.Union`.

   .. versionadded:: 4.7.0

.. class:: ValuesView

   See :py:class:`typing.ValuesView`.

   .. versionadded:: 4.7.0

.. function:: cast

   See :py:func:`typing.cast`.

   .. versionadded:: 4.7.0

.. decorator:: no_type_check

   See :py:func:`typing.no_type_check`.

   .. versionadded:: 4.7.0

.. decorator:: no_type_check_decorator

   See :py:func:`typing.no_type_check_decorator`.

   .. versionadded:: 4.7.0


Welcome to typing_extensions's documentation!
=============================================

``typing_extensions`` provides common building blocks for type annotations
that are understood by static type checkers.

The ``typing_extensions`` module serves two related purposes:

- Enable use of new type system features on older Python versions. For example,
  :py:data:`typing.TypeGuard` is new in Python 3.10, but ``typing_extensions`` allows
  users on previous Python versions to use it too.
- Enable experimentation with new type system PEPs before they are accepted and
  added to the :py:mod:`typing` module.

New features may be added to ``typing_extensions`` as soon as they are specified
in a PEP that has been added to the `python/peps <https://github.com/python/peps>`_
repository. If the PEP is accepted, the feature will then be added to :py:mod:`typing`
for the next CPython release. No typing PEP that affected ``typing_extensions``
has been rejected so far, so we haven't yet figured out how to deal with that possibility.

Starting with version 4.0.0, ``typing_extensions`` uses
`Semantic Versioning <https://semver.org>`_. The
major version is incremented for all backwards-incompatible changes.
Therefore, it's safe to depend
on ``typing_extensions`` like this: ``typing_extensions >=x.y, <(x+1)``,
where ``x.y`` is the first version that includes all features you need.
In view of the wide usage of ``typing_extensions`` across the ecosystem,
we are highly hesitant to break backwards compatibility, and we do not
expect to increase the major version number in the foreseeable future.

``typing_extensions`` supports Python versions 3.7 and higher. In the future,
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

   See :py:data:`typing.Annotated`. In ``typing`` since 3.9.

   .. versionchanged:: 4.1.0

      ``Annotated`` can now wrap :data:`ClassVar` and :data:`Final`.

.. data:: Any

   See :py:data:`typing.Any`.

   Since Python 3.11, ``typing.Any`` can be used as a base class.
   ``typing_extensions.Any`` supports this feature on older versions.

   .. versionadded:: 4.4.0

      Added to support inheritance from ``Any``.

.. data:: ClassVar

   See :py:data:`typing.ClassVar`. In ``typing`` since 3.5.3.

.. data:: Concatenate

   See :py:data:`typing.Concatenate`. In ``typing`` since 3.10.

.. data:: Final

   See :py:data:`typing.Final`. In ``typing`` since 3.8.

.. data:: Literal

   See :py:data:`typing.Literal`. In ``typing`` since 3.8.

   :py:data:`typing.Literal` does not flatten or deduplicate parameters on Python <3.9.1, and a
   caching bug was fixed in 3.10.1/3.9.8. The ``typing_extensions`` version
   flattens and deduplicates parameters on all Python versions, and the caching
   bug is also fixed on all versions.

   .. versionchanged:: 4.6.0

      Backported the bug fixes from :py:gh:`29334`, :py:gh:`23294`, and :py:gh:`23383`.

.. data:: LiteralString

   See :py:data:`typing.LiteralString`. In ``typing`` since 3.11.

   .. versionadded:: 4.1.0

.. class:: NamedTuple

   See :py:class:`typing.NamedTuple`.

   ``typing_extensions`` backports several changes
   to ``NamedTuple`` on Python 3.11 and lower: in 3.11,
   support for generic ``NamedTuple`` was added, and
   in 3.12, the ``__orig_bases__`` attribute was added.

   .. versionadded:: 4.3.0

      Added to provide support for generic ``NamedTuple``.

   .. versionchanged:: 4.6.0

      Support for the ``__orig_bases__`` attribute was added.

.. data:: Never

   See :py:data:`typing.Never`. In ``typing`` since 3.11.

   .. versionadded:: 4.1.0

.. class:: NewType(name, tp)

   See :py:class:`typing.NewType`. In ``typing`` since 3.5.2.

   ``NewType`` was made picklable in 3.10 and an error message was
   improved in 3.11; ``typing_extensions`` backports these changes.

   .. versionchanged:: 4.6.0

      The improvements from Python 3.10 and 3.11 were backported.

.. data:: NoReturn

   See :py:data:`typing.NoReturn`. In ``typing`` since 3.5.4 and 3.6.2.

.. data:: NotRequired

   See :py:data:`typing.NotRequired`. In ``typing`` since 3.11.

   .. versionadded:: 4.0.0

.. class:: ParamSpec(name, *, default=...)

   See :py:class:`typing.ParamSpec`. In ``typing`` since 3.10.

   The ``typing_extensions`` version adds support for the
   ``default=`` argument from :pep:`696`.

   .. versionchanged:: 4.4.0

      Added support for the ``default=`` argument.

   .. versionchanged:: 4.6.0

      The implementation was changed for compatibility with Python 3.12.

.. class:: ParamSpecArgs

.. class:: ParamSpecKwargs

   See :py:class:`typing.ParamSpecArgs` and :py:class:`typing.ParamSpecKwargs`.
   In ``typing`` since 3.10.

.. class:: Protocol

   See :py:class:`typing.Protocol`. In ``typing`` since 3.8.

   Python 3.12 improves the performance of runtime-checkable protocols;
   ``typing_extensions`` backports this improvement.

   .. versionchanged:: 4.6.0

      Backported the ability to define ``__init__`` methods on Protocol classes.

   .. versionchanged:: 4.6.0

      Backported changes to runtime-checkable protocols from Python 3.12.

.. data:: Required

   See :py:data:`typing.Required`. In ``typing`` since 3.11.

   .. versionadded:: 4.0.0

.. data:: Self

   See :py:data:`typing.Self`. In ``typing`` since 3.11.

   .. versionadded:: 4.0.0

.. class:: Type

   See :py:class:`typing.Type`. In ``typing`` since 3.5.2.

.. data:: TypeAlias

   See :py:data:`typing.TypeAlias`. In ``typing`` since 3.10.

.. class:: TypeAliasType(name, value, *, type_params=())

   See :py:class:`typing.TypeAliasType`. In ``typing`` since 3.12.

   .. versionadded:: 4.6.0

.. data:: TypeGuard

   See :py:data:`typing.TypeGuard`. In ``typing`` since 3.10.

.. class:: TypedDict

   See :py:class:`typing.TypedDict`. In ``typing`` since 3.8.

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

   .. versionadded:: 4.3.0

      Added to provide support for generic ``TypedDict``.

   .. versionchanged:: 4.6.0

      A :py:exc:`DeprecationWarning` is now emitted when a call-based
      ``TypedDict`` is constructed using keyword arguments.

   .. versionchanged:: 4.6.0

      Support for the ``__orig_bases__`` attribute was added.

.. class:: TypeVar(name, *constraints, bound=None, covariant=False,
                   contravariant=False, infer_variance=False, default=...)

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

.. class:: TypeVarTuple(name, *, default=...)

   See :py:class:`typing.TypeVarTuple`. In ``typing`` since 3.10.

   The ``typing_extensions`` version adds support for the
   ``default=`` argument from :pep:`696`.

   .. versionadded:: 4.1.0

   .. versionchanged:: 4.4.0

      Added support for the ``default=`` argument.

   .. versionchanged:: 4.6.0

      The implementation was changed for compatibility with Python 3.12.

.. data:: Unpack

   See :py:data:`typing.Unpack`. In ``typing`` since 3.10.

   In Python 3.12, the ``repr()`` was changed as a result of :pep:`692`.
   ``typing_extensions`` backports this change.

   .. versionadded:: 4.1.0

   .. versionchanged:: 4.6.0

      Backport ``repr()`` changes from Python 3.12.

Generic concrete collections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: ChainMap

   See :py:class:`typing.ChainMap`. In ``typing`` since 3.5.4 and 3.6.1.

.. class:: Counter

   See :py:class:`typing.Counter`. In ``typing`` since 3.5.4 and 3.6.1.

.. class:: DefaultDict

   See :py:class:`typing.DefaultDict`. In ``typing`` since 3.5.2.

.. class:: Deque

   See :py:class:`typing.Deque`. In ``typing`` since 3.5.4 and 3.6.1.

.. class:: OrderedDict

   See :py:class:`typing.OrderedDict`. In ``typing`` since 3.7.2.

Abstract Base Classes
~~~~~~~~~~~~~~~~~~~~~

.. class:: AsyncContextManager

   See :py:class:`typing.AsyncContextManager`. In ``typing`` since 3.5.4 and 3.6.2.

.. class:: AsyncGenerator

   See :py:class:`typing.AsyncGenerator`. In ``typing`` since 3.6.1.

.. class:: AsyncIterable

   See :py:class:`typing.AsyncIterable`. In ``typing`` since 3.5.2.

.. class:: AsyncIterator

   See :py:class:`typing.AsyncIterator`. In ``typing`` since 3.5.2.

.. class:: Awaitable

   See :py:class:`typing.Awaitable`. In ``typing`` since 3.5.2.

.. class:: Buffer

   See :py:class:`collections.abc.Buffer`. Added to the standard library
   in Python 3.12.

   .. versionadded:: 4.6.0

.. class:: ContextManager

   See :py:class:`typing.ContextManager`. In ``typing`` since 3.5.4.

.. class:: Coroutine

   See :py:class:`typing.Coroutine`. In ``typing`` since 3.5.3.

Protocols
~~~~~~~~~

.. class:: SupportsAbs

   See :py:class:`typing.SupportsAbs`.

   ``typing_extensions`` backports a more performant version of this
   protocol in Python 3.11 and lower.

   .. versionadded:: 4.6.0

.. class:: SupportsBytes

   See :py:class:`typing.SupportsBytes`.

   ``typing_extensions`` backports a more performant version of this
   protocol in Python 3.11 and lower.

   .. versionadded:: 4.6.0

.. class:: SupportsComplex

   See :py:class:`typing.SupportsComplex`.

   ``typing_extensions`` backports a more performant version of this
   protocol in Python 3.11 and lower.

   .. versionadded:: 4.6.0

.. class:: SupportsFloat

   See :py:class:`typing.SupportsFloat`.

   ``typing_extensions`` backports a more performant version of this
   protocol in Python 3.11 and lower.

   .. versionadded:: 4.6.0

.. class:: SupportsIndex

   See :py:class:`typing.SupportsIndex`. In ``typing`` since 3.8.

   ``typing_extensions`` backports a more performant version of this
   protocol in Python 3.11 and lower.

   .. versionchanged:: 4.6.0

      Backported the performance improvements from Python 3.12.

.. class:: SupportsInt

   See :py:class:`typing.SupportsInt`.

   ``typing_extensions`` backports a more performant version of this
   protocol in Python 3.11 and lower.

   .. versionadded:: 4.6.0

.. class:: SupportsRound

   See :py:class:`typing.SupportsRound`.

   ``typing_extensions`` backports a more performant version of this
   protocol in Python 3.11 and lower.

   .. versionadded:: 4.6.0

Decorators
~~~~~~~~~~

.. decorator:: dataclass_transform(*, eq_default=False, order_default=False,
                                   kw_only_default=False, frozen_default=False,
                                   field_specifiers=(), **kwargs)

   See :py:func:`typing.dataclass_transform`. In ``typing`` since 3.11.

   Python 3.12 adds the ``frozen_default`` parameter; ``typing_extensions``
   backports this parameter.

   .. versionadded:: 4.1.0

   .. versionchanged:: 4.2.0

      The ``field_descriptors`` parameter was renamed to ``field_specifiers``.
      For compatibility, the decorator now accepts arbitrary keyword arguments.

   .. versionchanged:: 4.5.0

      The ``frozen_default`` parameter was added.

.. decorator:: deprecated(msg, *, category=DeprecationWarning, stacklevel=1)

   See :pep:`702`.

   .. versionadded:: 4.5.0

.. decorator:: final

   See :py:func:`typing.final`. In ``typing`` since 3.8.

   Since Python 3.11, this decorator supports runtime introspection
   through the ``__final__`` attribute; ``typing_extensions.final``
   backports this feature.

   .. versionchanged:: 4.1.0

      The ``__final__`` attribute is now set on decorated objects.

.. decorator:: overload

   See :py:func:`typing.overload`.

   Since Python 3.11, this decorator supports runtime introspection
   through :func:`get_overloads`; ``typing_extensions.overload``
   backports this feature.

   .. versionchanged:: 4.2.0

      Added to provide support for introspection through :func:`get_overloads` was added.

.. decorator:: override

   See :py:func:`typing.override`. In ``typing`` since 3.12.

   .. versionadded:: 4.4.0

   .. versionchanged:: 4.5.0

      The decorator now sets the ``__override__`` attribute on the decorated
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
   :py:class:`TypedDict` on Python <=3.11.

   .. versionadded:: 4.6.0

.. function:: get_overloads(func)

   See :py:func:`typing.get_overloads`. In ``typing`` since 3.11.

   Before Python 3.11, this works only with overloads created through
   :func:`overload`, not with :py:func:`typing.overload`.

   .. versionadded:: 4.2.0

.. function:: get_type_hints(obj, globalns=None, localns=None, include_extras=False)

   See :py:func:`typing.get_type_hints`.

   In Python 3.11, this function was changed to support the new
   :py:data:`typing.Required`. ``typing_extensions`` backports these
   fixes.

   .. versionchanged:: 4.1.0

      Interaction with :data:`Required` and :data:`NotRequired`.

.. function:: is_typeddict(tp)

   See :py:func:`typing.is_typeddict`. In ``typing`` since 3.10.

   On versions where :class:`TypedDict` is not the same as
   :py:class:`typing.TypedDict`, this function recognizes
   ``TypedDict`` classes created through either.

   .. versionadded:: 4.1.0

.. function:: reveal_type(obj)

   See :py:func:`typing.reveal_type`. In ``typing`` since 3.11.

   .. versionadded:: 4.1.0

Other
~~~~~

.. class:: Text

   See :py:class:`typing.Text`. In ``typing`` since 3.5.2.

.. data:: TYPE_CHECKING

   See :py:data:`typing.TYPE_CHECKING`. In ``typing`` since 3.5.2.

Limitations
-----------

There are a few types whose interface was modified between different
versions of typing. For example, :py:class:`typing.Sequence` was modified to
subclass :py:class:`typing.Reversible` as of Python 3.5.3.

These changes are *not* backported to prevent subtle compatibility
issues when mixing the differing implementations of modified classes.

Certain types have incorrect runtime behavior due to limitations of older
versions of the typing module:

- :class:`ParamSpec` and :data:`Concatenate` will not work with :func:`get_args` and
  :func:`get_origin`.
- Certain :pep:`612` special cases in user-defined
  :py:class:`typing.Generic`\ s are also not available.

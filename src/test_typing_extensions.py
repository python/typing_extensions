import abc
import asyncio
import collections
import collections.abc
import contextlib
import copy
import functools
import gc
import importlib
import inspect
import io
import itertools
import pickle
import re
import subprocess
import sys
import tempfile
import textwrap
import types
import typing
import warnings
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from unittest import TestCase, main, skipIf, skipUnless
from unittest.mock import patch

import typing_extensions
from _typed_dict_test_helper import Foo, FooGeneric, VeryAnnotated
from typing_extensions import (
    _FORWARD_REF_HAS_CLASS,
    Annotated,
    Any,
    AnyStr,
    AsyncContextManager,
    AsyncIterator,
    Awaitable,
    Buffer,
    Callable,
    ClassVar,
    Concatenate,
    Dict,
    Doc,
    Final,
    Format,
    Generic,
    IntVar,
    Iterable,
    Iterator,
    List,
    Literal,
    LiteralString,
    NamedTuple,
    Never,
    NewType,
    NoDefault,
    NoExtraItems,
    NoReturn,
    NotRequired,
    Optional,
    ParamSpec,
    ParamSpecArgs,
    ParamSpecKwargs,
    Protocol,
    ReadOnly,
    Required,
    Self,
    Sentinel,
    Set,
    Tuple,
    Type,
    TypeAlias,
    TypeAliasType,
    TypedDict,
    TypeForm,
    TypeGuard,
    TypeIs,
    TypeVar,
    TypeVarTuple,
    Union,
    Unpack,
    assert_never,
    assert_type,
    clear_overloads,
    dataclass_transform,
    deprecated,
    disjoint_base,
    evaluate_forward_ref,
    final,
    get_annotations,
    get_args,
    get_origin,
    get_original_bases,
    get_overloads,
    get_protocol_members,
    get_type_hints,
    is_protocol,
    is_typeddict,
    no_type_check,
    overload,
    override,
    reveal_type,
    runtime,
    runtime_checkable,
    type_repr,
)

NoneType = type(None)
T = TypeVar("T")
KT = TypeVar("KT")
VT = TypeVar("VT")

# Flags used to mark tests that only apply after a specific
# version of the typing module.
TYPING_3_10_0 = sys.version_info[:3] >= (3, 10, 0)

# 3.11 makes runtime type checks (_type_check) more lenient.
TYPING_3_11_0 = sys.version_info[:3] >= (3, 11, 0)

# 3.12 changes the representation of Unpack[] (PEP 692)
# and adds PEP 695 to CPython's grammar
TYPING_3_12_0 = sys.version_info[:3] >= (3, 12, 0)

# @deprecated works differently in Python 3.12
TYPING_3_12_ONLY = (3, 12) <= sys.version_info < (3, 13)

# 3.13 drops support for the keyword argument syntax of TypedDict
TYPING_3_13_0 = sys.version_info[:3] >= (3, 13, 0)

# 3.13.0.rc1 fixes a problem with @deprecated
TYPING_3_13_0_RC = sys.version_info[:4] >= (3, 13, 0, "candidate")

TYPING_3_14_0 = sys.version_info[:3] >= (3, 14, 0)

# https://github.com/python/cpython/pull/27017 was backported into some 3.9 and 3.10
# versions, but not all
HAS_FORWARD_MODULE = "module" in inspect.signature(typing._type_check).parameters

skip_if_py313_beta_1 = skipIf(
    sys.version_info[:5] == (3, 13, 0, 'beta', 1),
    "Bugfixes will be released in 3.13.0b2"
)

ANN_MODULE_SOURCE = '''\
import sys
from typing import List, Optional
from functools import wraps

try:
    __annotations__[1] = 2
except NameError:
    assert sys.version_info >= (3, 14)

class C:

    x = 5; y: Optional['C'] = None

from typing import Tuple
x: int = 5; y: str = x; f: Tuple[int, int]

class M(type):
    try:
        __annotations__['123'] = 123
    except NameError:
        assert sys.version_info >= (3, 14)
    o: type = object

(pars): bool = True

class D(C):
    j: str = 'hi'; k: str= 'bye'

from types import new_class
h_class = new_class('H', (C,))
j_class = new_class('J')

class F():
    z: int = 5
    def __init__(self, x):
        pass

class Y(F):
    def __init__(self):
        super(F, self).__init__(123)

class Meta(type):
    def __new__(meta, name, bases, namespace):
        return super().__new__(meta, name, bases, namespace)

class S(metaclass = Meta):
    x: str = 'something'
    y: str = 'something else'

def foo(x: int = 10):
    def bar(y: List[str]):
        x: str = 'yes'
    bar()

def dec(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
'''

ANN_MODULE_2_SOURCE = '''\
from typing import no_type_check, ClassVar

i: int = 1
j: int
x: float = i/10

def f():
    class C: ...
    return C()

f().new_attr: object = object()

class C:
    def __init__(self, x: int) -> None:
        self.x = x

c = C(5)
c.new_attr: int = 10

__annotations__ = {}


@no_type_check
class NTC:
    def meth(self, param: complex) -> None:
        ...

class CV:
    var: ClassVar['CV']

CV.var = CV()
'''

ANN_MODULE_3_SOURCE = '''\
def f_bad_ann():
    __annotations__[1] = 2

class C_OK:
    def __init__(self, x: int) -> None:
        self.x: no_such_name = x  # This one is OK as proposed by Guido

class D_bad_ann:
    def __init__(self, x: int) -> None:
        sfel.y: int = 0

def g_bad_ann():
    no_such_name.attr: int = 0
'''


STOCK_ANNOTATIONS = """
a:int=3
b:str="foo"

class MyClass:
    a:int=4
    b:str="bar"
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __eq__(self, other):
        return isinstance(other, MyClass) and self.a == other.a and self.b == other.b

def function(a:int, b:str) -> MyClass:
    return MyClass(a, b)


def function2(a:int, b:"str", c:MyClass) -> MyClass:
    pass


def function3(a:"int", b:"str", c:"MyClass"):
    pass


class UnannotatedClass:
    pass

def unannotated_function(a, b, c): pass
"""

STRINGIZED_ANNOTATIONS = """
from __future__ import annotations

a:int=3
b:str="foo"

class MyClass:
    a:int=4
    b:str="bar"
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __eq__(self, other):
        return isinstance(other, MyClass) and self.a == other.a and self.b == other.b

def function(a:int, b:str) -> MyClass:
    return MyClass(a, b)


def function2(a:int, b:"str", c:MyClass) -> MyClass:
    pass


def function3(a:"int", b:"str", c:"MyClass"):
    pass


class UnannotatedClass:
    pass

def unannotated_function(a, b, c): pass

class MyClassWithLocalAnnotations:
    mytype = int
    x: mytype
"""

STRINGIZED_ANNOTATIONS_2 = """
from __future__ import annotations


def foo(a, b, c):  pass
"""

if TYPING_3_12_0:
    STRINGIZED_ANNOTATIONS_PEP_695 = textwrap.dedent(
        """
        from __future__ import annotations
        from typing import Callable, Unpack


        class A[T, *Ts, **P]:
            x: T
            y: tuple[*Ts]
            z: Callable[P, str]


        class B[T, *Ts, **P]:
            T = int
            Ts = str
            P = bytes
            x: T
            y: Ts
            z: P


        Eggs = int
        Spam = str


        class C[Eggs, **Spam]:
            x: Eggs
            y: Spam


        def generic_function[T, *Ts, **P](
            x: T, *y: Unpack[Ts], z: P.args, zz: P.kwargs
        ) -> None: ...


        def generic_function_2[Eggs, **Spam](x: Eggs, y: Spam): pass


        class D:
            Foo = int
            Bar = str

            def generic_method[Foo, **Bar](
                self, x: Foo, y: Bar
            ) -> None: ...

            def generic_method_2[Eggs, **Spam](self, x: Eggs, y: Spam): pass


        # Eggs is `int` in globals, a TypeVar in type_params, and `str` in locals:
        class E[Eggs]:
            Eggs = str
            x: Eggs



        def nested():
            from types import SimpleNamespace
            from typing_extensions import get_annotations

            Eggs = bytes
            Spam = memoryview


            class F[Eggs, **Spam]:
                x: Eggs
                y: Spam

                def generic_method[Eggs, **Spam](self, x: Eggs, y: Spam): pass


            def generic_function[Eggs, **Spam](x: Eggs, y: Spam): pass


            # Eggs is `int` in globals, `bytes` in the function scope,
            # a TypeVar in the type_params, and `str` in locals:
            class G[Eggs]:
                Eggs = str
                x: Eggs


            return SimpleNamespace(
                F=F,
                F_annotations=get_annotations(F, eval_str=True),
                F_meth_annotations=get_annotations(F.generic_method, eval_str=True),
                G_annotations=get_annotations(G, eval_str=True),
                generic_func=generic_function,
                generic_func_annotations=get_annotations(generic_function, eval_str=True)
            )
        """
    )
else:
    STRINGIZED_ANNOTATIONS_PEP_695 = None


class BaseTestCase(TestCase):
    def assertIsSubclass(self, cls, class_or_tuple, msg=None):
        if not issubclass(cls, class_or_tuple):
            message = f'{cls!r} is not a subclass of {class_or_tuple!r}'
            if msg is not None:
                message += f' : {msg}'
            raise self.failureException(message)

    def assertNotIsSubclass(self, cls, class_or_tuple, msg=None):
        if issubclass(cls, class_or_tuple):
            message = f'{cls!r} is a subclass of {class_or_tuple!r}'
            if msg is not None:
                message += f' : {msg}'
            raise self.failureException(message)


class EqualToForwardRef:
    """Helper to ease use of annotationlib.ForwardRef in tests.

    This checks only attributes that can be set using the constructor.

    """

    def __init__(
        self,
        arg,
        *,
        module=None,
        owner=None,
        is_class=False,
    ):
        self.__forward_arg__ = arg
        self.__forward_is_class__ = is_class
        self.__forward_module__ = module
        self.__owner__ = owner

    def __eq__(self, other):
        if not isinstance(other, (EqualToForwardRef, typing.ForwardRef)):
            return NotImplemented
        if sys.version_info >= (3, 14) and self.__owner__ != other.__owner__:
            return False
        return (
            self.__forward_arg__ == other.__forward_arg__
            and self.__forward_module__ == other.__forward_module__
            and self.__forward_is_class__ == other.__forward_is_class__
        )

    def __repr__(self):
        extra = []
        if self.__forward_module__ is not None:
            extra.append(f", module={self.__forward_module__!r}")
        if self.__forward_is_class__:
            extra.append(", is_class=True")
        if sys.version_info >= (3, 14) and self.__owner__ is not None:
            extra.append(f", owner={self.__owner__!r}")
        return f"EqualToForwardRef({self.__forward_arg__!r}{''.join(extra)})"


class Employee:
    pass


class BottomTypeTestsMixin:
    bottom_type: ClassVar[Any]

    def test_equality(self):
        self.assertEqual(self.bottom_type, self.bottom_type)
        self.assertIs(self.bottom_type, self.bottom_type)
        self.assertNotEqual(self.bottom_type, None)

    def test_get_origin(self):
        self.assertIs(get_origin(self.bottom_type), None)

    def test_instance_type_error(self):
        with self.assertRaises(TypeError):
            isinstance(42, self.bottom_type)

    def test_subclass_type_error(self):
        with self.assertRaises(TypeError):
            issubclass(Employee, self.bottom_type)
        with self.assertRaises(TypeError):
            issubclass(NoReturn, self.bottom_type)

    def test_not_generic(self):
        with self.assertRaises(TypeError):
            self.bottom_type[int]

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class A(self.bottom_type):
                pass
        with self.assertRaises(TypeError):
            class B(type(self.bottom_type)):
                pass

    def test_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            self.bottom_type()
        with self.assertRaises(TypeError):
            type(self.bottom_type)()

    def test_pickle(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            pickled = pickle.dumps(self.bottom_type, protocol=proto)
            self.assertIs(self.bottom_type, pickle.loads(pickled))


class AssertNeverTests(BaseTestCase):
    def test_exception(self):
        with self.assertRaises(AssertionError):
            assert_never(None)

        value = "some value"
        with self.assertRaisesRegex(AssertionError, value):
            assert_never(value)

        # Make sure a huge value doesn't get printed in its entirety
        huge_value = "a" * 10000
        with self.assertRaises(AssertionError) as cm:
            assert_never(huge_value)
        self.assertLess(
            len(cm.exception.args[0]),
            typing_extensions._ASSERT_NEVER_REPR_MAX_LENGTH * 2,
        )


class OverrideTests(BaseTestCase):
    def test_override(self):
        class Base:
            def normal_method(self): ...
            @staticmethod
            def static_method_good_order(): ...
            @staticmethod
            def static_method_bad_order(): ...
            @staticmethod
            def decorator_with_slots(): ...

        class Derived(Base):
            @override
            def normal_method(self):
                return 42

            @staticmethod
            @override
            def static_method_good_order():
                return 42

            @override
            @staticmethod
            def static_method_bad_order():
                return 42

        self.assertIsSubclass(Derived, Base)
        instance = Derived()
        self.assertEqual(instance.normal_method(), 42)
        self.assertIs(True, instance.normal_method.__override__)
        self.assertEqual(Derived.static_method_good_order(), 42)
        self.assertIs(True, Derived.static_method_good_order.__override__)
        self.assertEqual(Derived.static_method_bad_order(), 42)
        self.assertIs(False, hasattr(Derived.static_method_bad_order, "__override__"))


class DeprecatedTests(BaseTestCase):
    def test_dunder_deprecated(self):
        @deprecated("A will go away soon")
        class A:
            pass

        self.assertEqual(A.__deprecated__, "A will go away soon")
        self.assertIsInstance(A, type)

        @deprecated("b will go away soon")
        def b():
            pass

        self.assertEqual(b.__deprecated__, "b will go away soon")
        self.assertIsInstance(b, types.FunctionType)

        @overload
        @deprecated("no more ints")
        def h(x: int) -> int: ...
        @overload
        def h(x: str) -> str: ...
        def h(x):
            return x

        overloads = get_overloads(h)
        self.assertEqual(len(overloads), 2)
        self.assertEqual(overloads[0].__deprecated__, "no more ints")

    def test_class(self):
        @deprecated("A will go away soon")
        class A:
            pass

        with self.assertWarnsRegex(DeprecationWarning, "A will go away soon"):
            A()
        with self.assertWarnsRegex(DeprecationWarning, "A will go away soon"):
            with self.assertRaises(TypeError):
                A(42)

    def test_class_with_init(self):
        @deprecated("HasInit will go away soon")
        class HasInit:
            def __init__(self, x):
                self.x = x

        with self.assertWarnsRegex(DeprecationWarning, "HasInit will go away soon"):
            instance = HasInit(42)
        self.assertEqual(instance.x, 42)

    def test_class_with_new(self):
        has_new_called = False

        @deprecated("HasNew will go away soon")
        class HasNew:
            def __new__(cls, x):
                nonlocal has_new_called
                has_new_called = True
                return super().__new__(cls)

            def __init__(self, x) -> None:
                self.x = x

        with self.assertWarnsRegex(DeprecationWarning, "HasNew will go away soon"):
            instance = HasNew(42)
        self.assertEqual(instance.x, 42)
        self.assertTrue(has_new_called)

    def test_class_with_inherited_new(self):
        new_base_called = False

        class NewBase:
            def __new__(cls, x):
                nonlocal new_base_called
                new_base_called = True
                return super().__new__(cls)

            def __init__(self, x) -> None:
                self.x = x

        @deprecated("HasInheritedNew will go away soon")
        class HasInheritedNew(NewBase):
            pass

        with self.assertWarnsRegex(DeprecationWarning, "HasInheritedNew will go away soon"):
            instance = HasInheritedNew(42)
        self.assertEqual(instance.x, 42)
        self.assertTrue(new_base_called)

    def test_class_with_new_but_no_init(self):
        new_called = False

        @deprecated("HasNewNoInit will go away soon")
        class HasNewNoInit:
            def __new__(cls, x):
                nonlocal new_called
                new_called = True
                obj = super().__new__(cls)
                obj.x = x
                return obj

        with self.assertWarnsRegex(DeprecationWarning, "HasNewNoInit will go away soon"):
            instance = HasNewNoInit(42)
        self.assertEqual(instance.x, 42)
        self.assertTrue(new_called)

    def test_mixin_class(self):
        @deprecated("Mixin will go away soon")
        class Mixin:
            pass

        class Base:
            def __init__(self, a) -> None:
                self.a = a

        with self.assertWarnsRegex(DeprecationWarning, "Mixin will go away soon"):
            class Child(Base, Mixin):
                pass

        instance = Child(42)
        self.assertEqual(instance.a, 42)

    def test_do_not_shadow_user_arguments(self):
        new_called = False
        new_called_cls = None

        @deprecated("MyMeta will go away soon")
        class MyMeta(type):
            def __new__(mcs, name, bases, attrs, cls=None):
                nonlocal new_called, new_called_cls
                new_called = True
                new_called_cls = cls
                return super().__new__(mcs, name, bases, attrs)

        with self.assertWarnsRegex(DeprecationWarning, "MyMeta will go away soon"):
            class Foo(metaclass=MyMeta, cls='haha'):
                pass

        self.assertTrue(new_called)
        self.assertEqual(new_called_cls, 'haha')

    def test_existing_init_subclass(self):
        @deprecated("C will go away soon")
        class C:
            def __init_subclass__(cls) -> None:
                cls.inited = True

        with self.assertWarnsRegex(DeprecationWarning, "C will go away soon"):
            C()

        with self.assertWarnsRegex(DeprecationWarning, "C will go away soon"):
            class D(C):
                pass

        self.assertTrue(D.inited)
        self.assertIsInstance(D(), D)  # no deprecation

    def test_existing_init_subclass_in_base(self):
        class Base:
            def __init_subclass__(cls, x) -> None:
                cls.inited = x

        @deprecated("C will go away soon")
        class C(Base, x=42):
            pass

        self.assertEqual(C.inited, 42)

        with self.assertWarnsRegex(DeprecationWarning, "C will go away soon"):
            C()

        with self.assertWarnsRegex(DeprecationWarning, "C will go away soon"):
            class D(C, x=3):
                pass

        self.assertEqual(D.inited, 3)

    def test_init_subclass_has_correct_cls(self):
        init_subclass_saw = None

        @deprecated("Base will go away soon")
        class Base:
            def __init_subclass__(cls) -> None:
                nonlocal init_subclass_saw
                init_subclass_saw = cls

        self.assertIsNone(init_subclass_saw)

        with self.assertWarnsRegex(DeprecationWarning, "Base will go away soon"):
            class C(Base):
                pass

        self.assertIs(init_subclass_saw, C)

    def test_init_subclass_with_explicit_classmethod(self):
        init_subclass_saw = None

        @deprecated("Base will go away soon")
        class Base:
            @classmethod
            def __init_subclass__(cls) -> None:
                nonlocal init_subclass_saw
                init_subclass_saw = cls

        self.assertIsNone(init_subclass_saw)

        with self.assertWarnsRegex(DeprecationWarning, "Base will go away soon"):
            class C(Base):
                pass

        self.assertIs(init_subclass_saw, C)

    def test_function(self):
        @deprecated("b will go away soon")
        def b():
            pass

        with self.assertWarnsRegex(DeprecationWarning, "b will go away soon"):
            b()

    def test_method(self):
        class Capybara:
            @deprecated("x will go away soon")
            def x(self):
                pass

        instance = Capybara()
        with self.assertWarnsRegex(DeprecationWarning, "x will go away soon"):
            instance.x()

    def test_property(self):
        class Capybara:
            @property
            @deprecated("x will go away soon")
            def x(self):
                pass

            @property
            def no_more_setting(self):
                return 42

            @no_more_setting.setter
            @deprecated("no more setting")
            def no_more_setting(self, value):
                pass

        instance = Capybara()
        with self.assertWarnsRegex(DeprecationWarning, "x will go away soon"):
            instance.x

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(instance.no_more_setting, 42)

        with self.assertWarnsRegex(DeprecationWarning, "no more setting"):
            instance.no_more_setting = 42

    def test_category(self):
        @deprecated("c will go away soon", category=RuntimeWarning)
        def c():
            pass

        with self.assertWarnsRegex(RuntimeWarning, "c will go away soon"):
            c()

    def test_turn_off_warnings(self):
        @deprecated("d will go away soon", category=None)
        def d():
            pass

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            d()

    def test_only_strings_allowed(self):
        with self.assertRaisesRegex(
            TypeError,
            "Expected an object of type str for 'message', not 'type'"
        ):
            @deprecated
            class Foo: ...

        with self.assertRaisesRegex(
            TypeError,
            "Expected an object of type str for 'message', not 'function'"
        ):
            @deprecated
            def foo(): ...

    def test_no_retained_references_to_wrapper_instance(self):
        @deprecated('depr')
        def d(): pass

        self.assertFalse(any(
            isinstance(cell.cell_contents, deprecated) for cell in d.__closure__
        ))

@deprecated("depr")
def func():
    pass

@deprecated("depr")
async def coro():
    pass

class Cls:
    @deprecated("depr")
    def func(self):
        pass

    @deprecated("depr")
    async def coro(self):
        pass

class DeprecatedCoroTests(BaseTestCase):
    def test_asyncio_iscoroutinefunction(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.assertFalse(asyncio.coroutines.iscoroutinefunction(func))
            self.assertFalse(asyncio.coroutines.iscoroutinefunction(Cls.func))
            self.assertTrue(asyncio.coroutines.iscoroutinefunction(coro))
            self.assertTrue(asyncio.coroutines.iscoroutinefunction(Cls.coro))

    @skipUnless(TYPING_3_12_ONLY or TYPING_3_13_0_RC, "inspect.iscoroutinefunction works differently on Python < 3.12")
    def test_inspect_iscoroutinefunction(self):
        self.assertFalse(inspect.iscoroutinefunction(func))
        self.assertFalse(inspect.iscoroutinefunction(Cls.func))
        self.assertTrue(inspect.iscoroutinefunction(coro))
        self.assertTrue(inspect.iscoroutinefunction(Cls.coro))


class AnyTests(BaseTestCase):
    def test_can_subclass(self):
        class Mock(Any): pass
        self.assertTrue(issubclass(Mock, Any))
        self.assertIsInstance(Mock(), Mock)

        class Something: pass
        self.assertFalse(issubclass(Something, Any))
        self.assertNotIsInstance(Something(), Mock)

        class MockSomething(Something, Mock): pass
        self.assertTrue(issubclass(MockSomething, Any))
        ms = MockSomething()
        self.assertIsInstance(ms, MockSomething)
        self.assertIsInstance(ms, Something)
        self.assertIsInstance(ms, Mock)

    class SubclassesAny(Any):
        ...

    def test_repr(self):
        if sys.version_info >= (3, 11):
            mod_name = 'typing'
        else:
            mod_name = 'typing_extensions'
        self.assertEqual(repr(Any), f"{mod_name}.Any")

    @skipIf(sys.version_info[:3] == (3, 11, 0), "A bug was fixed in 3.11.1")
    def test_repr_on_Any_subclass(self):
        self.assertEqual(
            repr(self.SubclassesAny),
            f"<class '{self.SubclassesAny.__module__}.AnyTests.SubclassesAny'>"
        )

    def test_instantiation(self):
        with self.assertRaises(TypeError):
            Any()

        self.SubclassesAny()

    def test_isinstance(self):
        with self.assertRaises(TypeError):
            isinstance(object(), Any)

        isinstance(object(), self.SubclassesAny)


class ClassVarTests(BaseTestCase):

    def test_basics(self):
        if not TYPING_3_11_0:
            with self.assertRaises(TypeError):
                ClassVar[1]
        with self.assertRaises(TypeError):
            ClassVar[int, str]
        with self.assertRaises(TypeError):
            ClassVar[int][str]

    def test_repr(self):
        if hasattr(typing, 'ClassVar'):
            mod_name = 'typing'
        else:
            mod_name = 'typing_extensions'
        self.assertEqual(repr(ClassVar), mod_name + '.ClassVar')
        cv = ClassVar[int]
        self.assertEqual(repr(cv), mod_name + '.ClassVar[int]')
        cv = ClassVar[Employee]
        self.assertEqual(repr(cv), mod_name + f'.ClassVar[{__name__}.Employee]')

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class C(type(ClassVar)):
                pass
        with self.assertRaises(TypeError):
            class D(type(ClassVar[int])):
                pass

    def test_cannot_init(self):
        with self.assertRaises(TypeError):
            ClassVar()
        with self.assertRaises(TypeError):
            type(ClassVar)()
        with self.assertRaises(TypeError):
            type(ClassVar[Optional[int]])()

    def test_no_isinstance(self):
        with self.assertRaises(TypeError):
            isinstance(1, ClassVar[int])
        with self.assertRaises(TypeError):
            issubclass(int, ClassVar)


class FinalTests(BaseTestCase):

    def test_basics(self):
        if not TYPING_3_11_0:
            with self.assertRaises(TypeError):
                Final[1]
        with self.assertRaises(TypeError):
            Final[int, str]
        with self.assertRaises(TypeError):
            Final[int][str]

    def test_repr(self):
        self.assertEqual(repr(Final), 'typing.Final')
        cv = Final[int]
        self.assertEqual(repr(cv), 'typing.Final[int]')
        cv = Final[Employee]
        self.assertEqual(repr(cv), f'typing.Final[{__name__}.Employee]')

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class C(type(Final)):
                pass
        with self.assertRaises(TypeError):
            class D(type(Final[int])):
                pass

    def test_cannot_init(self):
        with self.assertRaises(TypeError):
            Final()
        with self.assertRaises(TypeError):
            type(Final)()
        with self.assertRaises(TypeError):
            type(Final[Optional[int]])()

    def test_no_isinstance(self):
        with self.assertRaises(TypeError):
            isinstance(1, Final[int])
        with self.assertRaises(TypeError):
            issubclass(int, Final)


class RequiredTests(BaseTestCase):

    def test_basics(self):
        if not TYPING_3_11_0:
            with self.assertRaises(TypeError):
                Required[1]
        with self.assertRaises(TypeError):
            Required[int, str]
        with self.assertRaises(TypeError):
            Required[int][str]

    def test_repr(self):
        if hasattr(typing, 'Required'):
            mod_name = 'typing'
        else:
            mod_name = 'typing_extensions'
        self.assertEqual(repr(Required), f'{mod_name}.Required')
        cv = Required[int]
        self.assertEqual(repr(cv), f'{mod_name}.Required[int]')
        cv = Required[Employee]
        self.assertEqual(repr(cv), f'{mod_name}.Required[{__name__}.Employee]')

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class C(type(Required)):
                pass
        with self.assertRaises(TypeError):
            class D(type(Required[int])):
                pass

    def test_cannot_init(self):
        with self.assertRaises(TypeError):
            Required()
        with self.assertRaises(TypeError):
            type(Required)()
        with self.assertRaises(TypeError):
            type(Required[Optional[int]])()

    def test_no_isinstance(self):
        with self.assertRaises(TypeError):
            isinstance(1, Required[int])
        with self.assertRaises(TypeError):
            issubclass(int, Required)


class NotRequiredTests(BaseTestCase):

    def test_basics(self):
        if not TYPING_3_11_0:
            with self.assertRaises(TypeError):
                NotRequired[1]
        with self.assertRaises(TypeError):
            NotRequired[int, str]
        with self.assertRaises(TypeError):
            NotRequired[int][str]

    def test_repr(self):
        if hasattr(typing, 'NotRequired'):
            mod_name = 'typing'
        else:
            mod_name = 'typing_extensions'
        self.assertEqual(repr(NotRequired), f'{mod_name}.NotRequired')
        cv = NotRequired[int]
        self.assertEqual(repr(cv), f'{mod_name}.NotRequired[int]')
        cv = NotRequired[Employee]
        self.assertEqual(repr(cv), f'{mod_name}.NotRequired[{ __name__}.Employee]')

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class C(type(NotRequired)):
                pass
        with self.assertRaises(TypeError):
            class D(type(NotRequired[int])):
                pass

    def test_cannot_init(self):
        with self.assertRaises(TypeError):
            NotRequired()
        with self.assertRaises(TypeError):
            type(NotRequired)()
        with self.assertRaises(TypeError):
            type(NotRequired[Optional[int]])()

    def test_no_isinstance(self):
        with self.assertRaises(TypeError):
            isinstance(1, NotRequired[int])
        with self.assertRaises(TypeError):
            issubclass(int, NotRequired)


class IntVarTests(BaseTestCase):
    def test_valid(self):
        IntVar("T_ints")

    def test_invalid(self):
        with self.assertRaises(TypeError):
            IntVar("T_ints", int)
        with self.assertRaises(TypeError):
            IntVar("T_ints", bound=int)
        with self.assertRaises(TypeError):
            IntVar("T_ints", covariant=True)


class LiteralTests(BaseTestCase):
    def test_basics(self):
        Literal[1]
        Literal[1, 2, 3]
        Literal["x", "y", "z"]
        Literal[None]

    def test_enum(self):
        import enum
        class My(enum.Enum):
            A = 'A'

        self.assertEqual(Literal[My.A].__args__, (My.A,))

    def test_strange_parameters_are_allowed(self):
        # These are explicitly allowed by the typing spec
        Literal[Literal[1, 2], Literal[4, 5]]
        Literal[b"foo", "bar"]

        # Type checkers should reject these types, but we do not
        # raise errors at runtime to maintain maximum flexibility
        Literal[int]
        Literal[3j + 2, ..., ()]
        Literal[{"foo": 3, "bar": 4}]
        Literal[T]

    def test_literals_inside_other_types(self):
        List[Literal[1, 2, 3]]
        List[Literal[("foo", "bar", "baz")]]

    def test_repr(self):
        # we backport various bugfixes that were added in 3.10.1 and earlier
        if sys.version_info >= (3, 10, 1):
            mod_name = 'typing'
        else:
            mod_name = 'typing_extensions'
        self.assertEqual(repr(Literal[1]), mod_name + ".Literal[1]")
        self.assertEqual(repr(Literal[1, True, "foo"]), mod_name + ".Literal[1, True, 'foo']")
        self.assertEqual(repr(Literal[int]), mod_name + ".Literal[int]")
        self.assertEqual(repr(Literal), mod_name + ".Literal")
        self.assertEqual(repr(Literal[None]), mod_name + ".Literal[None]")
        self.assertEqual(repr(Literal[1, 2, 3, 3]), mod_name + ".Literal[1, 2, 3]")

    def test_cannot_init(self):
        with self.assertRaises(TypeError):
            Literal()
        with self.assertRaises(TypeError):
            Literal[1]()
        with self.assertRaises(TypeError):
            type(Literal)()
        with self.assertRaises(TypeError):
            type(Literal[1])()

    def test_no_isinstance_or_issubclass(self):
        with self.assertRaises(TypeError):
            isinstance(1, Literal[1])
        with self.assertRaises(TypeError):
            isinstance(int, Literal[1])
        with self.assertRaises(TypeError):
            issubclass(1, Literal[1])
        with self.assertRaises(TypeError):
            issubclass(int, Literal[1])

    def test_no_subclassing(self):
        with self.assertRaises(TypeError):
            class Foo(Literal[1]): pass
        with self.assertRaises(TypeError):
            class Bar(Literal): pass

    def test_no_multiple_subscripts(self):
        with self.assertRaises(TypeError):
            Literal[1][1]

    def test_equal(self):
        self.assertNotEqual(Literal[0], Literal[False])
        self.assertNotEqual(Literal[True], Literal[1])
        self.assertNotEqual(Literal[1], Literal[2])
        self.assertNotEqual(Literal[1, True], Literal[1])
        self.assertNotEqual(Literal[1, True], Literal[1, 1])
        self.assertNotEqual(Literal[1, 2], Literal[True, 2])
        self.assertEqual(Literal[1], Literal[1])
        self.assertEqual(Literal[1, 2], Literal[2, 1])
        self.assertEqual(Literal[1, 2, 3], Literal[1, 2, 3, 3])

    def test_hash(self):
        self.assertEqual(hash(Literal[1]), hash(Literal[1]))
        self.assertEqual(hash(Literal[1, 2]), hash(Literal[2, 1]))
        self.assertEqual(hash(Literal[1, 2, 3]), hash(Literal[1, 2, 3, 3]))

    def test_args(self):
        self.assertEqual(Literal[1, 2, 3].__args__, (1, 2, 3))
        self.assertEqual(Literal[1, 2, 3, 3].__args__, (1, 2, 3))
        self.assertEqual(Literal[1, Literal[2], Literal[3, 4]].__args__, (1, 2, 3, 4))
        # Mutable arguments will not be deduplicated
        self.assertEqual(Literal[[], []].__args__, ([], []))

    def test_union_of_literals(self):
        self.assertEqual(Union[Literal[1], Literal[2]].__args__,
                         (Literal[1], Literal[2]))
        self.assertEqual(Union[Literal[1], Literal[1]],
                         Literal[1])

        self.assertEqual(Union[Literal[False], Literal[0]].__args__,
                         (Literal[False], Literal[0]))
        self.assertEqual(Union[Literal[True], Literal[1]].__args__,
                         (Literal[True], Literal[1]))

        import enum
        class Ints(enum.IntEnum):
            A = 0
            B = 1

        self.assertEqual(Union[Literal[Ints.A], Literal[Ints.B]].__args__,
                         (Literal[Ints.A], Literal[Ints.B]))

        self.assertEqual(Union[Literal[Ints.A], Literal[Ints.A]],
                         Literal[Ints.A])
        self.assertEqual(Union[Literal[Ints.B], Literal[Ints.B]],
                         Literal[Ints.B])

        self.assertEqual(Union[Literal[0], Literal[Ints.A], Literal[False]].__args__,
                         (Literal[0], Literal[Ints.A], Literal[False]))
        self.assertEqual(Union[Literal[1], Literal[Ints.B], Literal[True]].__args__,
                         (Literal[1], Literal[Ints.B], Literal[True]))

    @skipUnless(TYPING_3_10_0, "Python 3.10+ required")
    def test_or_type_operator_with_Literal(self):
        self.assertEqual((Literal[1] | Literal[2]).__args__,
                         (Literal[1], Literal[2]))

        self.assertEqual((Literal[0] | Literal[False]).__args__,
                         (Literal[0], Literal[False]))
        self.assertEqual((Literal[1] | Literal[True]).__args__,
                         (Literal[1], Literal[True]))

        self.assertEqual(Literal[1] | Literal[1], Literal[1])
        self.assertEqual(Literal['a'] | Literal['a'], Literal['a'])

        import enum
        class Ints(enum.IntEnum):
            A = 0
            B = 1

        self.assertEqual(Literal[Ints.A] | Literal[Ints.A], Literal[Ints.A])
        self.assertEqual(Literal[Ints.B] | Literal[Ints.B], Literal[Ints.B])

        self.assertEqual((Literal[Ints.B] | Literal[Ints.A]).__args__,
                         (Literal[Ints.B], Literal[Ints.A]))

        self.assertEqual((Literal[0] | Literal[Ints.A]).__args__,
                         (Literal[0], Literal[Ints.A]))
        self.assertEqual((Literal[1] | Literal[Ints.B]).__args__,
                         (Literal[1], Literal[Ints.B]))

    def test_flatten(self):
        l1 = Literal[Literal[1], Literal[2], Literal[3]]
        l2 = Literal[Literal[1, 2], 3]
        l3 = Literal[Literal[1, 2, 3]]
        for lit in l1, l2, l3:
            self.assertEqual(lit, Literal[1, 2, 3])
            self.assertEqual(lit.__args__, (1, 2, 3))

    def test_does_not_flatten_enum(self):
        import enum
        class Ints(enum.IntEnum):
            A = 1
            B = 2

        literal = Literal[
            Literal[Ints.A],
            Literal[Ints.B],
            Literal[1],
            Literal[2],
        ]
        self.assertEqual(literal.__args__, (Ints.A, Ints.B, 1, 2))

    def test_caching_of_Literal_respects_type(self):
        self.assertIs(type(Literal[1].__args__[0]), int)
        self.assertIs(type(Literal[True].__args__[0]), bool)


class MethodHolder:
    @classmethod
    def clsmethod(cls): ...
    @staticmethod
    def stmethod(): ...
    def method(self): ...


if TYPING_3_11_0:
    registry_holder = typing
else:
    registry_holder = typing_extensions


class OverloadTests(BaseTestCase):

    def test_overload_fails(self):
        with self.assertRaises(RuntimeError):

            @overload
            def blah():
                pass

            blah()

    def test_overload_succeeds(self):
        @overload
        def blah():
            pass

        def blah():
            pass

        blah()

    @skipIf(
        sys.implementation.name == "pypy",
        "sum() and print() are not compiled in pypy"
    )
    @patch(
        f"{registry_holder.__name__}._overload_registry",
        defaultdict(lambda: defaultdict(dict))
    )
    def test_overload_on_compiled_functions(self):
        registry = registry_holder._overload_registry
        # The registry starts out empty:
        self.assertEqual(registry, {})

        # This should just not fail:
        overload(sum)
        overload(print)

        # No overloads are recorded:
        self.assertEqual(get_overloads(sum), [])
        self.assertEqual(get_overloads(print), [])

    def set_up_overloads(self):
        def blah():
            pass

        overload1 = blah
        overload(blah)

        def blah():
            pass

        overload2 = blah
        overload(blah)

        def blah():
            pass

        return blah, [overload1, overload2]

    # Make sure we don't clear the global overload registry
    @patch(
        f"{registry_holder.__name__}._overload_registry",
        defaultdict(lambda: defaultdict(dict))
    )
    def test_overload_registry(self):
        registry = registry_holder._overload_registry
        # The registry starts out empty
        self.assertEqual(registry, {})

        impl, overloads = self.set_up_overloads()
        self.assertNotEqual(registry, {})
        self.assertEqual(list(get_overloads(impl)), overloads)

        def some_other_func(): pass
        overload(some_other_func)
        other_overload = some_other_func
        def some_other_func(): pass
        self.assertEqual(list(get_overloads(some_other_func)), [other_overload])
        # Unrelated function still has no overloads:
        def not_overloaded(): pass
        self.assertEqual(list(get_overloads(not_overloaded)), [])

        # Make sure that after we clear all overloads, the registry is
        # completely empty.
        clear_overloads()
        self.assertEqual(registry, {})
        self.assertEqual(get_overloads(impl), [])

        # Querying a function with no overloads shouldn't change the registry.
        def the_only_one(): pass
        self.assertEqual(get_overloads(the_only_one), [])
        self.assertEqual(registry, {})

    def test_overload_registry_repeated(self):
        for _ in range(2):
            impl, overloads = self.set_up_overloads()

            self.assertEqual(list(get_overloads(impl)), overloads)


class AssertTypeTests(BaseTestCase):

    def test_basics(self):
        arg = 42
        self.assertIs(assert_type(arg, int), arg)
        self.assertIs(assert_type(arg, Union[str, float]), arg)
        self.assertIs(assert_type(arg, AnyStr), arg)
        self.assertIs(assert_type(arg, None), arg)

    def test_errors(self):
        # Bogus calls are not expected to fail.
        arg = 42
        self.assertIs(assert_type(arg, 42), arg)
        self.assertIs(assert_type(arg, 'hello'), arg)


T_a = TypeVar('T_a')

class AwaitableWrapper(Awaitable[T_a]):

    def __init__(self, value):
        self.value = value

    def __await__(self) -> typing.Iterator[T_a]:
        yield
        return self.value

class AsyncIteratorWrapper(AsyncIterator[T_a]):

    def __init__(self, value: Iterable[T_a]):
        self.value = value

    def __aiter__(self) -> AsyncIterator[T_a]:
        return self

    async def __anext__(self) -> T_a:
        data = await self.value
        if data:
            return data
        else:
            raise StopAsyncIteration

class ACM:
    async def __aenter__(self) -> int:
        return 42

    async def __aexit__(self, etype, eval, tb):
        return None


class A:
    y: float
class B(A):
    x: ClassVar[Optional['B']] = None
    y: int
    b: int
class CSub(B):
    z: ClassVar['CSub'] = B()
class G(Generic[T]):
    lst: ClassVar[List[T]] = []

class Loop:
    attr: Final['Loop']

class NoneAndForward:
    parent: 'NoneAndForward'
    meaning: None

class XRepr(NamedTuple):
    x: int
    y: int = 1

    def __str__(self):
        return f'{self.x} -> {self.y}'

    def __add__(self, other):
        return 0

@runtime_checkable
class HasCallProtocol(Protocol):
    __call__: typing.Callable


async def g_with(am: AsyncContextManager[int]):
    x: int
    async with am as x:
        return x

try:
    g_with(ACM()).send(None)
except StopIteration as e:
    assert e.args[0] == 42

Label = TypedDict('Label', [('label', str)])

class Point2D(TypedDict):
    x: int
    y: int

class Point2Dor3D(Point2D, total=False):
    z: int

class LabelPoint2D(Point2D, Label): ...

class Options(TypedDict, total=False):
    log_level: int
    log_path: str

class BaseAnimal(TypedDict):
    name: str

class Animal(BaseAnimal, total=False):
    voice: str
    tail: bool

class Cat(Animal):
    fur_color: str

class TotalMovie(TypedDict):
    title: str
    year: NotRequired[int]

class NontotalMovie(TypedDict, total=False):
    title: Required[str]
    year: int

class ParentNontotalMovie(TypedDict, total=False):
    title: Required[str]

class ChildTotalMovie(ParentNontotalMovie):
    year: NotRequired[int]

class ParentDeeplyAnnotatedMovie(TypedDict):
    title: Annotated[Annotated[Required[str], "foobar"], "another level"]

class ChildDeeplyAnnotatedMovie(ParentDeeplyAnnotatedMovie):
    year: NotRequired[Annotated[int, 2000]]

class AnnotatedMovie(TypedDict):
    title: Annotated[Required[str], "foobar"]
    year: NotRequired[Annotated[int, 2000]]

class WeirdlyQuotedMovie(TypedDict):
    title: Annotated['Annotated[Required[str], "foobar"]', "another level"]
    year: NotRequired['Annotated[int, 2000]']


gth = get_type_hints


class GetTypeHintTests(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tempdir:
            sys.path.append(tempdir)
            Path(tempdir, "ann_module.py").write_text(ANN_MODULE_SOURCE)
            Path(tempdir, "ann_module2.py").write_text(ANN_MODULE_2_SOURCE)
            Path(tempdir, "ann_module3.py").write_text(ANN_MODULE_3_SOURCE)
            cls.ann_module = importlib.import_module("ann_module")
            cls.ann_module2 = importlib.import_module("ann_module2")
            cls.ann_module3 = importlib.import_module("ann_module3")
        sys.path.pop()

    @classmethod
    def tearDownClass(cls):
        for modname in "ann_module", "ann_module2", "ann_module3":
            delattr(cls, modname)
            del sys.modules[modname]

    def test_get_type_hints_modules(self):
        if sys.version_info >= (3, 14):
            ann_module_type_hints = {'f': Tuple[int, int], 'x': int, 'y': str}
        else:
            ann_module_type_hints = {1: 2, 'f': Tuple[int, int], 'x': int, 'y': str}
        self.assertEqual(gth(self.ann_module), ann_module_type_hints)
        self.assertEqual(gth(self.ann_module2), {})
        self.assertEqual(gth(self.ann_module3), {})

    def test_get_type_hints_classes(self):
        self.assertEqual(gth(self.ann_module.C, self.ann_module.__dict__),
                         {'y': Optional[self.ann_module.C]})
        self.assertIsInstance(gth(self.ann_module.j_class), dict)
        if sys.version_info >= (3, 14):
            self.assertEqual(gth(self.ann_module.M), {'o': type})
        else:
            self.assertEqual(gth(self.ann_module.M), {'123': 123, 'o': type})
        self.assertEqual(gth(self.ann_module.D),
                         {'j': str, 'k': str, 'y': Optional[self.ann_module.C]})
        self.assertEqual(gth(self.ann_module.Y), {'z': int})
        self.assertEqual(gth(self.ann_module.h_class),
                         {'y': Optional[self.ann_module.C]})
        self.assertEqual(gth(self.ann_module.S), {'x': str, 'y': str})
        self.assertEqual(gth(self.ann_module.foo), {'x': int})
        self.assertEqual(gth(NoneAndForward, globals()),
                         {'parent': NoneAndForward, 'meaning': type(None)})

    def test_respect_no_type_check(self):
        @no_type_check
        class NoTpCheck:
            class Inn:
                def __init__(self, x: 'not a type'): ...  # noqa: F722  # (yes, there's a syntax error in this annotation, that's the point)
        self.assertTrue(NoTpCheck.__no_type_check__)
        self.assertTrue(NoTpCheck.Inn.__init__.__no_type_check__)
        self.assertEqual(gth(self.ann_module2.NTC.meth), {})
        class ABase(Generic[T]):
            def meth(x: int): ...
        @no_type_check
        class Der(ABase): ...
        self.assertEqual(gth(ABase.meth), {'x': int})

    def test_get_type_hints_ClassVar(self):
        self.assertEqual(gth(self.ann_module2.CV, self.ann_module2.__dict__),
                         {'var': ClassVar[self.ann_module2.CV]})
        self.assertEqual(gth(B, globals()),
                         {'y': int, 'x': ClassVar[Optional[B]], 'b': int})
        self.assertEqual(gth(CSub, globals()),
                         {'z': ClassVar[CSub], 'y': int, 'b': int,
                          'x': ClassVar[Optional[B]]})
        self.assertEqual(gth(G), {'lst': ClassVar[List[T]]})

    def test_final_forward_ref(self):
        self.assertEqual(gth(Loop, globals())['attr'], Final[Loop])
        self.assertNotEqual(gth(Loop, globals())['attr'], Final[int])
        self.assertNotEqual(gth(Loop, globals())['attr'], Final)

    def test_annotation_and_optional_default(self):
        annotation = Annotated[Union[int, None], "data"]
        NoneAlias = None
        StrAlias = str
        T_default = TypeVar("T_default", default=None)
        Ts = TypeVarTuple("Ts")

        cases = {
            # annotation: expected_type_hints
            Annotated[None, "none"] : Annotated[None, "none"],
            annotation              : annotation,
            Optional[int]           : Optional[int],
            Optional[List[str]]     : Optional[List[str]],
            Optional[annotation]     : Optional[annotation],
            Union[str, None, str]   : Optional[str],
            Unpack[Tuple[int, None]]: Unpack[Tuple[int, None]],
            # Note: A starred *Ts will use typing.Unpack in 3.11+ see Issue #485
            Unpack[Ts]              : Unpack[Ts],
        }
        # contains a ForwardRef, TypeVar(~prefix) or no expression
        do_not_stringify_cases = {
            ()          : {},  # Special-cased below to create an unannotated parameter
            int         : int,
            "int"       : int,
            None        : type(None),
            "NoneAlias" : type(None),
            List["str"] : List[str],
            Union[str, "str"]                  : str,
            Union[str, None, "str"]            : Optional[str],
            Union[str, "NoneAlias", "StrAlias"]: Optional[str],
            Union[str, "Union[None, StrAlias]"]: Optional[str],
            Union["annotation", T_default]     : Union[annotation, T_default],
            Annotated["annotation", "nested"]  : Annotated[Union[int, None], "data", "nested"],
        }
        if TYPING_3_10_0:  # cannot construct UnionTypes before 3.10
            do_not_stringify_cases["str | NoneAlias | StrAlias"] = str | None
            cases[str | None] = Optional[str]
        cases.update(do_not_stringify_cases)
        for (annot, expected), none_default, as_str, wrap_optional in itertools.product(
            cases.items(), (False, True), (False, True), (False, True)
        ):
            # Special case:
            skip_reason = None
            annot_unchanged = annot
            if sys.version_info[:2] == (3, 10) and annot == "str | NoneAlias | StrAlias" and none_default:
                # In 3.10 converts Optional[str | None] to Optional[str] which has a different repr
                skip_reason = "UnionType not preserved in 3.10"
            if wrap_optional:
                if annot_unchanged == ():
                    continue
                annot = Optional[annot]
                expected = {"x": Optional[expected]}
            else:
                expected = {"x": expected} if annot_unchanged != () else {}
            if as_str:
                if annot_unchanged in do_not_stringify_cases or annot_unchanged == ():
                    continue
                annot = str(annot)
            with self.subTest(
                annotation=annot,
                as_str=as_str,
                wrap_optional=wrap_optional,
                none_default=none_default,
                expected_type_hints=expected,
            ):
                # Create function to check
                if annot_unchanged == ():
                    if none_default:
                        def func(x=None): pass
                    else:
                        def func(x): pass
                elif none_default:
                    def func(x: annot = None): pass
                else:
                    def func(x: annot): pass
                type_hints = get_type_hints(func, globals(), locals(), include_extras=True)
                # Equality
                self.assertEqual(type_hints, expected)
                # Hash
                for k in type_hints.keys():
                    self.assertEqual(hash(type_hints[k]), hash(expected[k]))
                    # Test if UnionTypes are preserved
                    self.assertIs(type(type_hints[k]), type(expected[k]))
                # Repr
                with self.subTest("Check str and repr"):
                    if skip_reason == "UnionType not preserved in 3.10":
                        self.skipTest(skip_reason)
                    self.assertEqual(repr(type_hints), repr(expected))


class GetUtilitiesTestCase(TestCase):
    def test_get_origin(self):
        T = TypeVar('T')
        P = ParamSpec('P')
        Ts = TypeVarTuple('Ts')
        class C(Generic[T]): pass
        self.assertIs(get_origin(C[int]), C)
        self.assertIs(get_origin(C[T]), C)
        self.assertIs(get_origin(int), None)
        self.assertIs(get_origin(ClassVar[int]), ClassVar)
        self.assertIs(get_origin(Union[int, str]), Union)
        self.assertIs(get_origin(Literal[42, 43]), Literal)
        self.assertIs(get_origin(Final[List[int]]), Final)
        self.assertIs(get_origin(Generic), Generic)
        self.assertIs(get_origin(Generic[T]), Generic)
        self.assertIs(get_origin(List[Tuple[T, T]][int]), list)
        self.assertIs(get_origin(Annotated[T, 'thing']), Annotated)
        self.assertIs(get_origin(List), list)
        self.assertIs(get_origin(Tuple), tuple)
        self.assertIs(get_origin(Callable), collections.abc.Callable)
        self.assertIs(get_origin(list[int]), list)
        self.assertIs(get_origin(list), None)
        self.assertIs(get_origin(P.args), P)
        self.assertIs(get_origin(P.kwargs), P)
        self.assertIs(get_origin(Required[int]), Required)
        self.assertIs(get_origin(NotRequired[int]), NotRequired)
        self.assertIs(get_origin(Unpack[Ts]), Unpack)
        self.assertIs(get_origin(Unpack), None)

    def test_get_args(self):
        T = TypeVar('T')
        Ts = TypeVarTuple('Ts')
        class C(Generic[T]): pass
        self.assertEqual(get_args(C[int]), (int,))
        self.assertEqual(get_args(C[T]), (T,))
        self.assertEqual(get_args(int), ())
        self.assertEqual(get_args(ClassVar[int]), (int,))
        self.assertEqual(get_args(Union[int, str]), (int, str))
        self.assertEqual(get_args(Literal[42, 43]), (42, 43))
        self.assertEqual(get_args(Final[List[int]]), (List[int],))
        self.assertEqual(get_args(Union[int, Tuple[T, int]][str]),
                         (int, Tuple[str, int]))
        self.assertEqual(get_args(typing.Dict[int, Tuple[T, T]][Optional[int]]),
                         (int, Tuple[Optional[int], Optional[int]]))
        self.assertEqual(get_args(Callable[[], T][int]), ([], int))
        self.assertEqual(get_args(Callable[..., int]), (..., int))
        self.assertEqual(get_args(Union[int, Callable[[Tuple[T, ...]], str]]),
                         (int, Callable[[Tuple[T, ...]], str]))
        self.assertEqual(get_args(Tuple[int, ...]), (int, ...))
        if TYPING_3_11_0:
            self.assertEqual(get_args(Tuple[()]), ())
        else:
            self.assertEqual(get_args(Tuple[()]), ((),))
        self.assertEqual(get_args(Annotated[T, 'one', 2, ['three']]), (T, 'one', 2, ['three']))
        self.assertEqual(get_args(List), ())
        self.assertEqual(get_args(Tuple), ())
        self.assertEqual(get_args(Callable), ())
        self.assertEqual(get_args(list[int]), (int,))
        self.assertEqual(get_args(list), ())
        # Support Python versions with and without the fix for
        # https://bugs.python.org/issue42195
        # The first variant is for 3.9.2+, the second for 3.9.0 and 1
        self.assertIn(get_args(collections.abc.Callable[[int], str]),
                        (([int], str), ([[int]], str)))
        self.assertIn(get_args(collections.abc.Callable[[], str]),
                        (([], str), ([[]], str)))
        self.assertEqual(get_args(collections.abc.Callable[..., str]), (..., str))
        P = ParamSpec('P')
        # In 3.9 we use typing_extensions's hacky implementation
        # of ParamSpec, which gets incorrectly wrapped in a list
        self.assertIn(get_args(Callable[P, int]), [(P, int), ([P], int)])
        self.assertEqual(get_args(Required[int]), (int,))
        self.assertEqual(get_args(NotRequired[int]), (int,))
        self.assertEqual(get_args(Unpack[Ts]), (Ts,))
        self.assertEqual(get_args(Unpack), ())
        self.assertEqual(get_args(Callable[Concatenate[int, P], int]),
                         (Concatenate[int, P], int))
        self.assertEqual(get_args(Callable[Concatenate[int, ...], int]),
                        (Concatenate[int, ...], int))


class CollectionsAbcTests(BaseTestCase):

    def test_isinstance_collections(self):
        self.assertNotIsInstance(1, collections.abc.Mapping)
        self.assertNotIsInstance(1, collections.abc.Iterable)
        self.assertNotIsInstance(1, collections.abc.Container)
        self.assertNotIsInstance(1, collections.abc.Sized)
        with self.assertRaises(TypeError):
            isinstance(collections.deque(), typing_extensions.Deque[int])
        with self.assertRaises(TypeError):
            issubclass(collections.Counter, typing_extensions.Counter[str])

    def test_awaitable(self):
        async def foo() -> typing_extensions.Awaitable[int]:
            return await AwaitableWrapper(42)

        g = foo()
        self.assertIsInstance(g, typing_extensions.Awaitable)
        self.assertNotIsInstance(foo, typing_extensions.Awaitable)
        g.send(None)  # Run foo() till completion, to avoid warning.

    def test_coroutine(self):
        async def foo():
            return

        g = foo()
        self.assertIsInstance(g, typing_extensions.Coroutine)
        with self.assertRaises(TypeError):
            isinstance(g, typing_extensions.Coroutine[int])
        self.assertNotIsInstance(foo, typing_extensions.Coroutine)
        try:
            g.send(None)
        except StopIteration:
            pass

    def test_async_iterable(self):
        base_it: Iterator[int] = range(10)
        it = AsyncIteratorWrapper(base_it)
        self.assertIsInstance(it, typing_extensions.AsyncIterable)
        self.assertIsInstance(it, typing_extensions.AsyncIterable)
        self.assertNotIsInstance(42, typing_extensions.AsyncIterable)

    def test_async_iterator(self):
        base_it: Iterator[int] = range(10)
        it = AsyncIteratorWrapper(base_it)
        self.assertIsInstance(it, typing_extensions.AsyncIterator)
        self.assertNotIsInstance(42, typing_extensions.AsyncIterator)

    def test_deque(self):
        self.assertIsSubclass(collections.deque, typing_extensions.Deque)
        class MyDeque(typing_extensions.Deque[int]): ...
        self.assertIsInstance(MyDeque(), collections.deque)

    def test_counter(self):
        self.assertIsSubclass(collections.Counter, typing_extensions.Counter)

    def test_defaultdict_instantiation(self):
        self.assertIs(
            type(typing_extensions.DefaultDict()),
            collections.defaultdict)
        self.assertIs(
            type(typing_extensions.DefaultDict[KT, VT]()),
            collections.defaultdict)
        self.assertIs(
            type(typing_extensions.DefaultDict[str, int]()),
            collections.defaultdict)

    def test_defaultdict_subclass(self):

        class MyDefDict(typing_extensions.DefaultDict[str, int]):
            pass

        dd = MyDefDict()
        self.assertIsInstance(dd, MyDefDict)

        self.assertIsSubclass(MyDefDict, collections.defaultdict)
        self.assertNotIsSubclass(collections.defaultdict, MyDefDict)

    def test_ordereddict_instantiation(self):
        self.assertIs(
            type(typing_extensions.OrderedDict()),
            collections.OrderedDict)
        self.assertIs(
            type(typing_extensions.OrderedDict[KT, VT]()),
            collections.OrderedDict)
        self.assertIs(
            type(typing_extensions.OrderedDict[str, int]()),
            collections.OrderedDict)

    def test_ordereddict_subclass(self):

        class MyOrdDict(typing_extensions.OrderedDict[str, int]):
            pass

        od = MyOrdDict()
        self.assertIsInstance(od, MyOrdDict)

        self.assertIsSubclass(MyOrdDict, collections.OrderedDict)
        self.assertNotIsSubclass(collections.OrderedDict, MyOrdDict)

    def test_chainmap_instantiation(self):
        self.assertIs(type(typing_extensions.ChainMap()), collections.ChainMap)
        self.assertIs(type(typing_extensions.ChainMap[KT, VT]()), collections.ChainMap)
        self.assertIs(type(typing_extensions.ChainMap[str, int]()), collections.ChainMap)
        class CM(typing_extensions.ChainMap[KT, VT]): ...
        self.assertIs(type(CM[int, str]()), CM)

    def test_chainmap_subclass(self):

        class MyChainMap(typing_extensions.ChainMap[str, int]):
            pass

        cm = MyChainMap()
        self.assertIsInstance(cm, MyChainMap)

        self.assertIsSubclass(MyChainMap, collections.ChainMap)
        self.assertNotIsSubclass(collections.ChainMap, MyChainMap)

    def test_deque_instantiation(self):
        self.assertIs(type(typing_extensions.Deque()), collections.deque)
        self.assertIs(type(typing_extensions.Deque[T]()), collections.deque)
        self.assertIs(type(typing_extensions.Deque[int]()), collections.deque)
        class D(typing_extensions.Deque[T]): ...
        self.assertIs(type(D[int]()), D)

    def test_counter_instantiation(self):
        self.assertIs(type(typing_extensions.Counter()), collections.Counter)
        self.assertIs(type(typing_extensions.Counter[T]()), collections.Counter)
        self.assertIs(type(typing_extensions.Counter[int]()), collections.Counter)
        class C(typing_extensions.Counter[T]): ...
        self.assertIs(type(C[int]()), C)
        self.assertEqual(C.__bases__, (collections.Counter, typing.Generic))

    def test_counter_subclass_instantiation(self):

        class MyCounter(typing_extensions.Counter[int]):
            pass

        d = MyCounter()
        self.assertIsInstance(d, MyCounter)
        self.assertIsInstance(d, collections.Counter)
        self.assertIsInstance(d, typing_extensions.Counter)


# These are a separate TestCase class,
# as (unlike most collections.abc aliases in typing_extensions),
# these are reimplemented on Python <=3.12 so that we can provide
# default values for the second and third parameters
class GeneratorTests(BaseTestCase):

    def test_generator_basics(self):
        def foo():
            yield 42
        g = foo()

        self.assertIsInstance(g, typing_extensions.Generator)
        self.assertNotIsInstance(foo, typing_extensions.Generator)
        self.assertIsSubclass(type(g), typing_extensions.Generator)
        self.assertNotIsSubclass(type(foo), typing_extensions.Generator)

        parameterized = typing_extensions.Generator[int, str, None]
        with self.assertRaises(TypeError):
            isinstance(g, parameterized)
        with self.assertRaises(TypeError):
            issubclass(type(g), parameterized)

    def test_generator_default(self):
        g1 = typing_extensions.Generator[int]
        g2 = typing_extensions.Generator[int, None, None]
        self.assertEqual(get_args(g1), (int, type(None), type(None)))
        self.assertEqual(get_args(g1), get_args(g2))

        g3 = typing_extensions.Generator[int, float]
        g4 = typing_extensions.Generator[int, float, None]
        self.assertEqual(get_args(g3), (int, float, type(None)))
        self.assertEqual(get_args(g3), get_args(g4))

    def test_no_generator_instantiation(self):
        with self.assertRaises(TypeError):
            typing_extensions.Generator()
        with self.assertRaises(TypeError):
            typing_extensions.Generator[T, T, T]()
        with self.assertRaises(TypeError):
            typing_extensions.Generator[int, int, int]()

    def test_subclassing_generator(self):
        class G(typing_extensions.Generator[int, int, None]):
            def send(self, value):
                pass
            def throw(self, typ, val=None, tb=None):
                pass

        def g(): yield 0

        self.assertIsSubclass(G, typing_extensions.Generator)
        self.assertIsSubclass(G, typing_extensions.Iterable)
        self.assertIsSubclass(G, collections.abc.Generator)
        self.assertIsSubclass(G, collections.abc.Iterable)
        self.assertNotIsSubclass(type(g), G)

        instance = G()
        self.assertIsInstance(instance, typing_extensions.Generator)
        self.assertIsInstance(instance, typing_extensions.Iterable)
        self.assertIsInstance(instance, collections.abc.Generator)
        self.assertIsInstance(instance, collections.abc.Iterable)
        self.assertNotIsInstance(type(g), G)
        self.assertNotIsInstance(g, G)

    def test_async_generator_basics(self):
        async def f():
            yield 42
        g = f()

        self.assertIsInstance(g, typing_extensions.AsyncGenerator)
        self.assertIsSubclass(type(g), typing_extensions.AsyncGenerator)
        self.assertNotIsInstance(f, typing_extensions.AsyncGenerator)
        self.assertNotIsSubclass(type(f), typing_extensions.AsyncGenerator)

        parameterized = typing_extensions.AsyncGenerator[int, str]
        with self.assertRaises(TypeError):
            isinstance(g, parameterized)
        with self.assertRaises(TypeError):
            issubclass(type(g), parameterized)

    def test_async_generator_default(self):
        ag1 = typing_extensions.AsyncGenerator[int]
        ag2 = typing_extensions.AsyncGenerator[int, None]
        self.assertEqual(get_args(ag1), (int, type(None)))
        self.assertEqual(get_args(ag1), get_args(ag2))

    def test_no_async_generator_instantiation(self):
        with self.assertRaises(TypeError):
            typing_extensions.AsyncGenerator()
        with self.assertRaises(TypeError):
            typing_extensions.AsyncGenerator[T, T]()
        with self.assertRaises(TypeError):
            typing_extensions.AsyncGenerator[int, int]()

    def test_subclassing_async_generator(self):
        class G(typing_extensions.AsyncGenerator[int, int]):
            def asend(self, value):
                pass
            def athrow(self, typ, val=None, tb=None):
                pass

        async def g(): yield 0

        self.assertIsSubclass(G, typing_extensions.AsyncGenerator)
        self.assertIsSubclass(G, typing_extensions.AsyncIterable)
        self.assertIsSubclass(G, collections.abc.AsyncGenerator)
        self.assertIsSubclass(G, collections.abc.AsyncIterable)
        self.assertNotIsSubclass(type(g), G)

        instance = G()
        self.assertIsInstance(instance, typing_extensions.AsyncGenerator)
        self.assertIsInstance(instance, typing_extensions.AsyncIterable)
        self.assertIsInstance(instance, collections.abc.AsyncGenerator)
        self.assertIsInstance(instance, collections.abc.AsyncIterable)
        self.assertNotIsInstance(type(g), G)
        self.assertNotIsInstance(g, G)

    def test_subclassing_subclasshook(self):

        class Base(typing_extensions.Generator):
            @classmethod
            def __subclasshook__(cls, other):
                if other.__name__ == 'Foo':
                    return True
                else:
                    return False

        class C(Base): ...
        class Foo: ...
        class Bar: ...
        self.assertIsSubclass(Foo, Base)
        self.assertIsSubclass(Foo, C)
        self.assertNotIsSubclass(Bar, C)

    def test_subclassing_register(self):

        class A(typing_extensions.Generator): ...
        class B(A): ...

        class C: ...
        A.register(C)
        self.assertIsSubclass(C, A)
        self.assertNotIsSubclass(C, B)

        class D: ...
        B.register(D)
        self.assertIsSubclass(D, A)
        self.assertIsSubclass(D, B)

        class M: ...
        collections.abc.Generator.register(M)
        self.assertIsSubclass(M, typing_extensions.Generator)

    def test_collections_as_base(self):

        class M(collections.abc.Generator): ...
        self.assertIsSubclass(M, typing_extensions.Generator)
        self.assertIsSubclass(M, typing_extensions.Iterable)

        class S(collections.abc.AsyncGenerator): ...
        self.assertIsSubclass(S, typing_extensions.AsyncGenerator)
        self.assertIsSubclass(S, typing_extensions.AsyncIterator)

        class A(collections.abc.Generator, metaclass=abc.ABCMeta): ...
        class B: ...
        A.register(B)
        self.assertIsSubclass(B, typing_extensions.Generator)

    @skipIf(sys.version_info < (3, 10), "PEP 604 has yet to be")
    def test_or_and_ror(self):
        self.assertEqual(
            typing_extensions.Generator | typing_extensions.AsyncGenerator,
            Union[typing_extensions.Generator, typing_extensions.AsyncGenerator]
        )
        self.assertEqual(
            typing_extensions.Generator | typing.Deque,
            Union[typing_extensions.Generator, typing.Deque]
        )


class OtherABCTests(BaseTestCase):

    def test_contextmanager(self):
        @contextlib.contextmanager
        def manager():
            yield 42

        cm = manager()
        self.assertIsInstance(cm, typing_extensions.ContextManager)
        self.assertNotIsInstance(42, typing_extensions.ContextManager)

    def test_contextmanager_type_params(self):
        cm1 = typing_extensions.ContextManager[int]
        self.assertEqual(get_args(cm1), (int, typing.Optional[bool]))
        cm2 = typing_extensions.ContextManager[int, None]
        self.assertEqual(get_args(cm2), (int, NoneType))

    def test_async_contextmanager(self):
        class NotACM:
            pass
        self.assertIsInstance(ACM(), typing_extensions.AsyncContextManager)
        self.assertNotIsInstance(NotACM(), typing_extensions.AsyncContextManager)
        @contextlib.contextmanager
        def manager():
            yield 42

        cm = manager()
        self.assertNotIsInstance(cm, typing_extensions.AsyncContextManager)
        self.assertEqual(
            typing_extensions.AsyncContextManager[int].__args__,
            (int, typing.Optional[bool])
        )
        with self.assertRaises(TypeError):
            isinstance(42, typing_extensions.AsyncContextManager[int])
        with self.assertRaises(TypeError):
            typing_extensions.AsyncContextManager[int, str, float]

    def test_asynccontextmanager_type_params(self):
        cm1 = typing_extensions.AsyncContextManager[int]
        self.assertEqual(get_args(cm1), (int, typing.Optional[bool]))
        cm2 = typing_extensions.AsyncContextManager[int, None]
        self.assertEqual(get_args(cm2), (int, NoneType))


class TypeTests(BaseTestCase):

    def test_type_basic(self):

        class User: pass
        class BasicUser(User): pass
        class ProUser(User): pass

        def new_user(user_class: Type[User]) -> User:
            return user_class()

        new_user(BasicUser)

    def test_type_typevar(self):

        class User: pass
        class BasicUser(User): pass
        class ProUser(User): pass

        U = TypeVar('U', bound=User)

        def new_user(user_class: Type[U]) -> U:
            return user_class()

        new_user(BasicUser)

    def test_type_optional(self):
        A = Optional[Type[BaseException]]

        def foo(a: A) -> Optional[BaseException]:
            if a is None:
                return None
            else:
                return a()

        assert isinstance(foo(KeyboardInterrupt), KeyboardInterrupt)
        assert foo(None) is None


class NewTypeTests(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        global UserId
        UserId = NewType('UserId', int)
        cls.UserName = NewType(cls.__qualname__ + '.UserName', str)

    @classmethod
    def tearDownClass(cls):
        global UserId
        del UserId
        del cls.UserName

    def test_basic(self):
        self.assertIsInstance(UserId(5), int)
        self.assertIsInstance(self.UserName('Joe'), str)
        self.assertEqual(UserId(5) + 1, 6)

    def test_errors(self):
        with self.assertRaises(TypeError):
            issubclass(UserId, int)
        with self.assertRaises(TypeError):
            class D(UserId):
                pass

    @skipUnless(TYPING_3_10_0, "PEP 604 has yet to be")
    def test_or(self):
        for cls in (int, self.UserName):
            with self.subTest(cls=cls):
                self.assertEqual(UserId | cls, Union[UserId, cls])
                self.assertEqual(cls | UserId, Union[cls, UserId])

                self.assertEqual(get_args(UserId | cls), (UserId, cls))
                self.assertEqual(get_args(cls | UserId), (cls, UserId))

    def test_special_attrs(self):
        self.assertEqual(UserId.__name__, 'UserId')
        self.assertEqual(UserId.__qualname__, 'UserId')
        self.assertEqual(UserId.__module__, __name__)
        self.assertEqual(UserId.__supertype__, int)

        UserName = self.UserName
        self.assertEqual(UserName.__name__, 'UserName')
        self.assertEqual(UserName.__qualname__,
                         self.__class__.__qualname__ + '.UserName')
        self.assertEqual(UserName.__module__, __name__)
        self.assertEqual(UserName.__supertype__, str)

    def test_repr(self):
        self.assertEqual(repr(UserId), f'{__name__}.UserId')
        self.assertEqual(repr(self.UserName),
                         f'{__name__}.{self.__class__.__qualname__}.UserName')

    def test_pickle(self):
        UserAge = NewType('UserAge', float)
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                pickled = pickle.dumps(UserId, proto)
                loaded = pickle.loads(pickled)
                self.assertIs(loaded, UserId)

                pickled = pickle.dumps(self.UserName, proto)
                loaded = pickle.loads(pickled)
                self.assertIs(loaded, self.UserName)

                with self.assertRaises(pickle.PicklingError):
                    pickle.dumps(UserAge, proto)

    def test_missing__name__(self):
        code = ("import typing_extensions\n"
                "NT = typing_extensions.NewType('NT', int)\n"
                )
        exec(code, {})

    def test_error_message_when_subclassing(self):
        with self.assertRaisesRegex(
            TypeError,
            re.escape(
                "Cannot subclass an instance of NewType. Perhaps you were looking for: "
                "`ProUserId = NewType('ProUserId', UserId)`"
            )
        ):
            class ProUserId(UserId):
                ...


class Coordinate(Protocol):
    x: int
    y: int

@runtime_checkable
class Point(Coordinate, Protocol):
    label: str

class MyPoint:
    x: int
    y: int
    label: str

class XAxis(Protocol):
    x: int

class YAxis(Protocol):
    y: int

@runtime_checkable
class Position(XAxis, YAxis, Protocol):
    pass

@runtime_checkable
class Proto(Protocol):
    attr: int

    def meth(self, arg: str) -> int:
        ...

class Concrete(Proto):
    pass

class Other:
    attr: int = 1

    def meth(self, arg: str) -> int:
        if arg == 'this':
            return 1
        return 0

class NT(NamedTuple):
    x: int
    y: int


skip_if_py312b1 = skipIf(
    sys.version_info == (3, 12, 0, 'beta', 1),
    "CPython had bugs in 3.12.0b1"
)


class Point2DGeneric(Generic[T], TypedDict):
    a: T
    b: T


class Bar(Foo):
    b: int


class BarGeneric(FooGeneric[T], total=False):
    b: int


class TypedDictTests(BaseTestCase):
    def test_basics_functional_syntax(self):
        Emp = TypedDict('Emp', {'name': str, 'id': int})
        self.assertIsSubclass(Emp, dict)
        self.assertIsSubclass(Emp, typing.MutableMapping)
        self.assertNotIsSubclass(Emp, collections.abc.Sequence)
        jim = Emp(name='Jim', id=1)
        self.assertIs(type(jim), dict)
        self.assertEqual(jim['name'], 'Jim')
        self.assertEqual(jim['id'], 1)
        self.assertEqual(Emp.__name__, 'Emp')
        self.assertEqual(Emp.__module__, __name__)
        self.assertEqual(Emp.__bases__, (dict,))
        self.assertEqual(Emp.__annotations__, {'name': str, 'id': int})
        self.assertEqual(Emp.__total__, True)

    def test_allowed_as_type_argument(self):
        # https://github.com/python/typing_extensions/issues/613
        obj = typing.Type[typing_extensions.TypedDict]
        self.assertIs(typing_extensions.get_origin(obj), type)
        self.assertEqual(typing_extensions.get_args(obj), (typing_extensions.TypedDict,))

    @skipIf(sys.version_info < (3, 13), "Change in behavior in 3.13")
    def test_keywords_syntax_raises_on_3_13(self):
        with self.assertRaises(TypeError), self.assertWarns(DeprecationWarning):
            TypedDict('Emp', name=str, id=int)

    @skipIf(sys.version_info >= (3, 13), "3.13 removes support for kwargs")
    def test_basics_keywords_syntax(self):
        with self.assertWarns(DeprecationWarning):
            Emp = TypedDict('Emp', name=str, id=int)
        self.assertIsSubclass(Emp, dict)
        self.assertIsSubclass(Emp, typing.MutableMapping)
        self.assertNotIsSubclass(Emp, collections.abc.Sequence)
        jim = Emp(name='Jim', id=1)
        self.assertIs(type(jim), dict)
        self.assertEqual(jim['name'], 'Jim')
        self.assertEqual(jim['id'], 1)
        self.assertEqual(Emp.__name__, 'Emp')
        self.assertEqual(Emp.__module__, __name__)
        self.assertEqual(Emp.__bases__, (dict,))
        self.assertEqual(Emp.__annotations__, {'name': str, 'id': int})
        self.assertEqual(Emp.__total__, True)

    @skipIf(sys.version_info >= (3, 13), "3.13 removes support for kwargs")
    def test_typeddict_special_keyword_names(self):
        with self.assertWarns(DeprecationWarning):
            TD = TypedDict("TD", cls=type, self=object, typename=str, _typename=int,
                           fields=list, _fields=dict,
                           closed=bool, extra_items=bool)
        self.assertEqual(TD.__name__, 'TD')
        self.assertEqual(TD.__annotations__, {'cls': type, 'self': object, 'typename': str,
                                              '_typename': int, 'fields': list, '_fields': dict,
                                              'closed': bool, 'extra_items': bool})
        self.assertIsNone(TD.__closed__)
        self.assertIs(TD.__extra_items__, NoExtraItems)
        a = TD(cls=str, self=42, typename='foo', _typename=53,
               fields=[('bar', tuple)], _fields={'baz', set},
               closed=None, extra_items="tea pot")
        self.assertEqual(a['cls'], str)
        self.assertEqual(a['self'], 42)
        self.assertEqual(a['typename'], 'foo')
        self.assertEqual(a['_typename'], 53)
        self.assertEqual(a['fields'], [('bar', tuple)])
        self.assertEqual(a['_fields'], {'baz', set})
        self.assertIsNone(a['closed'])
        self.assertEqual(a['extra_items'], "tea pot")

    def test_typeddict_create_errors(self):
        with self.assertRaises(TypeError):
            TypedDict.__new__()
        with self.assertRaises(TypeError):
            TypedDict()
        with self.assertRaises(TypeError):
            TypedDict('Emp', [('name', str)], None)

    def test_typeddict_errors(self):
        Emp = TypedDict('Emp', {'name': str, 'id': int})
        self.assertEqual(TypedDict.__module__, 'typing_extensions')
        jim = Emp(name='Jim', id=1)
        with self.assertRaises(TypeError):
            isinstance({}, Emp)
        with self.assertRaises(TypeError):
            isinstance(jim, Emp)
        with self.assertRaises(TypeError):
            issubclass(dict, Emp)

        if not TYPING_3_11_0:
            with self.assertRaises(TypeError), self.assertWarns(DeprecationWarning):
                TypedDict('Hi', x=1)
            with self.assertRaises(TypeError):
                TypedDict('Hi', [('x', int), ('y', 1)])
        with self.assertRaises(TypeError):
            TypedDict('Hi', [('x', int)], y=int)

    def test_py36_class_syntax_usage(self):
        self.assertEqual(LabelPoint2D.__name__, 'LabelPoint2D')
        self.assertEqual(LabelPoint2D.__module__, __name__)
        self.assertEqual(LabelPoint2D.__annotations__, {'x': int, 'y': int, 'label': str})
        self.assertEqual(LabelPoint2D.__bases__, (dict,))
        self.assertEqual(LabelPoint2D.__total__, True)
        self.assertNotIsSubclass(LabelPoint2D, typing.Sequence)
        not_origin = Point2D(x=0, y=1)
        self.assertEqual(not_origin['x'], 0)
        self.assertEqual(not_origin['y'], 1)
        other = LabelPoint2D(x=0, y=1, label='hi')
        self.assertEqual(other['label'], 'hi')

    def test_pickle(self):
        global EmpD  # pickle wants to reference the class by name
        EmpD = TypedDict('EmpD', {'name': str, 'id': int})
        jane = EmpD({'name': 'jane', 'id': 37})
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            z = pickle.dumps(jane, proto)
            jane2 = pickle.loads(z)
            self.assertEqual(jane2, jane)
            self.assertEqual(jane2, {'name': 'jane', 'id': 37})
            ZZ = pickle.dumps(EmpD, proto)
            EmpDnew = pickle.loads(ZZ)
            self.assertEqual(EmpDnew({'name': 'jane', 'id': 37}), jane)

    def test_pickle_generic(self):
        point = Point2DGeneric(a=5.0, b=3.0)
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            z = pickle.dumps(point, proto)
            point2 = pickle.loads(z)
            self.assertEqual(point2, point)
            self.assertEqual(point2, {'a': 5.0, 'b': 3.0})
            ZZ = pickle.dumps(Point2DGeneric, proto)
            Point2DGenericNew = pickle.loads(ZZ)
            self.assertEqual(Point2DGenericNew({'a': 5.0, 'b': 3.0}), point)

    def test_optional(self):
        EmpD = TypedDict('EmpD', {'name': str, 'id': int})

        self.assertEqual(typing.Optional[EmpD], typing.Union[None, EmpD])
        self.assertNotEqual(typing.List[EmpD], typing.Tuple[EmpD])

    def test_total(self):
        D = TypedDict('D', {'x': int}, total=False)
        self.assertEqual(D(), {})
        self.assertEqual(D(x=1), {'x': 1})
        self.assertEqual(D.__total__, False)
        self.assertEqual(D.__required_keys__, frozenset())
        self.assertEqual(D.__optional_keys__, {'x'})

        self.assertEqual(Options(), {})
        self.assertEqual(Options(log_level=2), {'log_level': 2})
        self.assertEqual(Options.__total__, False)
        self.assertEqual(Options.__required_keys__, frozenset())
        self.assertEqual(Options.__optional_keys__, {'log_level', 'log_path'})

    def test_total_inherits_non_total(self):
        class TD1(TypedDict, total=False):
            a: int

        self.assertIs(TD1.__total__, False)

        class TD2(TD1):
            b: str

        self.assertIs(TD2.__total__, True)

    def test_total_with_assigned_value(self):
        class TD(TypedDict):
            __total__ = "some_value"

        self.assertIs(TD.__total__, True)

        class TD2(TypedDict, total=True):
            __total__ = "some_value"

        self.assertIs(TD2.__total__, True)

        class TD3(TypedDict, total=False):
            __total__ = "some value"

        self.assertIs(TD3.__total__, False)

        TD4 = TypedDict('TD4', {'__total__': "some_value"})  # noqa: F821
        self.assertIs(TD4.__total__, True)


    def test_optional_keys(self):
        class Point2Dor3D(Point2D, total=False):
            z: int

        assert Point2Dor3D.__required_keys__ == frozenset(['x', 'y'])
        assert Point2Dor3D.__optional_keys__ == frozenset(['z'])

    def test_keys_inheritance(self):
        class BaseAnimal(TypedDict):
            name: str

        class Animal(BaseAnimal, total=False):
            voice: str
            tail: bool

        class Cat(Animal):
            fur_color: str

        assert BaseAnimal.__required_keys__ == frozenset(['name'])
        assert BaseAnimal.__optional_keys__ == frozenset([])
        assert BaseAnimal.__annotations__ == {'name': str}

        assert Animal.__required_keys__ == frozenset(['name'])
        assert Animal.__optional_keys__ == frozenset(['tail', 'voice'])
        assert Animal.__annotations__ == {
            'name': str,
            'tail': bool,
            'voice': str,
        }

        assert Cat.__required_keys__ == frozenset(['name', 'fur_color'])
        assert Cat.__optional_keys__ == frozenset(['tail', 'voice'])
        assert Cat.__annotations__ == {
            'fur_color': str,
            'name': str,
            'tail': bool,
            'voice': str,
        }

    @skipIf(sys.version_info == (3, 14, 0, "beta", 1), "Broken on beta 1, fixed in beta 2")
    def test_inheritance_pep563(self):
        def _make_td(future, class_name, annos, base, extra_names=None):
            lines = []
            if future:
                lines.append('from __future__ import annotations')
            lines.append('from typing import TypedDict')
            lines.append(f'class {class_name}({base}):')
            for name, anno in annos.items():
                lines.append(f'    {name}: {anno}')
            code = '\n'.join(lines)
            ns = {**extra_names} if extra_names else {}
            exec(code, ns)
            return ns[class_name]

        for base_future in (True, False):
            for child_future in (True, False):
                with self.subTest(base_future=base_future, child_future=child_future):
                    base = _make_td(
                        base_future, "Base", {"base": "int"}, "TypedDict"
                    )
                    if sys.version_info >= (3, 14):
                        self.assertIsNotNone(base.__annotate__)
                    child = _make_td(
                        child_future, "Child", {"child": "int"}, "Base", {"Base": base}
                    )
                    base_anno = typing.ForwardRef("int", module="builtins") if base_future else int
                    child_anno = typing.ForwardRef("int", module="builtins") if child_future else int
                    self.assertEqual(base.__annotations__, {'base': base_anno})
                    self.assertEqual(
                        child.__annotations__, {'child': child_anno, 'base': base_anno}
                    )

    def test_required_notrequired_keys(self):
        self.assertEqual(NontotalMovie.__required_keys__,
                         frozenset({"title"}))
        self.assertEqual(NontotalMovie.__optional_keys__,
                         frozenset({"year"}))

        self.assertEqual(TotalMovie.__required_keys__,
                         frozenset({"title"}))
        self.assertEqual(TotalMovie.__optional_keys__,
                         frozenset({"year"}))

        self.assertEqual(VeryAnnotated.__required_keys__,
                         frozenset())
        self.assertEqual(VeryAnnotated.__optional_keys__,
                         frozenset({"a"}))

        self.assertEqual(AnnotatedMovie.__required_keys__,
                         frozenset({"title"}))
        self.assertEqual(AnnotatedMovie.__optional_keys__,
                         frozenset({"year"}))

        self.assertEqual(WeirdlyQuotedMovie.__required_keys__,
                         frozenset({"title"}))
        self.assertEqual(WeirdlyQuotedMovie.__optional_keys__,
                         frozenset({"year"}))

        self.assertEqual(ChildTotalMovie.__required_keys__,
                         frozenset({"title"}))
        self.assertEqual(ChildTotalMovie.__optional_keys__,
                         frozenset({"year"}))

        self.assertEqual(ChildDeeplyAnnotatedMovie.__required_keys__,
                         frozenset({"title"}))
        self.assertEqual(ChildDeeplyAnnotatedMovie.__optional_keys__,
                         frozenset({"year"}))

    def test_multiple_inheritance(self):
        class One(TypedDict):
            one: int
        class Two(TypedDict):
            two: str
        class Untotal(TypedDict, total=False):
            untotal: str
        Inline = TypedDict('Inline', {'inline': bool})
        class Regular:
            pass

        class Child(One, Two):
            child: bool
        self.assertEqual(
            Child.__required_keys__,
            frozenset(['one', 'two', 'child']),
        )
        self.assertEqual(
            Child.__optional_keys__,
            frozenset([]),
        )
        self.assertEqual(
            Child.__annotations__,
            {'one': int, 'two': str, 'child': bool},
        )

        class ChildWithOptional(One, Untotal):
            child: bool
        self.assertEqual(
            ChildWithOptional.__required_keys__,
            frozenset(['one', 'child']),
        )
        self.assertEqual(
            ChildWithOptional.__optional_keys__,
            frozenset(['untotal']),
        )
        self.assertEqual(
            ChildWithOptional.__annotations__,
            {'one': int, 'untotal': str, 'child': bool},
        )

        class ChildWithTotalFalse(One, Untotal, total=False):
            child: bool
        self.assertEqual(
            ChildWithTotalFalse.__required_keys__,
            frozenset(['one']),
        )
        self.assertEqual(
            ChildWithTotalFalse.__optional_keys__,
            frozenset(['untotal', 'child']),
        )
        self.assertEqual(
            ChildWithTotalFalse.__annotations__,
            {'one': int, 'untotal': str, 'child': bool},
        )

        class ChildWithInlineAndOptional(Untotal, Inline):
            child: bool
        self.assertEqual(
            ChildWithInlineAndOptional.__required_keys__,
            frozenset(['inline', 'child']),
        )
        self.assertEqual(
            ChildWithInlineAndOptional.__optional_keys__,
            frozenset(['untotal']),
        )
        self.assertEqual(
            ChildWithInlineAndOptional.__annotations__,
            {'inline': bool, 'untotal': str, 'child': bool},
        )

        wrong_bases = [
            (One, Regular),
            (Regular, One),
            (One, Two, Regular),
            (Inline, Regular),
            (Untotal, Regular),
        ]
        for bases in wrong_bases:
            with self.subTest(bases=bases):
                with self.assertRaisesRegex(
                    TypeError,
                    'cannot inherit from both a TypedDict type and a non-TypedDict',
                ):
                    class Wrong(*bases):
                        pass

    def test_closed_values(self):
        class Implicit(TypedDict): ...
        class ExplicitTrue(TypedDict, closed=True): ...
        class ExplicitFalse(TypedDict, closed=False): ...

        self.assertIsNone(Implicit.__closed__)
        self.assertIs(ExplicitTrue.__closed__, True)
        self.assertIs(ExplicitFalse.__closed__, False)


    @skipIf(TYPING_3_14_0, "only supported on older versions")
    def test_closed_typeddict_compat(self):
        class Closed(TypedDict, closed=True):
            __extra_items__: None

        class Unclosed(TypedDict, closed=False):
            ...

        class ChildUnclosed(Closed, Unclosed):
            ...

        self.assertIsNone(ChildUnclosed.__closed__)
        self.assertEqual(ChildUnclosed.__extra_items__, NoExtraItems)

        class ChildClosed(Unclosed, Closed):
            ...

        self.assertIsNone(ChildClosed.__closed__)
        self.assertEqual(ChildClosed.__extra_items__, NoExtraItems)

    def test_extra_items_class_arg(self):
        class TD(TypedDict, extra_items=int):
            a: str

        self.assertIs(TD.__extra_items__, int)
        self.assertEqual(TD.__annotations__, {'a': str})
        self.assertEqual(TD.__required_keys__, frozenset({'a'}))
        self.assertEqual(TD.__optional_keys__, frozenset())

        class NoExtra(TypedDict):
            a: str

        self.assertIs(NoExtra.__extra_items__, NoExtraItems)
        self.assertEqual(NoExtra.__annotations__, {'a': str})
        self.assertEqual(NoExtra.__required_keys__, frozenset({'a'}))
        self.assertEqual(NoExtra.__optional_keys__, frozenset())

    def test_is_typeddict(self):
        self.assertIs(is_typeddict(Point2D), True)
        self.assertIs(is_typeddict(Point2Dor3D), True)
        self.assertIs(is_typeddict(Union[str, int]), False)
        # classes, not instances
        self.assertIs(is_typeddict(Point2D()), False)
        call_based = TypedDict('call_based', {'a': int})
        self.assertIs(is_typeddict(call_based), True)
        self.assertIs(is_typeddict(call_based()), False)

        T = TypeVar("T")
        class BarGeneric(TypedDict, Generic[T]):
            a: T
        self.assertIs(is_typeddict(BarGeneric), True)
        self.assertIs(is_typeddict(BarGeneric[int]), False)
        self.assertIs(is_typeddict(BarGeneric()), False)

        if hasattr(typing, "TypeAliasType"):
            ns = {"TypedDict": TypedDict}
            exec("""if True:
                class NewGeneric[T](TypedDict):
                    a: T
            """, ns)
            NewGeneric = ns["NewGeneric"]
            self.assertIs(is_typeddict(NewGeneric), True)
            self.assertIs(is_typeddict(NewGeneric[int]), False)
            self.assertIs(is_typeddict(NewGeneric()), False)

        # The TypedDict constructor is not itself a TypedDict
        self.assertIs(is_typeddict(TypedDict), False)
        if hasattr(typing, "TypedDict"):
            self.assertIs(is_typeddict(typing.TypedDict), False)

    def test_is_typeddict_against_typeddict_from_typing(self):
        Point = typing.TypedDict('Point', {'x': int, 'y': int})

        class PointDict2D(typing.TypedDict):
            x: int
            y: int

        class PointDict3D(PointDict2D, total=False):
            z: int

        assert is_typeddict(Point) is True
        assert is_typeddict(PointDict2D) is True
        assert is_typeddict(PointDict3D) is True

    @skipUnless(HAS_FORWARD_MODULE, "ForwardRef.__forward_module__ was added in 3.9.7")
    def test_get_type_hints_cross_module_subclass(self):
        self.assertNotIn("_DoNotImport", globals())
        self.assertEqual(
            {k: v.__name__ for k, v in get_type_hints(Bar).items()},
            {'a': "_DoNotImport", 'b': "int"}
        )

    def test_get_type_hints_generic(self):
        self.assertEqual(
            get_type_hints(BarGeneric),
            {'a': typing.Optional[T], 'b': int}
        )

        class FooBarGeneric(BarGeneric[int]):
            c: str

        self.assertEqual(
            get_type_hints(FooBarGeneric),
            {'a': typing.Optional[T], 'b': int, 'c': str}
        )

    @skipUnless(TYPING_3_12_0, "PEP 695 required")
    def test_pep695_generic_typeddict(self):
        ns = {"TypedDict": TypedDict}
        exec("""if True:
            class A[T](TypedDict):
                a: T
            """, ns)
        A = ns["A"]
        T, = A.__type_params__
        self.assertIsInstance(T, TypeVar)
        self.assertEqual(T.__name__, 'T')
        self.assertEqual(A.__bases__, (Generic, dict))
        self.assertEqual(A.__orig_bases__, (TypedDict, Generic[T]))
        self.assertEqual(A.__mro__, (A, Generic, dict, object))
        self.assertEqual(A.__parameters__, (T,))
        self.assertEqual(A[str].__parameters__, ())
        self.assertEqual(A[str].__args__, (str,))

    def test_generic_inheritance(self):
        class A(TypedDict, Generic[T]):
            a: T

        self.assertEqual(A.__bases__, (Generic, dict))
        self.assertEqual(A.__orig_bases__, (TypedDict, Generic[T]))
        self.assertEqual(A.__mro__, (A, Generic, dict, object))
        self.assertEqual(A.__parameters__, (T,))
        self.assertEqual(A[str].__parameters__, ())
        self.assertEqual(A[str].__args__, (str,))

        class A2(Generic[T], TypedDict):
            a: T

        self.assertEqual(A2.__bases__, (Generic, dict))
        self.assertEqual(A2.__orig_bases__, (Generic[T], TypedDict))
        self.assertEqual(A2.__mro__, (A2, Generic, dict, object))
        self.assertEqual(A2.__parameters__, (T,))
        self.assertEqual(A2[str].__parameters__, ())
        self.assertEqual(A2[str].__args__, (str,))

        class B(A[KT], total=False):
            b: KT

        self.assertEqual(B.__bases__, (Generic, dict))
        self.assertEqual(B.__orig_bases__, (A[KT],))
        self.assertEqual(B.__mro__, (B, Generic, dict, object))
        self.assertEqual(B.__parameters__, (KT,))
        self.assertEqual(B.__total__, False)
        self.assertEqual(B.__optional_keys__, frozenset(['b']))
        self.assertEqual(B.__required_keys__, frozenset(['a']))

        self.assertEqual(B[str].__parameters__, ())
        self.assertEqual(B[str].__args__, (str,))
        self.assertEqual(B[str].__origin__, B)

        class C(B[int]):
            c: int

        self.assertEqual(C.__bases__, (Generic, dict))
        self.assertEqual(C.__orig_bases__, (B[int],))
        self.assertEqual(C.__mro__, (C, Generic, dict, object))
        self.assertEqual(C.__parameters__, ())
        self.assertEqual(C.__total__, True)
        self.assertEqual(C.__optional_keys__, frozenset(['b']))
        self.assertEqual(C.__required_keys__, frozenset(['a', 'c']))
        assert C.__annotations__ == {
            'a': T,
            'b': KT,
            'c': int,
        }
        with self.assertRaises(TypeError):
            C[str]

        class Point3D(Point2DGeneric[T], Generic[T, KT]):
            c: KT

        self.assertEqual(Point3D.__bases__, (Generic, dict))
        self.assertEqual(Point3D.__orig_bases__, (Point2DGeneric[T], Generic[T, KT]))
        self.assertEqual(Point3D.__mro__, (Point3D, Generic, dict, object))
        self.assertEqual(Point3D.__parameters__, (T, KT))
        self.assertEqual(Point3D.__total__, True)
        self.assertEqual(Point3D.__optional_keys__, frozenset())
        self.assertEqual(Point3D.__required_keys__, frozenset(['a', 'b', 'c']))
        self.assertEqual(Point3D.__annotations__, {
            'a': T,
            'b': T,
            'c': KT,
        })
        self.assertEqual(Point3D[int, str].__origin__, Point3D)

        with self.assertRaises(TypeError):
            Point3D[int]

        with self.assertRaises(TypeError):
            class Point3D(Point2DGeneric[T], Generic[KT]):
                c: KT

    def test_implicit_any_inheritance(self):
        class A(TypedDict, Generic[T]):
            a: T

        class B(A[KT], total=False):
            b: KT

        class WithImplicitAny(B):
            c: int

        self.assertEqual(WithImplicitAny.__bases__, (Generic, dict,))
        self.assertEqual(WithImplicitAny.__mro__, (WithImplicitAny, Generic, dict, object))
        # Consistent with GenericTests.test_implicit_any
        self.assertEqual(WithImplicitAny.__parameters__, ())
        self.assertEqual(WithImplicitAny.__total__, True)
        self.assertEqual(WithImplicitAny.__optional_keys__, frozenset(['b']))
        self.assertEqual(WithImplicitAny.__required_keys__, frozenset(['a', 'c']))
        assert WithImplicitAny.__annotations__ == {
            'a': T,
            'b': KT,
            'c': int,
        }
        with self.assertRaises(TypeError):
            WithImplicitAny[str]

    def test_non_generic_subscript(self):
        # For backward compatibility, subscription works
        # on arbitrary TypedDict types.
        class TD(TypedDict):
            a: T
        A = TD[int]
        self.assertEqual(A.__origin__, TD)
        self.assertEqual(A.__parameters__, ())
        self.assertEqual(A.__args__, (int,))
        a = A(a=1)
        self.assertIs(type(a), dict)
        self.assertEqual(a, {'a': 1})

    def test_orig_bases(self):
        T = TypeVar('T')

        class Parent(TypedDict):
            pass

        class Child(Parent):
            pass

        class OtherChild(Parent):
            pass

        class MixedChild(Child, OtherChild, Parent):
            pass

        class GenericParent(TypedDict, Generic[T]):
            pass

        class GenericChild(GenericParent[int]):
            pass

        class OtherGenericChild(GenericParent[str]):
            pass

        class MixedGenericChild(GenericChild, OtherGenericChild, GenericParent[float]):
            pass

        class MultipleGenericBases(GenericParent[int], GenericParent[float]):
            pass

        CallTypedDict = TypedDict('CallTypedDict', {})

        self.assertEqual(Parent.__orig_bases__, (TypedDict,))
        self.assertEqual(Child.__orig_bases__, (Parent,))
        self.assertEqual(OtherChild.__orig_bases__, (Parent,))
        self.assertEqual(MixedChild.__orig_bases__, (Child, OtherChild, Parent,))
        self.assertEqual(GenericParent.__orig_bases__, (TypedDict, Generic[T]))
        self.assertEqual(GenericChild.__orig_bases__, (GenericParent[int],))
        self.assertEqual(OtherGenericChild.__orig_bases__, (GenericParent[str],))
        self.assertEqual(MixedGenericChild.__orig_bases__, (GenericChild, OtherGenericChild, GenericParent[float]))
        self.assertEqual(MultipleGenericBases.__orig_bases__, (GenericParent[int], GenericParent[float]))
        self.assertEqual(CallTypedDict.__orig_bases__, (TypedDict,))

    def test_zero_fields_typeddicts(self):
        T1 = TypedDict("T1", {})
        class T2(TypedDict): pass
        try:
            ns = {"TypedDict": TypedDict}
            exec("class T3[tvar](TypedDict): pass", ns)
            T3 = ns["T3"]
        except SyntaxError:
            class T3(TypedDict): pass
        S = TypeVar("S")
        class T4(TypedDict, Generic[S]): pass

        expected_warning = re.escape(
            "Failing to pass a value for the 'fields' parameter is deprecated "
            "and will be disallowed in Python 3.15. "
            "To create a TypedDict class with 0 fields "
            "using the functional syntax, "
            "pass an empty dictionary, e.g. `T5 = TypedDict('T5', {})`."
        )
        with self.assertWarnsRegex(DeprecationWarning, fr"^{expected_warning}$"):
            T5 = TypedDict('T5')

        expected_warning = re.escape(
            "Passing `None` as the 'fields' parameter is deprecated "
            "and will be disallowed in Python 3.15. "
            "To create a TypedDict class with 0 fields "
            "using the functional syntax, "
            "pass an empty dictionary, e.g. `T6 = TypedDict('T6', {})`."
        )
        with self.assertWarnsRegex(DeprecationWarning, fr"^{expected_warning}$"):
            T6 = TypedDict('T6', None)

        for klass in T1, T2, T3, T4, T5, T6:
            with self.subTest(klass=klass.__name__):
                self.assertEqual(klass.__annotations__, {})
                self.assertEqual(klass.__required_keys__, set())
                self.assertEqual(klass.__optional_keys__, set())
                self.assertIsInstance(klass(), dict)

    def test_readonly_inheritance(self):
        class Base1(TypedDict):
            a: ReadOnly[int]

        class Child1(Base1):
            b: str

        self.assertEqual(Child1.__readonly_keys__, frozenset({'a'}))
        self.assertEqual(Child1.__mutable_keys__, frozenset({'b'}))

        class Base2(TypedDict):
            a: int

        class Child2(Base2):
            b: ReadOnly[str]

        self.assertEqual(Child2.__readonly_keys__, frozenset({'b'}))
        self.assertEqual(Child2.__mutable_keys__, frozenset({'a'}))

    def test_make_mutable_key_readonly(self):
        class Base(TypedDict):
            a: int

        self.assertEqual(Base.__readonly_keys__, frozenset())
        self.assertEqual(Base.__mutable_keys__, frozenset({'a'}))

        class Child(Base):
            a: ReadOnly[int]  # type checker error, but allowed at runtime

        self.assertEqual(Child.__readonly_keys__, frozenset({'a'}))
        self.assertEqual(Child.__mutable_keys__, frozenset())

    def test_can_make_readonly_key_mutable(self):
        class Base(TypedDict):
            a: ReadOnly[int]

        class Child(Base):
            a: int

        self.assertEqual(Child.__readonly_keys__, frozenset())
        self.assertEqual(Child.__mutable_keys__, frozenset({'a'}))

    def test_combine_qualifiers(self):
        class AllTheThings(TypedDict):
            a: Annotated[Required[ReadOnly[int]], "why not"]
            b: Required[Annotated[ReadOnly[int], "why not"]]
            c: ReadOnly[NotRequired[Annotated[int, "why not"]]]
            d: NotRequired[Annotated[int, "why not"]]

        self.assertEqual(AllTheThings.__required_keys__, frozenset({'a', 'b'}))
        self.assertEqual(AllTheThings.__optional_keys__, frozenset({'c', 'd'}))
        self.assertEqual(AllTheThings.__readonly_keys__, frozenset({'a', 'b', 'c'}))
        self.assertEqual(AllTheThings.__mutable_keys__, frozenset({'d'}))

        self.assertEqual(
            get_type_hints(AllTheThings, include_extras=False),
            {'a': int, 'b': int, 'c': int, 'd': int},
        )
        self.assertEqual(
            get_type_hints(AllTheThings, include_extras=True),
            {
                'a': Annotated[Required[ReadOnly[int]], 'why not'],
                'b': Required[Annotated[ReadOnly[int], 'why not']],
                'c': ReadOnly[NotRequired[Annotated[int, 'why not']]],
                'd': NotRequired[Annotated[int, 'why not']],
            },
        )

    @skipIf(TYPING_3_14_0, "Old syntax only supported on <3.14")
    def test_extra_keys_non_readonly_legacy(self):
        class Base(TypedDict, closed=True):
            __extra_items__: str

        class Child(Base):
            a: NotRequired[int]

        self.assertEqual(Child.__required_keys__, frozenset({}))
        self.assertEqual(Child.__optional_keys__, frozenset({'a'}))
        self.assertEqual(Child.__readonly_keys__, frozenset({}))
        self.assertEqual(Child.__mutable_keys__, frozenset({'a'}))

    @skipIf(TYPING_3_14_0, "Only supported on <3.14")
    def test_extra_keys_readonly_legacy(self):
        class Base(TypedDict, closed=True):
            __extra_items__: ReadOnly[str]

        class Child(Base):
            a: NotRequired[str]

        self.assertEqual(Child.__required_keys__, frozenset({}))
        self.assertEqual(Child.__optional_keys__, frozenset({'a'}))
        self.assertEqual(Child.__readonly_keys__, frozenset({}))
        self.assertEqual(Child.__mutable_keys__, frozenset({'a'}))

    @skipIf(TYPING_3_14_0, "Only supported on <3.14")
    def test_extra_keys_readonly_explicit_closed_legacy(self):
        class Base(TypedDict, closed=True):
            __extra_items__: ReadOnly[str]

        class Child(Base, closed=True):
            a: NotRequired[str]

        self.assertEqual(Child.__required_keys__, frozenset({}))
        self.assertEqual(Child.__optional_keys__, frozenset({'a'}))
        self.assertEqual(Child.__readonly_keys__, frozenset({}))
        self.assertEqual(Child.__mutable_keys__, frozenset({'a'}))

    @skipIf(TYPING_3_14_0, "Only supported on <3.14")
    def test_extra_key_required_legacy(self):
        with self.assertRaisesRegex(
            TypeError,
            "Special key __extra_items__ does not support Required"
        ):
            TypedDict("A", {"__extra_items__": Required[int]}, closed=True)

        with self.assertRaisesRegex(
            TypeError,
            "Special key __extra_items__ does not support NotRequired"
        ):
            TypedDict("A", {"__extra_items__": NotRequired[int]}, closed=True)

    def test_regular_extra_items_legacy(self):
        class ExtraReadOnly(TypedDict):
            __extra_items__: ReadOnly[str]

        self.assertEqual(ExtraReadOnly.__required_keys__, frozenset({'__extra_items__'}))
        self.assertEqual(ExtraReadOnly.__optional_keys__, frozenset({}))
        self.assertEqual(ExtraReadOnly.__readonly_keys__, frozenset({'__extra_items__'}))
        self.assertEqual(ExtraReadOnly.__mutable_keys__, frozenset({}))
        self.assertIs(ExtraReadOnly.__extra_items__, NoExtraItems)
        self.assertIsNone(ExtraReadOnly.__closed__)

        class ExtraRequired(TypedDict):
            __extra_items__: Required[str]

        self.assertEqual(ExtraRequired.__required_keys__, frozenset({'__extra_items__'}))
        self.assertEqual(ExtraRequired.__optional_keys__, frozenset({}))
        self.assertEqual(ExtraRequired.__readonly_keys__, frozenset({}))
        self.assertEqual(ExtraRequired.__mutable_keys__, frozenset({'__extra_items__'}))
        self.assertIs(ExtraRequired.__extra_items__, NoExtraItems)
        self.assertIsNone(ExtraRequired.__closed__)

        class ExtraNotRequired(TypedDict):
            __extra_items__: NotRequired[str]

        self.assertEqual(ExtraNotRequired.__required_keys__, frozenset({}))
        self.assertEqual(ExtraNotRequired.__optional_keys__, frozenset({'__extra_items__'}))
        self.assertEqual(ExtraNotRequired.__readonly_keys__, frozenset({}))
        self.assertEqual(ExtraNotRequired.__mutable_keys__, frozenset({'__extra_items__'}))
        self.assertIs(ExtraNotRequired.__extra_items__, NoExtraItems)
        self.assertIsNone(ExtraNotRequired.__closed__)

    @skipIf(TYPING_3_14_0, "Only supported on <3.14")
    def test_closed_inheritance_legacy(self):
        class Base(TypedDict, closed=True):
            __extra_items__: ReadOnly[Union[str, None]]

        self.assertEqual(Base.__required_keys__, frozenset({}))
        self.assertEqual(Base.__optional_keys__, frozenset({}))
        self.assertEqual(Base.__readonly_keys__, frozenset({}))
        self.assertEqual(Base.__mutable_keys__, frozenset({}))
        self.assertEqual(Base.__annotations__, {})
        self.assertEqual(Base.__extra_items__, ReadOnly[Union[str, None]])
        self.assertIs(Base.__closed__, True)

        class Child(Base, closed=True):
            a: int
            __extra_items__: int

        self.assertEqual(Child.__required_keys__, frozenset({'a'}))
        self.assertEqual(Child.__optional_keys__, frozenset({}))
        self.assertEqual(Child.__readonly_keys__, frozenset({}))
        self.assertEqual(Child.__mutable_keys__, frozenset({'a'}))
        self.assertEqual(Child.__annotations__, {"a": int})
        self.assertIs(Child.__extra_items__, int)
        self.assertIs(Child.__closed__, True)

        class GrandChild(Child, closed=True):
            __extra_items__: str

        self.assertEqual(GrandChild.__required_keys__, frozenset({'a'}))
        self.assertEqual(GrandChild.__optional_keys__, frozenset({}))
        self.assertEqual(GrandChild.__readonly_keys__, frozenset({}))
        self.assertEqual(GrandChild.__mutable_keys__, frozenset({'a'}))
        self.assertEqual(GrandChild.__annotations__, {"a": int})
        self.assertIs(GrandChild.__extra_items__, str)
        self.assertIs(GrandChild.__closed__, True)

    def test_closed_inheritance(self):
        class Base(TypedDict, extra_items=ReadOnly[Union[str, None]]):
            a: int

        self.assertEqual(Base.__required_keys__, frozenset({"a"}))
        self.assertEqual(Base.__optional_keys__, frozenset({}))
        self.assertEqual(Base.__readonly_keys__, frozenset({}))
        self.assertEqual(Base.__mutable_keys__, frozenset({"a"}))
        self.assertEqual(Base.__annotations__, {"a": int})
        self.assertEqual(Base.__extra_items__, ReadOnly[Union[str, None]])
        self.assertIsNone(Base.__closed__)

        class Child(Base, extra_items=int):
            a: str

        self.assertEqual(Child.__required_keys__, frozenset({'a'}))
        self.assertEqual(Child.__optional_keys__, frozenset({}))
        self.assertEqual(Child.__readonly_keys__, frozenset({}))
        self.assertEqual(Child.__mutable_keys__, frozenset({'a'}))
        self.assertEqual(Child.__annotations__, {"a": str})
        self.assertIs(Child.__extra_items__, int)
        self.assertIsNone(Child.__closed__)

        class GrandChild(Child, closed=True):
            a: float

        self.assertEqual(GrandChild.__required_keys__, frozenset({'a'}))
        self.assertEqual(GrandChild.__optional_keys__, frozenset({}))
        self.assertEqual(GrandChild.__readonly_keys__, frozenset({}))
        self.assertEqual(GrandChild.__mutable_keys__, frozenset({'a'}))
        self.assertEqual(GrandChild.__annotations__, {"a": float})
        self.assertIs(GrandChild.__extra_items__, NoExtraItems)
        self.assertIs(GrandChild.__closed__, True)

        class GrandGrandChild(GrandChild):
            ...
        self.assertEqual(GrandGrandChild.__required_keys__, frozenset({'a'}))
        self.assertEqual(GrandGrandChild.__optional_keys__, frozenset({}))
        self.assertEqual(GrandGrandChild.__readonly_keys__, frozenset({}))
        self.assertEqual(GrandGrandChild.__mutable_keys__, frozenset({'a'}))
        self.assertEqual(GrandGrandChild.__annotations__, {"a": float})
        self.assertIs(GrandGrandChild.__extra_items__, NoExtraItems)
        self.assertIsNone(GrandGrandChild.__closed__)

    def test_implicit_extra_items(self):
        class Base(TypedDict):
            a: int

        self.assertIs(Base.__extra_items__, NoExtraItems)
        self.assertIsNone(Base.__closed__)

        class ChildA(Base, closed=True):
            ...

        self.assertEqual(ChildA.__extra_items__, NoExtraItems)
        self.assertIs(ChildA.__closed__, True)

    @skipIf(TYPING_3_14_0, "Backwards compatibility only for Python 3.13")
    def test_implicit_extra_items_before_3_14(self):
        class Base(TypedDict):
            a: int
        class ChildB(Base, closed=True):
            __extra_items__: None

        self.assertIs(ChildB.__extra_items__, type(None))
        self.assertIs(ChildB.__closed__, True)

    @skipIf(
        TYPING_3_13_0,
        "The keyword argument alternative to define a "
        "TypedDict type using the functional syntax is no longer supported"
    )
    def test_backwards_compatibility(self):
        with self.assertWarns(DeprecationWarning):
            TD = TypedDict("TD", closed=int)
        self.assertIs(TD.__closed__, None)
        self.assertEqual(TD.__annotations__, {"closed": int})

        with self.assertWarns(DeprecationWarning):
            TD = TypedDict("TD", extra_items=int)
        self.assertIs(TD.__extra_items__, NoExtraItems)
        self.assertEqual(TD.__annotations__, {"extra_items": int})

    def test_cannot_combine_closed_and_extra_items(self):
        with self.assertRaisesRegex(
            TypeError,
            "Cannot combine closed=True and extra_items"
        ):
            class TD(TypedDict, closed=True, extra_items=range):
                x: str

    def test_typed_dict_signature(self):
        self.assertListEqual(
            list(inspect.signature(TypedDict).parameters),
            ['typename', 'fields', 'total', 'closed', 'extra_items', 'kwargs']
        )

    def test_inline_too_many_arguments(self):
        with self.assertRaises(TypeError):
            TypedDict[{"a": int}, "extra"]

    def test_inline_not_a_dict(self):
        with self.assertRaises(TypeError):
            TypedDict["not_a_dict"]

        # a tuple of elements isn't allowed, even if the first element is a dict:
        with self.assertRaises(TypeError):
            TypedDict[({"key": int},)]

    def test_inline_empty(self):
        TD = TypedDict[{}]
        self.assertIs(TD.__total__, True)
        self.assertIs(TD.__closed__, True)
        self.assertEqual(TD.__extra_items__, NoExtraItems)
        self.assertEqual(TD.__required_keys__, set())
        self.assertEqual(TD.__optional_keys__, set())
        self.assertEqual(TD.__readonly_keys__, set())
        self.assertEqual(TD.__mutable_keys__,  set())

    def test_inline(self):
        TD = TypedDict[{
            "a": int,
            "b": Required[int],
            "c": NotRequired[int],
            "d": ReadOnly[int],
        }]
        self.assertIsSubclass(TD, dict)
        self.assertIsSubclass(TD, typing.MutableMapping)
        self.assertNotIsSubclass(TD, collections.abc.Sequence)
        self.assertTrue(is_typeddict(TD))
        self.assertEqual(TD.__name__, "<inline TypedDict>")
        self.assertEqual(
            TD.__annotations__,
            {"a": int, "b": Required[int], "c": NotRequired[int], "d": ReadOnly[int]},
        )
        self.assertEqual(TD.__module__, __name__)
        self.assertEqual(TD.__bases__, (dict,))
        self.assertIs(TD.__total__, True)
        self.assertIs(TD.__closed__, True)
        self.assertEqual(TD.__extra_items__, NoExtraItems)
        self.assertEqual(TD.__required_keys__, {"a", "b", "d"})
        self.assertEqual(TD.__optional_keys__, {"c"})
        self.assertEqual(TD.__readonly_keys__, {"d"})
        self.assertEqual(TD.__mutable_keys__, {"a", "b", "c"})

        inst = TD(a=1, b=2, d=3)
        self.assertIs(type(inst), dict)
        self.assertEqual(inst["a"], 1)

    def test_annotations(self):
        # _type_check is applied
        with self.assertRaisesRegex(TypeError, "Plain typing.Optional is not valid as type argument"):
            class X(TypedDict):
                a: Optional

        # _type_convert is applied
        class Y(TypedDict):
            a: None
            b: "int"
        if sys.version_info >= (3, 14):
            import annotationlib

            fwdref = EqualToForwardRef('int', module=__name__)
            self.assertEqual(Y.__annotations__, {'a': type(None), 'b': fwdref})
            self.assertEqual(Y.__annotate__(annotationlib.Format.FORWARDREF), {'a': type(None), 'b': fwdref})
        else:
            self.assertEqual(Y.__annotations__, {'a': type(None), 'b': typing.ForwardRef('int', module=__name__)})

    @skipUnless(TYPING_3_14_0, "Only supported on 3.14")
    def test_delayed_type_check(self):
        # _type_check is also applied later
        class Z(TypedDict):
            a: undefined  # noqa: F821

        with self.assertRaises(NameError):
            Z.__annotations__

        undefined = Final
        with self.assertRaisesRegex(TypeError, "Plain typing.Final is not valid as type argument"):
            Z.__annotations__

        undefined = None  # noqa: F841
        self.assertEqual(Z.__annotations__, {'a': type(None)})

    @skipUnless(TYPING_3_14_0, "Only supported on 3.14")
    def test_deferred_evaluation(self):
        class A(TypedDict):
            x: NotRequired[undefined]  # noqa: F821
            y: ReadOnly[undefined]  # noqa: F821
            z: Required[undefined]  # noqa: F821

        self.assertEqual(A.__required_keys__, frozenset({'y', 'z'}))
        self.assertEqual(A.__optional_keys__, frozenset({'x'}))
        self.assertEqual(A.__readonly_keys__, frozenset({'y'}))
        self.assertEqual(A.__mutable_keys__, frozenset({'x', 'z'}))

        with self.assertRaises(NameError):
            A.__annotations__

        import annotationlib
        self.assertEqual(
            A.__annotate__(annotationlib.Format.STRING),
            {'x': 'NotRequired[undefined]', 'y': 'ReadOnly[undefined]',
             'z': 'Required[undefined]'},
        )

    def test_dunder_dict(self):
        self.assertIsInstance(TypedDict.__dict__, dict)

class AnnotatedTests(BaseTestCase):

    def test_repr(self):
        if hasattr(typing, 'Annotated'):
            mod_name = 'typing'
        else:
            mod_name = "typing_extensions"
        self.assertEqual(
            repr(Annotated[int, 4, 5]),
            mod_name + ".Annotated[int, 4, 5]"
        )
        self.assertEqual(
            repr(Annotated[List[int], 4, 5]),
            mod_name + ".Annotated[typing.List[int], 4, 5]"
        )

    def test_flatten(self):
        A = Annotated[Annotated[int, 4], 5]
        self.assertEqual(A, Annotated[int, 4, 5])
        self.assertEqual(A.__metadata__, (4, 5))
        self.assertEqual(A.__origin__, int)

    def test_specialize(self):
        L = Annotated[List[T], "my decoration"]
        LI = Annotated[List[int], "my decoration"]
        self.assertEqual(L[int], Annotated[List[int], "my decoration"])
        self.assertEqual(L[int].__metadata__, ("my decoration",))
        self.assertEqual(L[int].__origin__, List[int])
        with self.assertRaises(TypeError):
            LI[int]
        with self.assertRaises(TypeError):
            L[int, float]

    def test_hash_eq(self):
        self.assertEqual(len({Annotated[int, 4, 5], Annotated[int, 4, 5]}), 1)
        self.assertNotEqual(Annotated[int, 4, 5], Annotated[int, 5, 4])
        self.assertNotEqual(Annotated[int, 4, 5], Annotated[str, 4, 5])
        self.assertNotEqual(Annotated[int, 4], Annotated[int, 4, 4])
        self.assertEqual(
            {Annotated[int, 4, 5], Annotated[int, 4, 5], Annotated[T, 4, 5]},
            {Annotated[int, 4, 5], Annotated[T, 4, 5]}
        )

    def test_instantiate(self):
        class C:
            classvar = 4

            def __init__(self, x):
                self.x = x

            def __eq__(self, other):
                if not isinstance(other, C):
                    return NotImplemented
                return other.x == self.x

        A = Annotated[C, "a decoration"]
        a = A(5)
        c = C(5)
        self.assertEqual(a, c)
        self.assertEqual(a.x, c.x)
        self.assertEqual(a.classvar, c.classvar)

    def test_instantiate_generic(self):
        MyCount = Annotated[typing_extensions.Counter[T], "my decoration"]
        self.assertEqual(MyCount([4, 4, 5]), {4: 2, 5: 1})
        self.assertEqual(MyCount[int]([4, 4, 5]), {4: 2, 5: 1})

    def test_cannot_instantiate_forward(self):
        A = Annotated["int", (5, 6)]
        with self.assertRaises(TypeError):
            A(5)

    def test_cannot_instantiate_type_var(self):
        A = Annotated[T, (5, 6)]
        with self.assertRaises(TypeError):
            A(5)

    def test_cannot_getattr_typevar(self):
        with self.assertRaises(AttributeError):
            Annotated[T, (5, 7)].x

    def test_attr_passthrough(self):
        class C:
            classvar = 4

        A = Annotated[C, "a decoration"]
        self.assertEqual(A.classvar, 4)
        A.x = 5
        self.assertEqual(C.x, 5)

    @skipIf(sys.version_info[:2] == (3, 10), "Waiting for https://github.com/python/cpython/issues/90649 bugfix.")
    def test_special_form_containment(self):
        class C:
            classvar: Annotated[ClassVar[int], "a decoration"] = 4
            const: Annotated[Final[int], "Const"] = 4

        self.assertEqual(get_type_hints(C, globals())["classvar"], ClassVar[int])
        self.assertEqual(get_type_hints(C, globals())["const"], Final[int])

    def test_cannot_subclass(self):
        with self.assertRaisesRegex(TypeError, "Cannot subclass .*Annotated"):
            class C(Annotated):
                pass

    def test_cannot_check_instance(self):
        with self.assertRaises(TypeError):
            isinstance(5, Annotated[int, "positive"])

    def test_cannot_check_subclass(self):
        with self.assertRaises(TypeError):
            issubclass(int, Annotated[int, "positive"])

    def test_pickle(self):
        samples = [typing.Any, typing.Union[int, str],
                   typing.Optional[str], Tuple[int, ...],
                   typing.Callable[[str], bytes],
                   Self, LiteralString, Never]

        for t in samples:
            x = Annotated[t, "a"]

            for prot in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(protocol=prot, type=t):
                    pickled = pickle.dumps(x, prot)
                    restored = pickle.loads(pickled)
                    self.assertEqual(x, restored)

        global _Annotated_test_G

        class _Annotated_test_G(Generic[T]):
            x = 1

        G = Annotated[_Annotated_test_G[int], "A decoration"]
        G.foo = 42
        G.bar = 'abc'

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            z = pickle.dumps(G, proto)
            x = pickle.loads(z)
            self.assertEqual(x.foo, 42)
            self.assertEqual(x.bar, 'abc')
            self.assertEqual(x.x, 1)

    def test_subst(self):
        dec = "a decoration"
        dec2 = "another decoration"

        S = Annotated[T, dec2]
        self.assertEqual(S[int], Annotated[int, dec2])

        self.assertEqual(S[Annotated[int, dec]], Annotated[int, dec, dec2])
        L = Annotated[List[T], dec]

        self.assertEqual(L[int], Annotated[List[int], dec])
        with self.assertRaises(TypeError):
            L[int, int]

        self.assertEqual(S[L[int]], Annotated[List[int], dec, dec2])

        D = Annotated[Dict[KT, VT], dec]
        self.assertEqual(D[str, int], Annotated[Dict[str, int], dec])
        with self.assertRaises(TypeError):
            D[int]

        It = Annotated[int, dec]
        with self.assertRaises(TypeError):
            It[None]

        LI = L[int]
        with self.assertRaises(TypeError):
            LI[None]

    def test_annotated_in_other_types(self):
        X = List[Annotated[T, 5]]
        self.assertEqual(X[int], List[Annotated[int, 5]])

    def test_nested_annotated_with_unhashable_metadata(self):
        X = Annotated[
            List[Annotated[str, {"unhashable_metadata"}]],
            "metadata"
        ]
        self.assertEqual(X.__origin__, List[Annotated[str, {"unhashable_metadata"}]])
        self.assertEqual(X.__metadata__, ("metadata",))

    def test_compatibility(self):
        # Test that the _AnnotatedAlias compatibility alias works
        self.assertTrue(hasattr(typing_extensions, "_AnnotatedAlias"))
        self.assertIs(typing_extensions._AnnotatedAlias, typing._AnnotatedAlias)


class GetTypeHintsTests(BaseTestCase):
    def test_get_type_hints(self):
        def foobar(x: List['X']): ...
        X = Annotated[int, (1, 10)]
        self.assertEqual(
            get_type_hints(foobar, globals(), locals()),
            {'x': List[int]}
        )
        self.assertEqual(
            get_type_hints(foobar, globals(), locals(), include_extras=True),
            {'x': List[Annotated[int, (1, 10)]]}
        )
        BA = Tuple[Annotated[T, (1, 0)], ...]
        def barfoo(x: BA): ...
        self.assertEqual(get_type_hints(barfoo, globals(), locals())['x'], Tuple[T, ...])
        self.assertIs(
            get_type_hints(barfoo, globals(), locals(), include_extras=True)['x'],
            BA
        )
        def barfoo2(x: typing.Callable[..., Annotated[List[T], "const"]],
                    y: typing.Union[int, Annotated[T, "mutable"]]): ...
        self.assertEqual(
            get_type_hints(barfoo2, globals(), locals()),
            {'x': typing.Callable[..., List[T]], 'y': typing.Union[int, T]}
        )
        BA2 = typing.Callable[..., List[T]]
        def barfoo3(x: BA2): ...
        self.assertIs(
            get_type_hints(barfoo3, globals(), locals(), include_extras=True)["x"],
            BA2
        )

    def test_get_type_hints_refs(self):

        Const = Annotated[T, "Const"]

        class MySet(Generic[T]):

            def __ior__(self, other: "Const[MySet[T]]") -> "MySet[T]":
                ...

            def __iand__(self, other: Const["MySet[T]"]) -> "MySet[T]":
                ...

        self.assertEqual(
            get_type_hints(MySet.__iand__, globals(), locals()),
            {'other': MySet[T], 'return': MySet[T]}
        )

        self.assertEqual(
            get_type_hints(MySet.__iand__, globals(), locals(), include_extras=True),
            {'other': Const[MySet[T]], 'return': MySet[T]}
        )

        self.assertEqual(
            get_type_hints(MySet.__ior__, globals(), locals()),
            {'other': MySet[T], 'return': MySet[T]}
        )

    def test_get_type_hints_typeddict(self):
        assert get_type_hints(TotalMovie) == {'title': str, 'year': int}
        assert get_type_hints(TotalMovie, include_extras=True) == {
            'title': str,
            'year': NotRequired[int],
        }

        assert get_type_hints(AnnotatedMovie) == {'title': str, 'year': int}
        assert get_type_hints(AnnotatedMovie, include_extras=True) == {
            'title': Annotated[Required[str], "foobar"],
            'year': NotRequired[Annotated[int, 2000]],
        }

    def test_orig_bases(self):
        T = TypeVar('T')

        class Parent(TypedDict):
            pass

        class Child(Parent):
            pass

        class OtherChild(Parent):
            pass

        class MixedChild(Child, OtherChild, Parent):
            pass

        class GenericParent(TypedDict, Generic[T]):
            pass

        class GenericChild(GenericParent[int]):
            pass

        class OtherGenericChild(GenericParent[str]):
            pass

        class MixedGenericChild(GenericChild, OtherGenericChild, GenericParent[float]):
            pass

        class MultipleGenericBases(GenericParent[int], GenericParent[float]):
            pass

        CallTypedDict = TypedDict('CallTypedDict', {})

        self.assertEqual(Parent.__orig_bases__, (TypedDict,))
        self.assertEqual(Child.__orig_bases__, (Parent,))
        self.assertEqual(OtherChild.__orig_bases__, (Parent,))
        self.assertEqual(MixedChild.__orig_bases__, (Child, OtherChild, Parent,))
        self.assertEqual(GenericParent.__orig_bases__, (TypedDict, Generic[T]))
        self.assertEqual(GenericChild.__orig_bases__, (GenericParent[int],))
        self.assertEqual(OtherGenericChild.__orig_bases__, (GenericParent[str],))
        self.assertEqual(MixedGenericChild.__orig_bases__, (GenericChild, OtherGenericChild, GenericParent[float]))
        self.assertEqual(MultipleGenericBases.__orig_bases__, (GenericParent[int], GenericParent[float]))
        self.assertEqual(CallTypedDict.__orig_bases__, (TypedDict,))


class TypeAliasTests(BaseTestCase):
    def test_canonical_usage_with_variable_annotation(self):
        ns = {}
        exec('Alias: TypeAlias = Employee', globals(), ns)

    def test_canonical_usage_with_type_comment(self):
        Alias: TypeAlias = Employee  # noqa: F841

    def test_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            TypeAlias()

    def test_no_isinstance(self):
        with self.assertRaises(TypeError):
            isinstance(42, TypeAlias)

    def test_no_issubclass(self):
        with self.assertRaises(TypeError):
            issubclass(Employee, TypeAlias)

        with self.assertRaises(TypeError):
            issubclass(TypeAlias, Employee)

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class C(TypeAlias):
                pass

        with self.assertRaises(TypeError):
            class D(type(TypeAlias)):
                pass

    def test_repr(self):
        if hasattr(typing, 'TypeAlias'):
            self.assertEqual(repr(TypeAlias), 'typing.TypeAlias')
        else:
            self.assertEqual(repr(TypeAlias), 'typing_extensions.TypeAlias')

    def test_cannot_subscript(self):
        with self.assertRaises(TypeError):
            TypeAlias[int]

class ParamSpecTests(BaseTestCase):

    def test_basic_plain(self):
        P = ParamSpec('P')
        self.assertEqual(P, P)
        self.assertIsInstance(P, ParamSpec)
        self.assertEqual(P.__name__, 'P')
        # Should be hashable
        hash(P)

    def test_repr(self):
        P = ParamSpec('P')
        P_co = ParamSpec('P_co', covariant=True)
        P_contra = ParamSpec('P_contra', contravariant=True)
        P_infer = ParamSpec('P_infer', infer_variance=True)
        P_2 = ParamSpec('P_2')
        self.assertEqual(repr(P), '~P')
        self.assertEqual(repr(P_2), '~P_2')

        # Note: PEP 612 doesn't require these to be repr-ed correctly, but
        # just follow CPython.
        self.assertEqual(repr(P_co), '+P_co')
        self.assertEqual(repr(P_contra), '-P_contra')
        # On other versions we use typing.ParamSpec, but it is not aware of
        # infer_variance=. Not worth creating our own version of ParamSpec
        # for this.
        if hasattr(typing, 'TypeAliasType') or not hasattr(typing, 'ParamSpec'):
            self.assertEqual(repr(P_infer), 'P_infer')
        else:
            self.assertEqual(repr(P_infer), '~P_infer')

    def test_variance(self):
        P_co = ParamSpec('P_co', covariant=True)
        P_contra = ParamSpec('P_contra', contravariant=True)
        P_infer = ParamSpec('P_infer', infer_variance=True)

        self.assertIs(P_co.__covariant__, True)
        self.assertIs(P_co.__contravariant__, False)
        self.assertIs(P_co.__infer_variance__, False)

        self.assertIs(P_contra.__covariant__, False)
        self.assertIs(P_contra.__contravariant__, True)
        self.assertIs(P_contra.__infer_variance__, False)

        self.assertIs(P_infer.__covariant__, False)
        self.assertIs(P_infer.__contravariant__, False)
        self.assertIs(P_infer.__infer_variance__, True)

    def test_valid_uses(self):
        P = ParamSpec('P')
        T = TypeVar('T')
        C1 = typing.Callable[P, int]
        self.assertEqual(C1.__args__, (P, int))
        self.assertEqual(C1.__parameters__, (P,))
        C2 = typing.Callable[P, T]
        self.assertEqual(C2.__args__, (P, T))
        self.assertEqual(C2.__parameters__, (P, T))

        # Test collections.abc.Callable too.
        # Note: no tests for Callable.__parameters__ here
        # because types.GenericAlias Callable is hardcoded to search
        # for tp_name "TypeVar" in C.  This was changed in 3.10.
        C3 = collections.abc.Callable[P, int]
        self.assertEqual(C3.__args__, (P, int))
        C4 = collections.abc.Callable[P, T]
        self.assertEqual(C4.__args__, (P, T))

        # ParamSpec instances should also have args and kwargs attributes.
        # Note: not in dir(P) because of __class__ hacks
        self.assertTrue(hasattr(P, 'args'))
        self.assertTrue(hasattr(P, 'kwargs'))

    @skipIf((3, 10, 0) <= sys.version_info[:3] <= (3, 10, 2), "Needs https://github.com/python/cpython/issues/90834.")
    def test_args_kwargs(self):
        P = ParamSpec('P')
        P_2 = ParamSpec('P_2')
        # Note: not in dir(P) because of __class__ hacks
        self.assertTrue(hasattr(P, 'args'))
        self.assertTrue(hasattr(P, 'kwargs'))
        self.assertIsInstance(P.args, ParamSpecArgs)
        self.assertIsInstance(P.kwargs, ParamSpecKwargs)
        self.assertIs(P.args.__origin__, P)
        self.assertIs(P.kwargs.__origin__, P)
        self.assertEqual(P.args, P.args)
        self.assertEqual(P.kwargs, P.kwargs)
        self.assertNotEqual(P.args, P_2.args)
        self.assertNotEqual(P.kwargs, P_2.kwargs)
        self.assertNotEqual(P.args, P.kwargs)
        self.assertNotEqual(P.kwargs, P.args)
        self.assertNotEqual(P.args, P_2.kwargs)
        self.assertEqual(repr(P.args), "P.args")
        self.assertEqual(repr(P.kwargs), "P.kwargs")

    def test_user_generics(self):
        T = TypeVar("T")
        P = ParamSpec("P")
        P_2 = ParamSpec("P_2")

        class X(Generic[T, P]):
            pass

        class Y(Protocol[T, P]):
            pass

        things = "arguments" if sys.version_info >= (3, 10) else "parameters"
        for klass in X, Y:
            with self.subTest(klass=klass.__name__):
                G1 = klass[int, P_2]
                self.assertEqual(G1.__args__, (int, P_2))
                self.assertEqual(G1.__parameters__, (P_2,))

                G2 = klass[int, Concatenate[int, P_2]]
                self.assertEqual(G2.__args__, (int, Concatenate[int, P_2]))
                self.assertEqual(G2.__parameters__, (P_2,))

                G3 = klass[int, Concatenate[int, ...]]
                self.assertEqual(G3.__args__, (int, Concatenate[int, ...]))
                self.assertEqual(G3.__parameters__, ())

                with self.assertRaisesRegex(
                    TypeError,
                    f"Too few {things} for {klass}"
                ):
                    klass[int]

        # The following are some valid uses cases in PEP 612 that don't work:
        # These do not work in 3.9, _type_check blocks the list and ellipsis.
        # G3 = X[int, [int, bool]]
        # G4 = X[int, ...]
        # G5 = Z[[int, str, bool]]

    def test_single_argument_generic(self):
        P = ParamSpec("P")
        T = TypeVar("T")
        P_2 = ParamSpec("P_2")

        class Z(Generic[P]):
            pass

        class ProtoZ(Protocol[P]):
            pass

        for klass in Z, ProtoZ:
            with self.subTest(klass=klass.__name__):
                # Note: For 3.10+ __args__ are nested tuples here ((int, ),) instead of (int, )
                G6 = klass[int, str, T]
                G6args = G6.__args__[0] if sys.version_info >= (3, 10) else G6.__args__
                self.assertEqual(G6args, (int, str, T))
                self.assertEqual(G6.__parameters__, (T,))

                # P = [int]
                G7 = klass[int]
                G7args = G7.__args__[0] if sys.version_info >= (3, 10) else G7.__args__
                self.assertEqual(G7args, (int,))
                self.assertEqual(G7.__parameters__, ())

                G8 = klass[Concatenate[T, ...]]
                self.assertEqual(G8.__args__, (Concatenate[T, ...], ))
                self.assertEqual(G8.__parameters__, (T,))

                G9 = klass[Concatenate[T, P_2]]
                self.assertEqual(G9.__args__, (Concatenate[T, P_2], ))

                # This is an invalid form but useful for testing correct subsitution
                G10 = klass[int, Concatenate[str, P]]
                G10args = G10.__args__[0] if sys.version_info >= (3, 10) else G10.__args__
                self.assertEqual(G10args, (int, Concatenate[str, P], ))

    @skipUnless(TYPING_3_10_0, "ParamSpec not present before 3.10")
    def test_is_param_expr(self):
        P = ParamSpec("P")
        P_typing = typing.ParamSpec("P_typing")
        self.assertTrue(typing_extensions._is_param_expr(P))
        self.assertTrue(typing_extensions._is_param_expr(P_typing))
        if hasattr(typing, "_is_param_expr"):
            self.assertTrue(typing._is_param_expr(P))
            self.assertTrue(typing._is_param_expr(P_typing))

    def test_single_argument_generic_with_parameter_expressions(self):
        P = ParamSpec("P")
        T = TypeVar("T")
        P_2 = ParamSpec("P_2")

        class Z(Generic[P]):
            pass

        class ProtoZ(Protocol[P]):
            pass

        things = "arguments" if sys.version_info >= (3, 10) else "parameters"
        for klass in Z, ProtoZ:
            with self.subTest(klass=klass.__name__):
                G8 = klass[Concatenate[T, ...]]

                H8_1 = G8[int]
                self.assertEqual(H8_1.__parameters__, ())
                with self.assertRaisesRegex(TypeError, "not a generic class"):
                    H8_1[str]

                H8_2 = G8[T][int]
                self.assertEqual(H8_2.__parameters__, ())
                with self.assertRaisesRegex(TypeError, "not a generic class"):
                    H8_2[str]

                G9 = klass[Concatenate[T, P_2]]
                self.assertEqual(G9.__parameters__, (T, P_2))

                with self.assertRaisesRegex(TypeError,
                    "The last parameter to Concatenate should be a ParamSpec variable or ellipsis."
                    if sys.version_info < (3, 10) else
                    # from __typing_subst__
                    "Expected a list of types, an ellipsis, ParamSpec, or Concatenate"
                ):
                    G9[int, int]

                with self.assertRaisesRegex(TypeError, f"Too few {things}"):
                    G9[int]

                with self.subTest("Check list as parameter expression", klass=klass.__name__):
                    if sys.version_info < (3, 10):
                        self.skipTest("Cannot pass non-types")
                    G5 = klass[[int, str, T]]
                    self.assertEqual(G5.__parameters__, (T,))
                    self.assertEqual(G5.__args__, ((int, str, T),))

                    H9 = G9[int, [T]]
                    self.assertEqual(H9.__parameters__, (T,))

                # This is an invalid parameter expression but useful for testing correct subsitution
                G10 = klass[int, Concatenate[str, P]]
                with self.subTest("Check invalid form substitution"):
                    self.assertEqual(G10.__parameters__, (P, ))
                    H10 = G10[int]
                    if (3, 10) <= sys.version_info < (3, 11, 3):
                        self.skipTest("3.10-3.11.2 does not substitute Concatenate here")
                    self.assertEqual(H10.__parameters__, ())
                    H10args = H10.__args__[0] if sys.version_info >= (3, 10) else H10.__args__
                    self.assertEqual(H10args, (int, (str, int)))

    @skipUnless(TYPING_3_10_0, "ParamSpec not present before 3.10")
    def test_substitution_with_typing_variants(self):
        # verifies substitution and typing._check_generic working with typing variants
        P = ParamSpec("P")
        typing_P = typing.ParamSpec("typing_P")
        typing_Concatenate = typing.Concatenate[int, P]

        class Z(Generic[typing_P]):
            pass

        P1 = Z[typing_P]
        self.assertEqual(P1.__parameters__, (typing_P,))
        self.assertEqual(P1.__args__, (typing_P,))

        C1 = Z[typing_Concatenate]
        self.assertEqual(C1.__parameters__, (P,))
        self.assertEqual(C1.__args__, (typing_Concatenate,))

    def test_pickle(self):
        global P, P_co, P_contra, P_default
        P = ParamSpec('P')
        P_co = ParamSpec('P_co', covariant=True)
        P_contra = ParamSpec('P_contra', contravariant=True)
        P_default = ParamSpec('P_default', default=[int])
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(f'Pickle protocol {proto}'):
                for paramspec in (P, P_co, P_contra, P_default):
                    z = pickle.loads(pickle.dumps(paramspec, proto))
                    self.assertEqual(z.__name__, paramspec.__name__)
                    self.assertEqual(z.__covariant__, paramspec.__covariant__)
                    self.assertEqual(z.__contravariant__, paramspec.__contravariant__)
                    self.assertEqual(z.__bound__, paramspec.__bound__)
                    self.assertEqual(z.__default__, paramspec.__default__)

    def test_eq(self):
        P = ParamSpec('P')
        self.assertEqual(P, P)
        self.assertEqual(hash(P), hash(P))
        # ParamSpec should compare by id similar to TypeVar in CPython
        self.assertNotEqual(ParamSpec('P'), P)
        self.assertIsNot(ParamSpec('P'), P)
        # Note: normally you don't test this as it breaks when there's
        # a hash collision. However, ParamSpec *must* guarantee that
        # as long as two objects don't have the same ID, their hashes
        # won't be the same.
        self.assertNotEqual(hash(ParamSpec('P')), hash(P))

    def test_isinstance_results_unaffected_by_presence_of_tracing_function(self):
        # See https://github.com/python/typing_extensions/issues/318

        code = textwrap.dedent(
            """\
            import sys, typing

            def trace_call(*args):
                return trace_call

            def run():
                sys.modules.pop("typing_extensions", None)
                from typing_extensions import ParamSpec
                return isinstance(ParamSpec("P"), typing.TypeVar)

            isinstance_result_1 = run()
            sys.setprofile(trace_call)
            isinstance_result_2 = run()
            sys.stdout.write(f"{isinstance_result_1} {isinstance_result_2}")
            """
        )

        # Run this in an isolated process or it pollutes the environment
        # and makes other tests fail:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code], check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as exc:
            print("stdout", exc.stdout, sep="\n")
            print("stderr", exc.stderr, sep="\n")
            raise

        # Sanity checks that assert the test is working as expected
        self.assertIsInstance(proc.stdout, str)
        result1, result2 = proc.stdout.split(" ")
        self.assertIn(result1, {"True", "False"})
        self.assertIn(result2, {"True", "False"})

        # The actual test:
        self.assertEqual(result1, result2)


class ConcatenateTests(BaseTestCase):
    def test_basics(self):
        P = ParamSpec('P')

        class MyClass: ...

        c = Concatenate[MyClass, P]
        self.assertNotEqual(c, Concatenate)

        # Test Ellipsis Concatenation
        d = Concatenate[MyClass, ...]
        self.assertNotEqual(d, c)
        self.assertNotEqual(d, Concatenate)

    @skipUnless(TYPING_3_10_0, "Concatenate not available in <3.10")
    def test_typing_compatibility(self):
        P = ParamSpec('P')
        C1 = Concatenate[int, P][typing.Concatenate[int, P]]
        self.assertEqual(C1, Concatenate[int, int, P])
        self.assertEqual(get_args(C1), (int, int, P))

        C2 = typing.Concatenate[int, P][Concatenate[int, P]]
        with self.subTest("typing compatibility with typing_extensions"):
            if sys.version_info < (3, 10, 3):
                self.skipTest("Unpacking not introduced until 3.10.3")
            self.assertEqual(get_args(C2), (int, int, P))

    def test_valid_uses(self):
        P = ParamSpec('P')
        T = TypeVar('T')
        for callable_variant in (Callable, collections.abc.Callable):
            with self.subTest(callable_variant=callable_variant):
                C1 = callable_variant[Concatenate[int, P], int]
                C2 = callable_variant[Concatenate[int, T, P], T]
                self.assertEqual(C1.__origin__, C2.__origin__)
                self.assertNotEqual(C1, C2)

                C3 = callable_variant[Concatenate[int, ...], int]
                C4 = callable_variant[Concatenate[int, T, ...], T]
                self.assertEqual(C3.__origin__, C4.__origin__)
                self.assertNotEqual(C3, C4)

    def test_invalid_uses(self):
        P = ParamSpec('P')
        T = TypeVar('T')

        with self.assertRaisesRegex(
            TypeError,
            'Cannot take a Concatenate of no types',
        ):
            Concatenate[()]

        with self.assertRaisesRegex(
            TypeError,
            'The last parameter to Concatenate should be a ParamSpec variable or ellipsis',
        ):
            Concatenate[P, T]

        # Test with tuple argument
        with self.assertRaisesRegex(
            TypeError,
            "The last parameter to Concatenate should be a ParamSpec variable or ellipsis.",
        ):
            Concatenate[(P, T)]

        with self.assertRaisesRegex(
            TypeError,
            'is not a generic class',
        ):
            Callable[Concatenate[int, ...], Any][Any]

        # Assure that `_type_check` is called.
        P = ParamSpec('P')
        with self.assertRaisesRegex(
            TypeError,
            "each arg must be a type",
        ):
            Concatenate[(str,), P]

    @skipUnless(TYPING_3_10_0, "Missing backport to 3.9. See issue #48")
    def test_alias_subscription_with_ellipsis(self):
        P = ParamSpec('P')
        X = Callable[Concatenate[int, P], Any]

        C1 = X[...]
        self.assertEqual(C1.__parameters__, ())
        self.assertEqual(get_args(C1), (Concatenate[int, ...], Any))

    def test_basic_introspection(self):
        P = ParamSpec('P')
        C1 = Concatenate[int, P]
        C2 = Concatenate[int, T, P]
        C3 = Concatenate[int, ...]
        C4 = Concatenate[int, T, ...]
        self.assertEqual(C1.__origin__, Concatenate)
        self.assertEqual(C1.__args__, (int, P))
        self.assertEqual(C2.__origin__, Concatenate)
        self.assertEqual(C2.__args__, (int, T, P))
        self.assertEqual(C3.__origin__, Concatenate)
        self.assertEqual(C3.__args__, (int, Ellipsis))
        self.assertEqual(C4.__origin__, Concatenate)
        self.assertEqual(C4.__args__, (int, T, Ellipsis))

    def test_eq(self):
        P = ParamSpec('P')
        C1 = Concatenate[int, P]
        C2 = Concatenate[int, P]
        C3 = Concatenate[int, T, P]
        self.assertEqual(C1, C2)
        self.assertEqual(hash(C1), hash(C2))
        self.assertNotEqual(C1, C3)

        C4 = Concatenate[int, ...]
        C5 = Concatenate[int, ...]
        C6 = Concatenate[int, T, ...]
        self.assertEqual(C4, C5)
        self.assertEqual(hash(C4), hash(C5))
        self.assertNotEqual(C4, C6)

    def test_substitution(self):
        T = TypeVar('T')
        P = ParamSpec('P')
        Ts = TypeVarTuple("Ts")

        C1 = Concatenate[str, T, ...]
        self.assertEqual(C1[int], Concatenate[str, int, ...])

        C2 = Concatenate[str, P]
        self.assertEqual(C2[...], Concatenate[str, ...])
        self.assertEqual(C2[int], (str, int))
        U1 = Unpack[Tuple[int, str]]
        U2 = Unpack[Ts]
        self.assertEqual(C2[U1], (str, int, str))
        self.assertEqual(C2[U2], (str, Unpack[Ts]))
        self.assertEqual(C2["U2"], (str, EqualToForwardRef("U2")))

        if (3, 12, 0) <= sys.version_info < (3, 12, 4):
            with self.assertRaises(AssertionError):
                C2[Unpack[U2]]
        else:
            with self.assertRaisesRegex(TypeError, "must be used with a tuple type"):
                C2[Unpack[U2]]

        C3 = Concatenate[str, T, P]
        self.assertEqual(C3[int, [bool]], (str, int, bool))

    @skipUnless(TYPING_3_10_0, "Concatenate not present before 3.10")
    def test_is_param_expr(self):
        P = ParamSpec('P')
        concat = Concatenate[str, P]
        typing_concat = typing.Concatenate[str, P]
        self.assertTrue(typing_extensions._is_param_expr(concat))
        self.assertTrue(typing_extensions._is_param_expr(typing_concat))
        if hasattr(typing, "_is_param_expr"):
            self.assertTrue(typing._is_param_expr(concat))
            self.assertTrue(typing._is_param_expr(typing_concat))

class TypeGuardTests(BaseTestCase):
    def test_basics(self):
        TypeGuard[int]  # OK
        self.assertEqual(TypeGuard[int], TypeGuard[int])

        def foo(arg) -> TypeGuard[int]: ...
        self.assertEqual(gth(foo), {'return': TypeGuard[int]})

    def test_repr(self):
        if hasattr(typing, 'TypeGuard'):
            mod_name = 'typing'
        else:
            mod_name = 'typing_extensions'
        self.assertEqual(repr(TypeGuard), f'{mod_name}.TypeGuard')
        cv = TypeGuard[int]
        self.assertEqual(repr(cv), f'{mod_name}.TypeGuard[int]')
        cv = TypeGuard[Employee]
        self.assertEqual(repr(cv), f'{mod_name}.TypeGuard[{__name__}.Employee]')
        cv = TypeGuard[Tuple[int]]
        self.assertEqual(repr(cv), f'{mod_name}.TypeGuard[typing.Tuple[int]]')

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class C(type(TypeGuard)):
                pass
        with self.assertRaises(TypeError):
            class D(type(TypeGuard[int])):
                pass

    def test_cannot_init(self):
        with self.assertRaises(TypeError):
            TypeGuard()
        with self.assertRaises(TypeError):
            type(TypeGuard)()
        with self.assertRaises(TypeError):
            type(TypeGuard[Optional[int]])()

    def test_no_isinstance(self):
        with self.assertRaises(TypeError):
            isinstance(1, TypeGuard[int])
        with self.assertRaises(TypeError):
            issubclass(int, TypeGuard)


class TypeIsTests(BaseTestCase):
    def test_basics(self):
        TypeIs[int]  # OK
        self.assertEqual(TypeIs[int], TypeIs[int])

        def foo(arg) -> TypeIs[int]: ...
        self.assertEqual(gth(foo), {'return': TypeIs[int]})

    def test_repr(self):
        if hasattr(typing, 'TypeIs'):
            mod_name = 'typing'
        else:
            mod_name = 'typing_extensions'
        self.assertEqual(repr(TypeIs), f'{mod_name}.TypeIs')
        cv = TypeIs[int]
        self.assertEqual(repr(cv), f'{mod_name}.TypeIs[int]')
        cv = TypeIs[Employee]
        self.assertEqual(repr(cv), f'{mod_name}.TypeIs[{__name__}.Employee]')
        cv = TypeIs[Tuple[int]]
        self.assertEqual(repr(cv), f'{mod_name}.TypeIs[typing.Tuple[int]]')

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class C(type(TypeIs)):
                pass
        with self.assertRaises(TypeError):
            class D(type(TypeIs[int])):
                pass

    def test_cannot_init(self):
        with self.assertRaises(TypeError):
            TypeIs()
        with self.assertRaises(TypeError):
            type(TypeIs)()
        with self.assertRaises(TypeError):
            type(TypeIs[Optional[int]])()

    def test_no_isinstance(self):
        with self.assertRaises(TypeError):
            isinstance(1, TypeIs[int])
        with self.assertRaises(TypeError):
            issubclass(int, TypeIs)


class TypeFormTests(BaseTestCase):
    def test_basics(self):
        TypeForm[int]  # OK
        self.assertEqual(TypeForm[int], TypeForm[int])

        def foo(arg) -> TypeForm[int]: ...
        self.assertEqual(gth(foo), {'return': TypeForm[int]})

    def test_repr(self):
        if hasattr(typing, 'TypeForm'):
            mod_name = 'typing'
        else:
            mod_name = 'typing_extensions'
        self.assertEqual(repr(TypeForm), f'{mod_name}.TypeForm')
        cv = TypeForm[int]
        self.assertEqual(repr(cv), f'{mod_name}.TypeForm[int]')
        cv = TypeForm[Employee]
        self.assertEqual(repr(cv), f'{mod_name}.TypeForm[{__name__}.Employee]')
        cv = TypeForm[Tuple[int]]
        self.assertEqual(repr(cv), f'{mod_name}.TypeForm[typing.Tuple[int]]')

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class C(type(TypeForm)):
                pass
        with self.assertRaises(TypeError):
            class D(type(TypeForm[int])):
                pass

    def test_call(self):
        objs = [
            1,
            "int",
            int,
            Tuple[int, str],
        ]
        for obj in objs:
            with self.subTest(obj=obj):
                self.assertIs(TypeForm(obj), obj)

        with self.assertRaises(TypeError):
            TypeForm()
        with self.assertRaises(TypeError):
            TypeForm("too", "many")

    def test_cannot_init_type(self):
        with self.assertRaises(TypeError):
            type(TypeForm)()
        with self.assertRaises(TypeError):
            type(TypeForm[Optional[int]])()

    def test_no_isinstance(self):
        with self.assertRaises(TypeError):
            isinstance(1, TypeForm[int])
        with self.assertRaises(TypeError):
            issubclass(int, TypeForm)


class CoolEmployee(NamedTuple):
    name: str
    cool: int


class CoolEmployeeWithDefault(NamedTuple):
    name: str
    cool: int = 0


class XMeth(NamedTuple):
    x: int

    def double(self):
        return 2 * self.x


class NamedTupleTests(BaseTestCase):
    class NestedEmployee(NamedTuple):
        name: str
        cool: int

    def test_basics(self):
        Emp = NamedTuple('Emp', [('name', str), ('id', int)])
        self.assertIsSubclass(Emp, tuple)
        joe = Emp('Joe', 42)
        jim = Emp(name='Jim', id=1)
        self.assertIsInstance(joe, Emp)
        self.assertIsInstance(joe, tuple)
        self.assertEqual(joe.name, 'Joe')
        self.assertEqual(joe.id, 42)
        self.assertEqual(jim.name, 'Jim')
        self.assertEqual(jim.id, 1)
        self.assertEqual(Emp.__name__, 'Emp')
        self.assertEqual(Emp._fields, ('name', 'id'))
        self.assertEqual(Emp.__annotations__,
                         collections.OrderedDict([('name', str), ('id', int)]))

    def test_annotation_usage(self):
        tim = CoolEmployee('Tim', 9000)
        self.assertIsInstance(tim, CoolEmployee)
        self.assertIsInstance(tim, tuple)
        self.assertEqual(tim.name, 'Tim')
        self.assertEqual(tim.cool, 9000)
        self.assertEqual(CoolEmployee.__name__, 'CoolEmployee')
        self.assertEqual(CoolEmployee._fields, ('name', 'cool'))
        self.assertEqual(CoolEmployee.__annotations__,
                         collections.OrderedDict(name=str, cool=int))

    def test_annotation_usage_with_default(self):
        jelle = CoolEmployeeWithDefault('Jelle')
        self.assertIsInstance(jelle, CoolEmployeeWithDefault)
        self.assertIsInstance(jelle, tuple)
        self.assertEqual(jelle.name, 'Jelle')
        self.assertEqual(jelle.cool, 0)
        cooler_employee = CoolEmployeeWithDefault('Sjoerd', 1)
        self.assertEqual(cooler_employee.cool, 1)

        self.assertEqual(CoolEmployeeWithDefault.__name__, 'CoolEmployeeWithDefault')
        self.assertEqual(CoolEmployeeWithDefault._fields, ('name', 'cool'))
        self.assertEqual(CoolEmployeeWithDefault.__annotations__,
                         dict(name=str, cool=int))

        with self.assertRaisesRegex(
            TypeError,
            'Non-default namedtuple field y cannot follow default field x'
        ):
            class NonDefaultAfterDefault(NamedTuple):
                x: int = 3
                y: int

    def test_field_defaults(self):
        self.assertEqual(CoolEmployeeWithDefault._field_defaults, dict(cool=0))

    def test_annotation_usage_with_methods(self):
        self.assertEqual(XMeth(1).double(), 2)
        self.assertEqual(XMeth(42).x, XMeth(42)[0])
        self.assertEqual(str(XRepr(42)), '42 -> 1')
        self.assertEqual(XRepr(1, 2) + XRepr(3), 0)

        bad_overwrite_error_message = 'Cannot overwrite NamedTuple attribute'

        with self.assertRaisesRegex(AttributeError, bad_overwrite_error_message):
            class XMethBad(NamedTuple):
                x: int
                def _fields(self):
                    return 'no chance for this'

        with self.assertRaisesRegex(AttributeError, bad_overwrite_error_message):
            class XMethBad2(NamedTuple):
                x: int
                def _source(self):
                    return 'no chance for this as well'

    def test_multiple_inheritance(self):
        class A:
            pass
        with self.assertRaisesRegex(
            TypeError,
            'can only inherit from a NamedTuple type and Generic'
        ):
            class X(NamedTuple, A):
                x: int

        with self.assertRaisesRegex(
            TypeError,
            'can only inherit from a NamedTuple type and Generic'
        ):
            class Y(NamedTuple, tuple):
                x: int

        with self.assertRaisesRegex(TypeError, 'duplicate base class'):
            class Z(NamedTuple, NamedTuple):
                x: int

        class A(NamedTuple):
            x: int
        with self.assertRaisesRegex(
            TypeError,
            'can only inherit from a NamedTuple type and Generic'
        ):
            class XX(NamedTuple, A):
                y: str

    def test_generic(self):
        class X(NamedTuple, Generic[T]):
            x: T
        self.assertEqual(X.__bases__, (tuple, Generic))
        self.assertEqual(X.__orig_bases__, (NamedTuple, Generic[T]))
        self.assertEqual(X.__mro__, (X, tuple, Generic, object))

        class Y(Generic[T], NamedTuple):
            x: T
        self.assertEqual(Y.__bases__, (Generic, tuple))
        self.assertEqual(Y.__orig_bases__, (Generic[T], NamedTuple))
        self.assertEqual(Y.__mro__, (Y, Generic, tuple, object))

        for G in X, Y:
            with self.subTest(type=G):
                self.assertEqual(G.__parameters__, (T,))
                A = G[int]
                self.assertIs(A.__origin__, G)
                self.assertEqual(A.__args__, (int,))
                self.assertEqual(A.__parameters__, ())

                a = A(3)
                self.assertIs(type(a), G)
                self.assertIsInstance(a, G)
                self.assertEqual(a.x, 3)

                things = "arguments" if sys.version_info >= (3, 10) else "parameters"
                with self.assertRaisesRegex(TypeError, f'Too many {things}'):
                    G[int, str]

    def test_non_generic_subscript_py39_plus(self):
        # For backward compatibility, subscription works
        # on arbitrary NamedTuple types.
        class Group(NamedTuple):
            key: T
            group: list[T]
        A = Group[int]
        self.assertEqual(A.__origin__, Group)
        self.assertEqual(A.__parameters__, ())
        self.assertEqual(A.__args__, (int,))
        a = A(1, [2])
        self.assertIs(type(a), Group)
        self.assertEqual(a, (1, [2]))

    @skipUnless(sys.version_info <= (3, 15), "Behavior removed in 3.15")
    def test_namedtuple_keyword_usage(self):
        with self.assertWarnsRegex(
            DeprecationWarning,
            "Creating NamedTuple classes using keyword arguments is deprecated"
        ):
            LocalEmployee = NamedTuple("LocalEmployee", name=str, age=int)

        nick = LocalEmployee('Nick', 25)
        self.assertIsInstance(nick, tuple)
        self.assertEqual(nick.name, 'Nick')
        self.assertEqual(LocalEmployee.__name__, 'LocalEmployee')
        self.assertEqual(LocalEmployee._fields, ('name', 'age'))
        self.assertEqual(LocalEmployee.__annotations__, dict(name=str, age=int))

        with self.assertRaisesRegex(
            TypeError,
            "Either list of fields or keywords can be provided to NamedTuple, not both"
        ):
            NamedTuple('Name', [('x', int)], y=str)

        with self.assertRaisesRegex(
            TypeError,
            "Either list of fields or keywords can be provided to NamedTuple, not both"
        ):
            NamedTuple('Name', [], y=str)

        with self.assertRaisesRegex(
            TypeError,
            (
                r"Cannot pass `None` as the 'fields' parameter "
                r"and also specify fields using keyword arguments"
            )
        ):
            NamedTuple('Name', None, x=int)

    @skipUnless(sys.version_info <= (3, 15), "Behavior removed in 3.15")
    def test_namedtuple_special_keyword_names(self):
        with self.assertWarnsRegex(
            DeprecationWarning,
            "Creating NamedTuple classes using keyword arguments is deprecated"
        ):
            NT = NamedTuple("NT", cls=type, self=object, typename=str, fields=list)

        self.assertEqual(NT.__name__, 'NT')
        self.assertEqual(NT._fields, ('cls', 'self', 'typename', 'fields'))
        a = NT(cls=str, self=42, typename='foo', fields=[('bar', tuple)])
        self.assertEqual(a.cls, str)
        self.assertEqual(a.self, 42)
        self.assertEqual(a.typename, 'foo')
        self.assertEqual(a.fields, [('bar', tuple)])

    @skipUnless(sys.version_info <= (3, 15), "Behavior removed in 3.15")
    def test_empty_namedtuple(self):
        expected_warning = re.escape(
            "Failing to pass a value for the 'fields' parameter is deprecated "
            "and will be disallowed in Python 3.15. "
            "To create a NamedTuple class with 0 fields "
            "using the functional syntax, "
            "pass an empty list, e.g. `NT1 = NamedTuple('NT1', [])`."
        )
        with self.assertWarnsRegex(DeprecationWarning, fr"^{expected_warning}$"):
            NT1 = NamedTuple('NT1')

        expected_warning = re.escape(
            "Passing `None` as the 'fields' parameter is deprecated "
            "and will be disallowed in Python 3.15. "
            "To create a NamedTuple class with 0 fields "
            "using the functional syntax, "
            "pass an empty list, e.g. `NT2 = NamedTuple('NT2', [])`."
        )
        with self.assertWarnsRegex(DeprecationWarning, fr"^{expected_warning}$"):
            NT2 = NamedTuple('NT2', None)

        NT3 = NamedTuple('NT2', [])

        class CNT(NamedTuple):
            pass  # empty body

        for struct in NT1, NT2, NT3, CNT:
            with self.subTest(struct=struct):
                self.assertEqual(struct._fields, ())
                self.assertEqual(struct.__annotations__, {})
                self.assertIsInstance(struct(), struct)
                self.assertEqual(struct._field_defaults, {})

    def test_namedtuple_errors(self):
        with self.assertRaises(TypeError):
            NamedTuple.__new__()
        with self.assertRaises(TypeError):
            NamedTuple()
        with self.assertRaises(TypeError):
            NamedTuple('Emp', [('name', str)], None)
        with self.assertRaisesRegex(ValueError, 'cannot start with an underscore'):
            NamedTuple('Emp', [('_name', str)])
        with self.assertRaises(TypeError):
            NamedTuple(typename='Emp', name=str, id=int)

    def test_copy_and_pickle(self):
        global Emp  # pickle wants to reference the class by name
        Emp = NamedTuple('Emp', [('name', str), ('cool', int)])
        for cls in Emp, CoolEmployee, self.NestedEmployee:
            with self.subTest(cls=cls):
                jane = cls('jane', 37)
                for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                    z = pickle.dumps(jane, proto)
                    jane2 = pickle.loads(z)
                    self.assertEqual(jane2, jane)
                    self.assertIsInstance(jane2, cls)

                jane2 = copy.copy(jane)
                self.assertEqual(jane2, jane)
                self.assertIsInstance(jane2, cls)

                jane2 = copy.deepcopy(jane)
                self.assertEqual(jane2, jane)
                self.assertIsInstance(jane2, cls)

    def test_docstring(self):
        self.assertIsInstance(NamedTuple.__doc__, str)

    def test_same_as_typing_NamedTuple(self):
        self.assertEqual(
            set(dir(NamedTuple)) - {"__text_signature__"},
            set(dir(typing.NamedTuple))
        )
        self.assertIs(type(NamedTuple), type(typing.NamedTuple))

    def test_orig_bases(self):
        T = TypeVar('T')

        class SimpleNamedTuple(NamedTuple):
            pass

        class GenericNamedTuple(NamedTuple, Generic[T]):
            pass

        self.assertEqual(SimpleNamedTuple.__orig_bases__, (NamedTuple,))
        self.assertEqual(GenericNamedTuple.__orig_bases__, (NamedTuple, Generic[T]))

        CallNamedTuple = NamedTuple('CallNamedTuple', [])

        self.assertEqual(CallNamedTuple.__orig_bases__, (NamedTuple,))

    def test_setname_called_on_values_in_class_dictionary(self):
        class Vanilla:
            def __set_name__(self, owner, name):
                self.name = name

        class Foo(NamedTuple):
            attr = Vanilla()

        foo = Foo()
        self.assertEqual(len(foo), 0)
        self.assertNotIn('attr', Foo._fields)
        self.assertIsInstance(foo.attr, Vanilla)
        self.assertEqual(foo.attr.name, "attr")

        class Bar(NamedTuple):
            attr: Vanilla = Vanilla()

        bar = Bar()
        self.assertEqual(len(bar), 1)
        self.assertIn('attr', Bar._fields)
        self.assertIsInstance(bar.attr, Vanilla)
        self.assertEqual(bar.attr.name, "attr")

    @skipIf(
        TYPING_3_12_0,
        "__set_name__ behaviour changed on py312+ to use BaseException.add_note()"
    )
    def test_setname_raises_the_same_as_on_other_classes_py311_minus(self):
        class CustomException(BaseException): pass

        class Annoying:
            def __set_name__(self, owner, name):
                raise CustomException

        annoying = Annoying()

        with self.assertRaises(RuntimeError) as cm:
            class NormalClass:
                attr = annoying
        normal_exception = cm.exception

        with self.assertRaises(RuntimeError) as cm:
            class NamedTupleClass(NamedTuple):
                attr = annoying
        namedtuple_exception = cm.exception

        self.assertIs(type(namedtuple_exception), RuntimeError)
        self.assertIs(type(namedtuple_exception), type(normal_exception))
        self.assertEqual(len(namedtuple_exception.args), len(normal_exception.args))
        self.assertEqual(
            namedtuple_exception.args[0],
            normal_exception.args[0].replace("NormalClass", "NamedTupleClass")
        )

        self.assertIs(type(namedtuple_exception.__cause__), CustomException)
        self.assertIs(
            type(namedtuple_exception.__cause__), type(normal_exception.__cause__)
        )
        self.assertEqual(
            namedtuple_exception.__cause__.args, normal_exception.__cause__.args
        )

    @skipUnless(
        TYPING_3_12_0,
        "__set_name__ behaviour changed on py312+ to use BaseException.add_note()"
    )
    def test_setname_raises_the_same_as_on_other_classes_py312_plus(self):
        class CustomException(BaseException): pass

        class Annoying:
            def __set_name__(self, owner, name):
                raise CustomException

        annoying = Annoying()

        with self.assertRaises(CustomException) as cm:
            class NormalClass:
                attr = annoying
        normal_exception = cm.exception

        with self.assertRaises(CustomException) as cm:
            class NamedTupleClass(NamedTuple):
                attr = annoying
        namedtuple_exception = cm.exception

        expected_note = (
            "Error calling __set_name__ on 'Annoying' instance "
            "'attr' in 'NamedTupleClass'"
        )

        self.assertIs(type(namedtuple_exception), CustomException)
        self.assertIs(type(namedtuple_exception), type(normal_exception))
        self.assertEqual(namedtuple_exception.args, normal_exception.args)

        self.assertEqual(len(namedtuple_exception.__notes__), 1)
        self.assertEqual(
            len(namedtuple_exception.__notes__), len(normal_exception.__notes__)
        )

        self.assertEqual(namedtuple_exception.__notes__[0], expected_note)
        self.assertEqual(
            namedtuple_exception.__notes__[0],
            normal_exception.__notes__[0].replace("NormalClass", "NamedTupleClass")
        )

    def test_strange_errors_when_accessing_set_name_itself(self):
        class CustomException(Exception): pass

        class Meta(type):
            def __getattribute__(self, attr):
                if attr == "__set_name__":
                    raise CustomException
                return object.__getattribute__(self, attr)

        class VeryAnnoying(metaclass=Meta): pass

        very_annoying = VeryAnnoying()

        with self.assertRaises(CustomException):
            class Foo(NamedTuple):
                attr = very_annoying


class TypeVarTests(BaseTestCase):
    def test_basic_plain(self):
        T = TypeVar('T')
        # T equals itself.
        self.assertEqual(T, T)
        # T is an instance of TypeVar
        self.assertIsInstance(T, TypeVar)
        self.assertEqual(T.__name__, 'T')
        self.assertEqual(T.__constraints__, ())
        self.assertIs(T.__bound__, None)
        self.assertIs(T.__covariant__, False)
        self.assertIs(T.__contravariant__, False)
        self.assertIs(T.__infer_variance__, False)

    def test_attributes(self):
        T_bound = TypeVar('T_bound', bound=int)
        self.assertEqual(T_bound.__name__, 'T_bound')
        self.assertEqual(T_bound.__constraints__, ())
        self.assertIs(T_bound.__bound__, int)

        T_constraints = TypeVar('T_constraints', int, str)
        self.assertEqual(T_constraints.__name__, 'T_constraints')
        self.assertEqual(T_constraints.__constraints__, (int, str))
        self.assertIs(T_constraints.__bound__, None)

        T_co = TypeVar('T_co', covariant=True)
        self.assertEqual(T_co.__name__, 'T_co')
        self.assertIs(T_co.__covariant__, True)
        self.assertIs(T_co.__contravariant__, False)
        self.assertIs(T_co.__infer_variance__, False)

        T_contra = TypeVar('T_contra', contravariant=True)
        self.assertEqual(T_contra.__name__, 'T_contra')
        self.assertIs(T_contra.__covariant__, False)
        self.assertIs(T_contra.__contravariant__, True)
        self.assertIs(T_contra.__infer_variance__, False)

        T_infer = TypeVar('T_infer', infer_variance=True)
        self.assertEqual(T_infer.__name__, 'T_infer')
        self.assertIs(T_infer.__covariant__, False)
        self.assertIs(T_infer.__contravariant__, False)
        self.assertIs(T_infer.__infer_variance__, True)

    def test_typevar_instance_type_error(self):
        T = TypeVar('T')
        with self.assertRaises(TypeError):
            isinstance(42, T)

    def test_typevar_subclass_type_error(self):
        T = TypeVar('T')
        with self.assertRaises(TypeError):
            issubclass(int, T)
        with self.assertRaises(TypeError):
            issubclass(T, int)

    def test_constrained_error(self):
        with self.assertRaises(TypeError):
            X = TypeVar('X', int)
            X

    def test_union_unique(self):
        X = TypeVar('X')
        Y = TypeVar('Y')
        self.assertNotEqual(X, Y)
        self.assertEqual(Union[X], X)
        self.assertNotEqual(Union[X], Union[X, Y])
        self.assertEqual(Union[X, X], X)
        self.assertNotEqual(Union[X, int], Union[X])
        self.assertNotEqual(Union[X, int], Union[int])
        self.assertEqual(Union[X, int].__args__, (X, int))
        self.assertEqual(Union[X, int].__parameters__, (X,))
        self.assertIs(Union[X, int].__origin__, Union)

    if hasattr(types, "UnionType"):
        def test_or(self):
            X = TypeVar('X')
            # use a string because str doesn't implement
            # __or__/__ror__ itself
            self.assertEqual(X | "x", Union[X, "x"])
            self.assertEqual("x" | X, Union["x", X])
            # make sure the order is correct
            self.assertEqual(get_args(X | "x"), (X, EqualToForwardRef("x")))
            self.assertEqual(get_args("x" | X), (EqualToForwardRef("x"), X))

    def test_union_constrained(self):
        A = TypeVar('A', str, bytes)
        self.assertNotEqual(Union[A, str], Union[A])

    def test_repr(self):
        self.assertEqual(repr(T), '~T')
        self.assertEqual(repr(KT), '~KT')
        self.assertEqual(repr(VT), '~VT')
        self.assertEqual(repr(AnyStr), '~AnyStr')
        T_co = TypeVar('T_co', covariant=True)
        self.assertEqual(repr(T_co), '+T_co')
        T_contra = TypeVar('T_contra', contravariant=True)
        self.assertEqual(repr(T_contra), '-T_contra')

    def test_no_redefinition(self):
        self.assertNotEqual(TypeVar('T'), TypeVar('T'))
        self.assertNotEqual(TypeVar('T', int, str), TypeVar('T', int, str))

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class V(TypeVar): pass
        T = TypeVar("T")
        with self.assertRaises(TypeError):
            class W(T): pass

    def test_cannot_instantiate_vars(self):
        with self.assertRaises(TypeError):
            TypeVar('A')()

    def test_bound_errors(self):
        with self.assertRaises(TypeError):
            TypeVar('X', bound=Optional)
        with self.assertRaises(TypeError):
            TypeVar('X', str, float, bound=Employee)
        with self.assertRaisesRegex(TypeError,
                                    r"Bound must be a type\. Got \(1, 2\)\."):
            TypeVar('X', bound=(1, 2))

    def test_missing__name__(self):
        # See https://github.com/python/cpython/issues/84123
        code = ("import typing\n"
                "T = typing.TypeVar('T')\n"
                )
        exec(code, {})

    def test_no_bivariant(self):
        with self.assertRaises(ValueError):
            TypeVar('T', covariant=True, contravariant=True)

    def test_cannot_combine_explicit_and_infer(self):
        with self.assertRaises(ValueError):
            TypeVar('T', covariant=True, infer_variance=True)
        with self.assertRaises(ValueError):
            TypeVar('T', contravariant=True, infer_variance=True)


class MyClass:
    def __repr__(self):
        return "my repr"


def times_three(fn):
    @functools.wraps(fn)
    def wrapper(a, b):
        return fn(a * 3, b * 3)

    return wrapper


if __name__ == '__main__':
    main()

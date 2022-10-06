import sys
import os
import abc
import contextlib
import collections
from collections import defaultdict
import collections.abc
import copy
from functools import lru_cache
import inspect
import pickle
import subprocess
import types
from unittest import TestCase, main, skipUnless, skipIf
from unittest.mock import patch
from test import ann_module, ann_module2, ann_module3
import typing
from typing import TypeVar, Optional, Union, AnyStr
from typing import T, KT, VT  # Not in __all__.
from typing import Tuple, List, Dict, Iterable, Iterator, Callable
from typing import Generic
from typing import no_type_check
import typing_extensions
from typing_extensions import NoReturn, Any, ClassVar, Final, IntVar, Literal, Type, NewType, TypedDict, Self
from typing_extensions import TypeAlias, ParamSpec, Concatenate, ParamSpecArgs, ParamSpecKwargs, TypeGuard
from typing_extensions import Awaitable, AsyncIterator, AsyncContextManager, Required, NotRequired
from typing_extensions import Protocol, runtime, runtime_checkable, Annotated, final, is_typeddict
from typing_extensions import TypeVarTuple, Unpack, dataclass_transform, reveal_type, Never, assert_never, LiteralString
from typing_extensions import assert_type, get_type_hints, get_origin, get_args
from typing_extensions import clear_overloads, get_overloads, overload
from typing_extensions import NamedTuple
from typing_extensions import override
from _typed_dict_test_helper import FooGeneric

# Flags used to mark tests that only apply after a specific
# version of the typing module.
TYPING_3_8_0 = sys.version_info[:3] >= (3, 8, 0)
TYPING_3_9_0 = sys.version_info[:3] >= (3, 9, 0)
TYPING_3_10_0 = sys.version_info[:3] >= (3, 10, 0)

# 3.11 makes runtime type checks (_type_check) more lenient.
TYPING_3_11_0 = sys.version_info[:3] >= (3, 11, 0)


class BaseTestCase(TestCase):
    def assertIsSubclass(self, cls, class_or_tuple, msg=None):
        if not issubclass(cls, class_or_tuple):
            message = f'{cls!r} is not a subclass of {repr(class_or_tuple)}'
            if msg is not None:
                message += f' : {msg}'
            raise self.failureException(message)

    def assertNotIsSubclass(self, cls, class_or_tuple, msg=None):
        if issubclass(cls, class_or_tuple):
            message = f'{cls!r} is a subclass of {repr(class_or_tuple)}'
            if msg is not None:
                message += f' : {msg}'
            raise self.failureException(message)


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
            class A(type(self.bottom_type)):
                pass

    def test_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            self.bottom_type()
        with self.assertRaises(TypeError):
            type(self.bottom_type)()

    def test_pickle(self):
        for proto in range(pickle.HIGHEST_PROTOCOL):
            pickled = pickle.dumps(self.bottom_type, protocol=proto)
            self.assertIs(self.bottom_type, pickle.loads(pickled))


class NoReturnTests(BottomTypeTestsMixin, BaseTestCase):
    bottom_type = NoReturn

    def test_repr(self):
        if hasattr(typing, 'NoReturn'):
            self.assertEqual(repr(NoReturn), 'typing.NoReturn')
        else:
            self.assertEqual(repr(NoReturn), 'typing_extensions.NoReturn')

    def test_get_type_hints(self):
        def some(arg: NoReturn) -> NoReturn: ...
        def some_str(arg: 'NoReturn') -> 'typing.NoReturn': ...

        expected = {'arg': NoReturn, 'return': NoReturn}
        targets = [some]

        # On 3.7.0 and 3.7.1, https://github.com/python/cpython/pull/10772
        # wasn't applied yet and NoReturn fails _type_check.
        if not ((3, 7, 0) <= sys.version_info < (3, 7, 2)):
            targets.append(some_str)
        for target in targets:
            with self.subTest(target=target):
                self.assertEqual(gth(target), expected)

    def test_not_equality(self):
        self.assertNotEqual(NoReturn, Never)
        self.assertNotEqual(Never, NoReturn)


class NeverTests(BottomTypeTestsMixin, BaseTestCase):
    bottom_type = Never

    def test_repr(self):
        if hasattr(typing, 'Never'):
            self.assertEqual(repr(Never), 'typing.Never')
        else:
            self.assertEqual(repr(Never), 'typing_extensions.Never')

    def test_get_type_hints(self):
        def some(arg: Never) -> Never: ...
        def some_str(arg: 'Never') -> 'typing_extensions.Never': ...

        expected = {'arg': Never, 'return': Never}
        for target in [some, some_str]:
            with self.subTest(target=target):
                self.assertEqual(gth(target), expected)


class AssertNeverTests(BaseTestCase):
    def test_exception(self):
        with self.assertRaises(AssertionError):
            assert_never(None)


class OverrideTests(BaseTestCase):
    def test_override(self):
        class Base:
            def foo(self): ...

        class Derived(Base):
            @override
            def foo(self):
                return 42

        self.assertIsSubclass(Derived, Base)
        self.assertEqual(Derived().foo(), 42)
        self.assertEqual(dir(Base.foo), dir(Derived.foo))


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
        if sys.version_info < (3, 11):  # skip for now on 3.11+ see python/cpython#95987
            self.assertEqual(repr(self.SubclassesAny), "<class 'test_typing_extensions.AnyTests.SubclassesAny'>")

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
            class C(type(ClassVar[int])):
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
        if hasattr(typing, 'Final') and sys.version_info[:2] >= (3, 7):
            mod_name = 'typing'
        else:
            mod_name = 'typing_extensions'
        self.assertEqual(repr(Final), mod_name + '.Final')
        cv = Final[int]
        self.assertEqual(repr(cv), mod_name + '.Final[int]')
        cv = Final[Employee]
        self.assertEqual(repr(cv), mod_name + f'.Final[{__name__}.Employee]')

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class C(type(Final)):
                pass
        with self.assertRaises(TypeError):
            class C(type(Final[int])):
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
        self.assertEqual(repr(Required), mod_name + '.Required')
        cv = Required[int]
        self.assertEqual(repr(cv), mod_name + '.Required[int]')
        cv = Required[Employee]
        self.assertEqual(repr(cv), mod_name + '.Required[%s.Employee]' % __name__)

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class C(type(Required)):
                pass
        with self.assertRaises(TypeError):
            class C(type(Required[int])):
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
        self.assertEqual(repr(NotRequired), mod_name + '.NotRequired')
        cv = NotRequired[int]
        self.assertEqual(repr(cv), mod_name + '.NotRequired[int]')
        cv = NotRequired[Employee]
        self.assertEqual(repr(cv), mod_name + '.NotRequired[%s.Employee]' % __name__)

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class C(type(NotRequired)):
                pass
        with self.assertRaises(TypeError):
            class C(type(NotRequired[int])):
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
        T_ints = IntVar("T_ints")  # noqa

    def test_invalid(self):
        with self.assertRaises(TypeError):
            T_ints = IntVar("T_ints", int)
        with self.assertRaises(TypeError):
            T_ints = IntVar("T_ints", bound=int)
        with self.assertRaises(TypeError):
            T_ints = IntVar("T_ints", covariant=True)  # noqa


class LiteralTests(BaseTestCase):
    def test_basics(self):
        Literal[1]
        Literal[1, 2, 3]
        Literal["x", "y", "z"]
        Literal[None]

    def test_illegal_parameters_do_not_raise_runtime_errors(self):
        # Type checkers should reject these types, but we do not
        # raise errors at runtime to maintain maximum flexibility
        Literal[int]
        Literal[Literal[1, 2], Literal[4, 5]]
        Literal[3j + 2, ..., ()]
        Literal[b"foo", u"bar"]
        Literal[{"foo": 3, "bar": 4}]
        Literal[T]

    def test_literals_inside_other_types(self):
        List[Literal[1, 2, 3]]
        List[Literal[("foo", "bar", "baz")]]

    def test_repr(self):
        if hasattr(typing, 'Literal'):
            mod_name = 'typing'
        else:
            mod_name = 'typing_extensions'
        self.assertEqual(repr(Literal[1]), mod_name + ".Literal[1]")
        self.assertEqual(repr(Literal[1, True, "foo"]), mod_name + ".Literal[1, True, 'foo']")
        self.assertEqual(repr(Literal[int]), mod_name + ".Literal[int]")
        self.assertEqual(repr(Literal), mod_name + ".Literal")
        self.assertEqual(repr(Literal[None]), mod_name + ".Literal[None]")

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

@runtime
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

class AnnotatedMovie(TypedDict):
    title: Annotated[Required[str], "foobar"]
    year: NotRequired[Annotated[int, 2000]]


gth = get_type_hints


class GetTypeHintTests(BaseTestCase):
    def test_get_type_hints_modules(self):
        ann_module_type_hints = {1: 2, 'f': Tuple[int, int], 'x': int, 'y': str}
        if (TYPING_3_11_0
                or (TYPING_3_10_0 and sys.version_info.releaselevel in {'candidate', 'final'})):
            # More tests were added in 3.10rc1.
            ann_module_type_hints['u'] = int | float
        self.assertEqual(gth(ann_module), ann_module_type_hints)
        self.assertEqual(gth(ann_module2), {})
        self.assertEqual(gth(ann_module3), {})

    def test_get_type_hints_classes(self):
        self.assertEqual(gth(ann_module.C, ann_module.__dict__),
                         {'y': Optional[ann_module.C]})
        self.assertIsInstance(gth(ann_module.j_class), dict)
        self.assertEqual(gth(ann_module.M), {'123': 123, 'o': type})
        self.assertEqual(gth(ann_module.D),
                         {'j': str, 'k': str, 'y': Optional[ann_module.C]})
        self.assertEqual(gth(ann_module.Y), {'z': int})
        self.assertEqual(gth(ann_module.h_class),
                         {'y': Optional[ann_module.C]})
        self.assertEqual(gth(ann_module.S), {'x': str, 'y': str})
        self.assertEqual(gth(ann_module.foo), {'x': int})
        self.assertEqual(gth(NoneAndForward, globals()),
                         {'parent': NoneAndForward, 'meaning': type(None)})

    def test_respect_no_type_check(self):
        @no_type_check
        class NoTpCheck:
            class Inn:
                def __init__(self, x: 'not a type'): ...  # noqa
        self.assertTrue(NoTpCheck.__no_type_check__)
        self.assertTrue(NoTpCheck.Inn.__init__.__no_type_check__)
        self.assertEqual(gth(ann_module2.NTC.meth), {})
        class ABase(Generic[T]):
            def meth(x: int): ...
        @no_type_check
        class Der(ABase): ...
        self.assertEqual(gth(ABase.meth), {'x': int})

    def test_get_type_hints_ClassVar(self):
        self.assertEqual(gth(ann_module2.CV, ann_module2.__dict__),
                         {'var': ClassVar[ann_module2.CV]})
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
        if sys.version_info >= (3, 9):
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
        if sys.version_info >= (3, 9):
            self.assertEqual(get_args(list[int]), (int,))
        self.assertEqual(get_args(list), ())
        if sys.version_info >= (3, 9):
            # Support Python versions with and without the fix for
            # https://bugs.python.org/issue42195
            # The first variant is for 3.9.2+, the second for 3.9.0 and 1
            self.assertIn(get_args(collections.abc.Callable[[int], str]),
                          (([int], str), ([[int]], str)))
            self.assertIn(get_args(collections.abc.Callable[[], str]),
                          (([], str), ([[]], str)))
            self.assertEqual(get_args(collections.abc.Callable[..., str]), (..., str))
        P = ParamSpec('P')
        # In 3.9 and lower we use typing_extensions's hacky implementation
        # of ParamSpec, which gets incorrectly wrapped in a list
        self.assertIn(get_args(Callable[P, int]), [(P, int), ([P], int)])
        self.assertEqual(get_args(Callable[Concatenate[int, P], int]),
                         (Concatenate[int, P], int))
        self.assertEqual(get_args(Required[int]), (int,))
        self.assertEqual(get_args(NotRequired[int]), (int,))
        self.assertEqual(get_args(Unpack[Ts]), (Ts,))
        self.assertEqual(get_args(Unpack), ())


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
        ns = {}
        exec(
            "async def foo() -> typing_extensions.Awaitable[int]:\n"
            "    return await AwaitableWrapper(42)\n",
            globals(), ns)
        foo = ns['foo']
        g = foo()
        self.assertIsInstance(g, typing_extensions.Awaitable)
        self.assertNotIsInstance(foo, typing_extensions.Awaitable)
        g.send(None)  # Run foo() till completion, to avoid warning.

    def test_coroutine(self):
        ns = {}
        exec(
            "async def foo():\n"
            "    return\n",
            globals(), ns)
        foo = ns['foo']
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
        base_it = range(10)  # type: Iterator[int]
        it = AsyncIteratorWrapper(base_it)
        self.assertIsInstance(it, typing_extensions.AsyncIterable)
        self.assertIsInstance(it, typing_extensions.AsyncIterable)
        self.assertNotIsInstance(42, typing_extensions.AsyncIterable)

    def test_async_iterator(self):
        base_it = range(10)  # type: Iterator[int]
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

    def test_async_generator(self):
        ns = {}
        exec("async def f():\n"
             "    yield 42\n", globals(), ns)
        g = ns['f']()
        self.assertIsSubclass(type(g), typing_extensions.AsyncGenerator)

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

        ns = {}
        exec('async def g(): yield 0', globals(), ns)
        g = ns['g']
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


class OtherABCTests(BaseTestCase):

    def test_contextmanager(self):
        @contextlib.contextmanager
        def manager():
            yield 42

        cm = manager()
        self.assertIsInstance(cm, typing_extensions.ContextManager)
        self.assertNotIsInstance(42, typing_extensions.ContextManager)

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
        self.assertEqual(typing_extensions.AsyncContextManager[int].__args__, (int,))
        with self.assertRaises(TypeError):
            isinstance(42, typing_extensions.AsyncContextManager[int])
        with self.assertRaises(TypeError):
            typing_extensions.AsyncContextManager[int, str]


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

    def test_basic(self):
        UserId = NewType('UserId', int)
        UserName = NewType('UserName', str)
        self.assertIsInstance(UserId(5), int)
        self.assertIsInstance(UserName('Joe'), str)
        self.assertEqual(UserId(5) + 1, 6)

    def test_errors(self):
        UserId = NewType('UserId', int)
        UserName = NewType('UserName', str)
        with self.assertRaises(TypeError):
            issubclass(UserId, int)
        with self.assertRaises(TypeError):
            class D(UserName):
                pass


class Coordinate(Protocol):
    x: int
    y: int

@runtime
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

@runtime
class Position(XAxis, YAxis, Protocol):
    pass

@runtime
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


class ProtocolTests(BaseTestCase):

    def test_basic_protocol(self):
        @runtime
        class P(Protocol):
            def meth(self):
                pass
        class C: pass
        class D:
            def meth(self):
                pass
        def f():
            pass
        self.assertIsSubclass(D, P)
        self.assertIsInstance(D(), P)
        self.assertNotIsSubclass(C, P)
        self.assertNotIsInstance(C(), P)
        self.assertNotIsSubclass(types.FunctionType, P)
        self.assertNotIsInstance(f, P)

    def test_everything_implements_empty_protocol(self):
        @runtime
        class Empty(Protocol): pass
        class C: pass
        def f():
            pass
        for thing in (object, type, tuple, C, types.FunctionType):
            self.assertIsSubclass(thing, Empty)
        for thing in (object(), 1, (), typing, f):
            self.assertIsInstance(thing, Empty)

    def test_function_implements_protocol(self):
        def f():
            pass
        self.assertIsInstance(f, HasCallProtocol)

    def test_no_inheritance_from_nominal(self):
        class C: pass
        class BP(Protocol): pass
        with self.assertRaises(TypeError):
            class P(C, Protocol):
                pass
        with self.assertRaises(TypeError):
            class P(Protocol, C):
                pass
        with self.assertRaises(TypeError):
            class P(BP, C, Protocol):
                pass
        class D(BP, C): pass
        class E(C, BP): pass
        self.assertNotIsInstance(D(), E)
        self.assertNotIsInstance(E(), D)

    def test_no_instantiation(self):
        class P(Protocol): pass
        with self.assertRaises(TypeError):
            P()
        class C(P): pass
        self.assertIsInstance(C(), C)
        T = TypeVar('T')
        class PG(Protocol[T]): pass
        with self.assertRaises(TypeError):
            PG()
        with self.assertRaises(TypeError):
            PG[int]()
        with self.assertRaises(TypeError):
            PG[T]()
        class CG(PG[T]): pass
        self.assertIsInstance(CG[int](), CG)

    def test_cannot_instantiate_abstract(self):
        @runtime
        class P(Protocol):
            @abc.abstractmethod
            def ameth(self) -> int:
                raise NotImplementedError
        class B(P):
            pass
        class C(B):
            def ameth(self) -> int:
                return 26
        with self.assertRaises(TypeError):
            B()
        self.assertIsInstance(C(), P)

    def test_subprotocols_extending(self):
        class P1(Protocol):
            def meth1(self):
                pass
        @runtime
        class P2(P1, Protocol):
            def meth2(self):
                pass
        class C:
            def meth1(self):
                pass
            def meth2(self):
                pass
        class C1:
            def meth1(self):
                pass
        class C2:
            def meth2(self):
                pass
        self.assertNotIsInstance(C1(), P2)
        self.assertNotIsInstance(C2(), P2)
        self.assertNotIsSubclass(C1, P2)
        self.assertNotIsSubclass(C2, P2)
        self.assertIsInstance(C(), P2)
        self.assertIsSubclass(C, P2)

    def test_subprotocols_merging(self):
        class P1(Protocol):
            def meth1(self):
                pass
        class P2(Protocol):
            def meth2(self):
                pass
        @runtime
        class P(P1, P2, Protocol):
            pass
        class C:
            def meth1(self):
                pass
            def meth2(self):
                pass
        class C1:
            def meth1(self):
                pass
        class C2:
            def meth2(self):
                pass
        self.assertNotIsInstance(C1(), P)
        self.assertNotIsInstance(C2(), P)
        self.assertNotIsSubclass(C1, P)
        self.assertNotIsSubclass(C2, P)
        self.assertIsInstance(C(), P)
        self.assertIsSubclass(C, P)

    def test_protocols_issubclass(self):
        T = TypeVar('T')
        @runtime
        class P(Protocol):
            def x(self): ...
        @runtime
        class PG(Protocol[T]):
            def x(self): ...
        class BadP(Protocol):
            def x(self): ...
        class BadPG(Protocol[T]):
            def x(self): ...
        class C:
            def x(self): ...
        self.assertIsSubclass(C, P)
        self.assertIsSubclass(C, PG)
        self.assertIsSubclass(BadP, PG)
        with self.assertRaises(TypeError):
            issubclass(C, PG[T])
        with self.assertRaises(TypeError):
            issubclass(C, PG[C])
        with self.assertRaises(TypeError):
            issubclass(C, BadP)
        with self.assertRaises(TypeError):
            issubclass(C, BadPG)
        with self.assertRaises(TypeError):
            issubclass(P, PG[T])
        with self.assertRaises(TypeError):
            issubclass(PG, PG[int])

    def test_protocols_issubclass_non_callable(self):
        class C:
            x = 1
        @runtime
        class PNonCall(Protocol):
            x = 1
        with self.assertRaises(TypeError):
            issubclass(C, PNonCall)
        self.assertIsInstance(C(), PNonCall)
        PNonCall.register(C)
        with self.assertRaises(TypeError):
            issubclass(C, PNonCall)
        self.assertIsInstance(C(), PNonCall)
        # check that non-protocol subclasses are not affected
        class D(PNonCall): ...
        self.assertNotIsSubclass(C, D)
        self.assertNotIsInstance(C(), D)
        D.register(C)
        self.assertIsSubclass(C, D)
        self.assertIsInstance(C(), D)
        with self.assertRaises(TypeError):
            issubclass(D, PNonCall)

    def test_protocols_isinstance(self):
        T = TypeVar('T')
        @runtime
        class P(Protocol):
            def meth(x): ...
        @runtime
        class PG(Protocol[T]):
            def meth(x): ...
        class BadP(Protocol):
            def meth(x): ...
        class BadPG(Protocol[T]):
            def meth(x): ...
        class C:
            def meth(x): ...
        self.assertIsInstance(C(), P)
        self.assertIsInstance(C(), PG)
        with self.assertRaises(TypeError):
            isinstance(C(), PG[T])
        with self.assertRaises(TypeError):
            isinstance(C(), PG[C])
        with self.assertRaises(TypeError):
            isinstance(C(), BadP)
        with self.assertRaises(TypeError):
            isinstance(C(), BadPG)

    def test_protocols_isinstance_py36(self):
        class APoint:
            def __init__(self, x, y, label):
                self.x = x
                self.y = y
                self.label = label
        class BPoint:
            label = 'B'
            def __init__(self, x, y):
                self.x = x
                self.y = y
        class C:
            def __init__(self, attr):
                self.attr = attr
            def meth(self, arg):
                return 0
        class Bad: pass
        self.assertIsInstance(APoint(1, 2, 'A'), Point)
        self.assertIsInstance(BPoint(1, 2), Point)
        self.assertNotIsInstance(MyPoint(), Point)
        self.assertIsInstance(BPoint(1, 2), Position)
        self.assertIsInstance(Other(), Proto)
        self.assertIsInstance(Concrete(), Proto)
        self.assertIsInstance(C(42), Proto)
        self.assertNotIsInstance(Bad(), Proto)
        self.assertNotIsInstance(Bad(), Point)
        self.assertNotIsInstance(Bad(), Position)
        self.assertNotIsInstance(Bad(), Concrete)
        self.assertNotIsInstance(Other(), Concrete)
        self.assertIsInstance(NT(1, 2), Position)

    def test_protocols_isinstance_init(self):
        T = TypeVar('T')
        @runtime
        class P(Protocol):
            x = 1
        @runtime
        class PG(Protocol[T]):
            x = 1
        class C:
            def __init__(self, x):
                self.x = x
        self.assertIsInstance(C(1), P)
        self.assertIsInstance(C(1), PG)

    def test_protocols_support_register(self):
        @runtime
        class P(Protocol):
            x = 1
        class PM(Protocol):
            def meth(self): pass
        class D(PM): pass
        class C: pass
        D.register(C)
        P.register(C)
        self.assertIsInstance(C(), P)
        self.assertIsInstance(C(), D)

    def test_none_on_non_callable_doesnt_block_implementation(self):
        @runtime
        class P(Protocol):
            x = 1
        class A:
            x = 1
        class B(A):
            x = None
        class C:
            def __init__(self):
                self.x = None
        self.assertIsInstance(B(), P)
        self.assertIsInstance(C(), P)

    def test_none_on_callable_blocks_implementation(self):
        @runtime
        class P(Protocol):
            def x(self): ...
        class A:
            def x(self): ...
        class B(A):
            x = None
        class C:
            def __init__(self):
                self.x = None
        self.assertNotIsInstance(B(), P)
        self.assertNotIsInstance(C(), P)

    def test_non_protocol_subclasses(self):
        class P(Protocol):
            x = 1
        @runtime
        class PR(Protocol):
            def meth(self): pass
        class NonP(P):
            x = 1
        class NonPR(PR): pass
        class C:
            x = 1
        class D:
            def meth(self): pass
        self.assertNotIsInstance(C(), NonP)
        self.assertNotIsInstance(D(), NonPR)
        self.assertNotIsSubclass(C, NonP)
        self.assertNotIsSubclass(D, NonPR)
        self.assertIsInstance(NonPR(), PR)
        self.assertIsSubclass(NonPR, PR)

    def test_custom_subclasshook(self):
        class P(Protocol):
            x = 1
        class OKClass: pass
        class BadClass:
            x = 1
        class C(P):
            @classmethod
            def __subclasshook__(cls, other):
                return other.__name__.startswith("OK")
        self.assertIsInstance(OKClass(), C)
        self.assertNotIsInstance(BadClass(), C)
        self.assertIsSubclass(OKClass, C)
        self.assertNotIsSubclass(BadClass, C)

    def test_issubclass_fails_correctly(self):
        @runtime
        class P(Protocol):
            x = 1
        class C: pass
        with self.assertRaises(TypeError):
            issubclass(C(), P)

    def test_defining_generic_protocols(self):
        T = TypeVar('T')
        S = TypeVar('S')
        @runtime
        class PR(Protocol[T, S]):
            def meth(self): pass
        class P(PR[int, T], Protocol[T]):
            y = 1
        with self.assertRaises(TypeError):
            issubclass(PR[int, T], PR)
        with self.assertRaises(TypeError):
            issubclass(P[str], PR)
        with self.assertRaises(TypeError):
            PR[int]
        with self.assertRaises(TypeError):
            P[int, str]
        if not TYPING_3_10_0:
            with self.assertRaises(TypeError):
                PR[int, 1]
            with self.assertRaises(TypeError):
                PR[int, ClassVar]
        class C(PR[int, T]): pass
        self.assertIsInstance(C[str](), C)

    def test_defining_generic_protocols_old_style(self):
        T = TypeVar('T')
        S = TypeVar('S')
        @runtime
        class PR(Protocol, Generic[T, S]):
            def meth(self): pass
        class P(PR[int, str], Protocol):
            y = 1
        with self.assertRaises(TypeError):
            self.assertIsSubclass(PR[int, str], PR)
        self.assertIsSubclass(P, PR)
        with self.assertRaises(TypeError):
            PR[int]
        if not TYPING_3_10_0:
            with self.assertRaises(TypeError):
                PR[int, 1]
        class P1(Protocol, Generic[T]):
            def bar(self, x: T) -> str: ...
        class P2(Generic[T], Protocol):
            def bar(self, x: T) -> str: ...
        @runtime
        class PSub(P1[str], Protocol):
            x = 1
        class Test:
            x = 1
            def bar(self, x: str) -> str:
                return x
        self.assertIsInstance(Test(), PSub)
        if not TYPING_3_10_0:
            with self.assertRaises(TypeError):
                PR[int, ClassVar]

    def test_init_called(self):
        T = TypeVar('T')
        class P(Protocol[T]): pass
        class C(P[T]):
            def __init__(self):
                self.test = 'OK'
        self.assertEqual(C[int]().test, 'OK')

    def test_protocols_bad_subscripts(self):
        T = TypeVar('T')
        S = TypeVar('S')
        with self.assertRaises(TypeError):
            class P(Protocol[T, T]): pass
        with self.assertRaises(TypeError):
            class P(Protocol[int]): pass
        with self.assertRaises(TypeError):
            class P(Protocol[T], Protocol[S]): pass
        with self.assertRaises(TypeError):
            class P(typing.Mapping[T, S], Protocol[T]): pass

    def test_generic_protocols_repr(self):
        T = TypeVar('T')
        S = TypeVar('S')
        class P(Protocol[T, S]): pass
        self.assertTrue(repr(P[T, S]).endswith('P[~T, ~S]'))
        self.assertTrue(repr(P[int, str]).endswith('P[int, str]'))

    def test_generic_protocols_eq(self):
        T = TypeVar('T')
        S = TypeVar('S')
        class P(Protocol[T, S]): pass
        self.assertEqual(P, P)
        self.assertEqual(P[int, T], P[int, T])
        self.assertEqual(P[T, T][Tuple[T, S]][int, str],
                         P[Tuple[int, str], Tuple[int, str]])

    def test_generic_protocols_special_from_generic(self):
        T = TypeVar('T')
        class P(Protocol[T]): pass
        self.assertEqual(P.__parameters__, (T,))
        self.assertEqual(P[int].__parameters__, ())
        self.assertEqual(P[int].__args__, (int,))
        self.assertIs(P[int].__origin__, P)

    def test_generic_protocols_special_from_protocol(self):
        @runtime
        class PR(Protocol):
            x = 1
        class P(Protocol):
            def meth(self):
                pass
        T = TypeVar('T')
        class PG(Protocol[T]):
            x = 1
            def meth(self):
                pass
        self.assertTrue(P._is_protocol)
        self.assertTrue(PR._is_protocol)
        self.assertTrue(PG._is_protocol)
        if hasattr(typing, 'Protocol'):
            self.assertFalse(P._is_runtime_protocol)
        else:
            with self.assertRaises(AttributeError):
                self.assertFalse(P._is_runtime_protocol)
        self.assertTrue(PR._is_runtime_protocol)
        self.assertTrue(PG[int]._is_protocol)
        self.assertEqual(typing_extensions._get_protocol_attrs(P), {'meth'})
        self.assertEqual(typing_extensions._get_protocol_attrs(PR), {'x'})
        self.assertEqual(frozenset(typing_extensions._get_protocol_attrs(PG)),
                         frozenset({'x', 'meth'}))

    def test_no_runtime_deco_on_nominal(self):
        with self.assertRaises(TypeError):
            @runtime
            class C: pass
        class Proto(Protocol):
            x = 1
        with self.assertRaises(TypeError):
            @runtime
            class Concrete(Proto):
                pass

    def test_none_treated_correctly(self):
        @runtime
        class P(Protocol):
            x = None  # type: int
        class B(object): pass
        self.assertNotIsInstance(B(), P)
        class C:
            x = 1
        class D:
            x = None
        self.assertIsInstance(C(), P)
        self.assertIsInstance(D(), P)
        class CI:
            def __init__(self):
                self.x = 1
        class DI:
            def __init__(self):
                self.x = None
        self.assertIsInstance(C(), P)
        self.assertIsInstance(D(), P)

    def test_protocols_in_unions(self):
        class P(Protocol):
            x = None  # type: int
        Alias = typing.Union[typing.Iterable, P]
        Alias2 = typing.Union[P, typing.Iterable]
        self.assertEqual(Alias, Alias2)

    def test_protocols_pickleable(self):
        global P, CP  # pickle wants to reference the class by name
        T = TypeVar('T')

        @runtime
        class P(Protocol[T]):
            x = 1
        class CP(P[int]):
            pass

        c = CP()
        c.foo = 42
        c.bar = 'abc'
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            z = pickle.dumps(c, proto)
            x = pickle.loads(z)
            self.assertEqual(x.foo, 42)
            self.assertEqual(x.bar, 'abc')
            self.assertEqual(x.x, 1)
            self.assertEqual(x.__dict__, {'foo': 42, 'bar': 'abc'})
            s = pickle.dumps(P)
            D = pickle.loads(s)
            class E:
                x = 1
            self.assertIsInstance(E(), D)

    def test_collections_protocols_allowed(self):
        @runtime_checkable
        class Custom(collections.abc.Iterable, Protocol):
            def close(self): pass

        class A: ...
        class B:
            def __iter__(self):
                return []
            def close(self):
                return 0

        self.assertIsSubclass(B, Custom)
        self.assertNotIsSubclass(A, Custom)

    def test_no_init_same_for_different_protocol_implementations(self):
        class CustomProtocolWithoutInitA(Protocol):
            pass

        class CustomProtocolWithoutInitB(Protocol):
            pass

        self.assertEqual(CustomProtocolWithoutInitA.__init__, CustomProtocolWithoutInitB.__init__)


class Point2DGeneric(Generic[T], TypedDict):
    a: T
    b: T


class BarGeneric(FooGeneric[T], total=False):
    b: int


class TypedDictTests(BaseTestCase):

    def test_basics_iterable_syntax(self):
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

    def test_basics_keywords_syntax(self):
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

    def test_typeddict_special_keyword_names(self):
        TD = TypedDict("TD", cls=type, self=object, typename=str, _typename=int,
                       fields=list, _fields=dict)
        self.assertEqual(TD.__name__, 'TD')
        self.assertEqual(TD.__annotations__, {'cls': type, 'self': object, 'typename': str,
                                              '_typename': int, 'fields': list, '_fields': dict})
        a = TD(cls=str, self=42, typename='foo', _typename=53,
               fields=[('bar', tuple)], _fields={'baz', set})
        self.assertEqual(a['cls'], str)
        self.assertEqual(a['self'], 42)
        self.assertEqual(a['typename'], 'foo')
        self.assertEqual(a['_typename'], 53)
        self.assertEqual(a['fields'], [('bar', tuple)])
        self.assertEqual(a['_fields'], {'baz', set})

    @skipIf(hasattr(typing, 'TypedDict'), "Should be tested by upstream")
    def test_typeddict_create_errors(self):
        with self.assertRaises(TypeError):
            TypedDict.__new__()
        with self.assertRaises(TypeError):
            TypedDict()
        with self.assertRaises(TypeError):
            TypedDict('Emp', [('name', str)], None)

        with self.assertWarns(DeprecationWarning):
            Emp = TypedDict(_typename='Emp', name=str, id=int)
        self.assertEqual(Emp.__name__, 'Emp')
        self.assertEqual(Emp.__annotations__, {'name': str, 'id': int})

        with self.assertWarns(DeprecationWarning):
            Emp = TypedDict('Emp', _fields={'name': str, 'id': int})
        self.assertEqual(Emp.__name__, 'Emp')
        self.assertEqual(Emp.__annotations__, {'name': str, 'id': int})

    def test_typeddict_errors(self):
        Emp = TypedDict('Emp', {'name': str, 'id': int})
        if hasattr(typing, "Required"):
            self.assertEqual(TypedDict.__module__, 'typing')
        else:
            self.assertEqual(TypedDict.__module__, 'typing_extensions')
        jim = Emp(name='Jim', id=1)
        with self.assertRaises(TypeError):
            isinstance({}, Emp)
        with self.assertRaises(TypeError):
            isinstance(jim, Emp)
        with self.assertRaises(TypeError):
            issubclass(dict, Emp)

        if not TYPING_3_11_0:
            with self.assertRaises(TypeError):
                TypedDict('Hi', x=1)
            with self.assertRaises(TypeError):
                TypedDict('Hi', [('x', int), ('y', 1)])
        with self.assertRaises(TypeError):
            TypedDict('Hi', [('x', int)], y=int)

    def test_py36_class_syntax_usage(self):
        self.assertEqual(LabelPoint2D.__name__, 'LabelPoint2D')
        self.assertEqual(LabelPoint2D.__module__, __name__)
        self.assertEqual(get_type_hints(LabelPoint2D), {'x': int, 'y': int, 'label': str})
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
        EmpD = TypedDict('EmpD', name=str, id=int)
        jane = EmpD({'name': 'jane', 'id': 37})
        point = Point2DGeneric(a=5.0, b=3.0)
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            # Test non-generic TypedDict
            z = pickle.dumps(jane, proto)
            jane2 = pickle.loads(z)
            self.assertEqual(jane2, jane)
            self.assertEqual(jane2, {'name': 'jane', 'id': 37})
            ZZ = pickle.dumps(EmpD, proto)
            EmpDnew = pickle.loads(ZZ)
            self.assertEqual(EmpDnew({'name': 'jane', 'id': 37}), jane)
            # and generic TypedDict
            y = pickle.dumps(point, proto)
            point2 = pickle.loads(y)
            self.assertEqual(point, point2)
            self.assertEqual(point2, {'a': 5.0, 'b': 3.0})
            YY = pickle.dumps(Point2DGeneric, proto)
            Point2DGenericNew = pickle.loads(YY)
            self.assertEqual(Point2DGenericNew({'a': 5.0, 'b': 3.0}), point)

    def test_optional(self):
        EmpD = TypedDict('EmpD', name=str, id=int)

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

    def test_optional_keys(self):
        assert Point2Dor3D.__required_keys__ == frozenset(['x', 'y'])
        assert Point2Dor3D.__optional_keys__ == frozenset(['z'])

    def test_required_notrequired_keys(self):
        assert NontotalMovie.__required_keys__ == frozenset({'title'})
        assert NontotalMovie.__optional_keys__ == frozenset({'year'})

        assert TotalMovie.__required_keys__ == frozenset({'title'})
        assert TotalMovie.__optional_keys__ == frozenset({'year'})


    def test_keys_inheritance(self):
        assert BaseAnimal.__required_keys__ == frozenset(['name'])
        assert BaseAnimal.__optional_keys__ == frozenset([])
        assert get_type_hints(BaseAnimal) == {'name': str}

        assert Animal.__required_keys__ == frozenset(['name'])
        assert Animal.__optional_keys__ == frozenset(['tail', 'voice'])
        assert get_type_hints(Animal) == {
            'name': str,
            'tail': bool,
            'voice': str,
        }

        assert Cat.__required_keys__ == frozenset(['name', 'fur_color'])
        assert Cat.__optional_keys__ == frozenset(['tail', 'voice'])
        assert get_type_hints(Cat) == {
            'fur_color': str,
            'name': str,
            'tail': bool,
            'voice': str,
        }

    def test_is_typeddict(self):
        assert is_typeddict(Point2D) is True
        assert is_typeddict(Point2Dor3D) is True
        assert is_typeddict(Union[str, int]) is False
        # classes, not instances
        assert is_typeddict(Point2D()) is False

    @skipUnless(TYPING_3_8_0, "Python 3.8+ required")
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
        assert Point3D.__annotations__ == {
            'a': T,
            'b': T,
            'c': KT,
        }
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

    @skipIf(sys.version_info[:2] in ((3, 9), (3, 10)), "Waiting for bpo-46491 bugfix.")
    def test_special_form_containment(self):
        class C:
            classvar: Annotated[ClassVar[int], "a decoration"] = 4
            const: Annotated[Final[int], "Const"] = 4

        if sys.version_info[:2] >= (3, 7):
            self.assertEqual(get_type_hints(C, globals())["classvar"], ClassVar[int])
            self.assertEqual(get_type_hints(C, globals())["const"], Final[int])
        else:
            self.assertEqual(
                get_type_hints(C, globals())["classvar"],
                Annotated[ClassVar[int], "a decoration"]
            )
            self.assertEqual(
                get_type_hints(C, globals())["const"], Annotated[Final[int], "Const"]
            )

    def test_hash_eq(self):
        self.assertEqual(len({Annotated[int, 4, 5], Annotated[int, 4, 5]}), 1)
        self.assertNotEqual(Annotated[int, 4, 5], Annotated[int, 5, 4])
        self.assertNotEqual(Annotated[int, 4, 5], Annotated[str, 4, 5])
        self.assertNotEqual(Annotated[int, 4], Annotated[int, 4, 4])
        self.assertEqual(
            {Annotated[int, 4, 5], Annotated[int, 4, 5], Annotated[T, 4, 5]},
            {Annotated[int, 4, 5], Annotated[T, 4, 5]}
        )

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


class TypeAliasTests(BaseTestCase):
    def test_canonical_usage_with_variable_annotation(self):
        ns = {}
        exec('Alias: TypeAlias = Employee', globals(), ns)

    def test_canonical_usage_with_type_comment(self):
        Alias = Employee  # type: TypeAlias

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
            class C(type(TypeAlias)):
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
        # Should be hashable
        hash(P)

    def test_repr(self):
        P = ParamSpec('P')
        P_co = ParamSpec('P_co', covariant=True)
        P_contra = ParamSpec('P_contra', contravariant=True)
        P_2 = ParamSpec('P_2')
        self.assertEqual(repr(P), '~P')
        self.assertEqual(repr(P_2), '~P_2')

        # Note: PEP 612 doesn't require these to be repr-ed correctly, but
        # just follow CPython.
        self.assertEqual(repr(P_co), '+P_co')
        self.assertEqual(repr(P_contra), '-P_contra')

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
        if sys.version_info[:2] >= (3, 9):
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

    @skipIf((3, 10, 0) <= sys.version_info[:3] <= (3, 10, 2), "Needs bpo-46676.")
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

        G1 = X[int, P_2]
        self.assertEqual(G1.__args__, (int, P_2))
        self.assertEqual(G1.__parameters__, (P_2,))

        G2 = X[int, Concatenate[int, P_2]]
        self.assertEqual(G2.__args__, (int, Concatenate[int, P_2]))
        self.assertEqual(G2.__parameters__, (P_2,))

        # The following are some valid uses cases in PEP 612 that don't work:
        # These do not work in 3.9, _type_check blocks the list and ellipsis.
        # G3 = X[int, [int, bool]]
        # G4 = X[int, ...]
        # G5 = Z[[int, str, bool]]
        # Not working because this is special-cased in 3.10.
        # G6 = Z[int, str, bool]

        class Z(Generic[P]):
            pass

    def test_pickle(self):
        global P, P_co, P_contra, P_default
        P = ParamSpec('P')
        P_co = ParamSpec('P_co', covariant=True)
        P_contra = ParamSpec('P_contra', contravariant=True)
        P_default = ParamSpec('P_default', default=int)
        for proto in range(pickle.HIGHEST_PROTOCOL):
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


class ConcatenateTests(BaseTestCase):
    def test_basics(self):
        P = ParamSpec('P')

        class MyClass: ...

        c = Concatenate[MyClass, P]
        self.assertNotEqual(c, Concatenate)

    def test_valid_uses(self):
        P = ParamSpec('P')
        T = TypeVar('T')

        C1 = Callable[Concatenate[int, P], int]
        C2 = Callable[Concatenate[int, T, P], T]

        # Test collections.abc.Callable too.
        if sys.version_info[:2] >= (3, 9):
            C3 = collections.abc.Callable[Concatenate[int, P], int]
            C4 = collections.abc.Callable[Concatenate[int, T, P], T]

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
            'The last parameter to Concatenate should be a ParamSpec variable',
        ):
            Concatenate[P, T]

        if not TYPING_3_11_0:
            with self.assertRaisesRegex(
                TypeError,
                'each arg must be a type',
            ):
                Concatenate[1, P]

    def test_basic_introspection(self):
        P = ParamSpec('P')
        C1 = Concatenate[int, P]
        C2 = Concatenate[int, T, P]
        self.assertEqual(C1.__origin__, Concatenate)
        self.assertEqual(C1.__args__, (int, P))
        self.assertEqual(C2.__origin__, Concatenate)
        self.assertEqual(C2.__args__, (int, T, P))

    def test_eq(self):
        P = ParamSpec('P')
        C1 = Concatenate[int, P]
        C2 = Concatenate[int, P]
        C3 = Concatenate[int, T, P]
        self.assertEqual(C1, C2)
        self.assertEqual(hash(C1), hash(C2))
        self.assertNotEqual(C1, C3)


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
            class C(type(TypeGuard[int])):
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


class LiteralStringTests(BaseTestCase):
    def test_basics(self):
        class Foo:
            def bar(self) -> LiteralString: ...
            def baz(self) -> "LiteralString": ...

        self.assertEqual(gth(Foo.bar), {'return': LiteralString})
        self.assertEqual(gth(Foo.baz), {'return': LiteralString})

    def test_get_origin(self):
        self.assertIsNone(get_origin(LiteralString))

    def test_repr(self):
        if hasattr(typing, 'LiteralString'):
            mod_name = 'typing'
        else:
            mod_name = 'typing_extensions'
        self.assertEqual(repr(LiteralString), '{}.LiteralString'.format(mod_name))

    def test_cannot_subscript(self):
        with self.assertRaises(TypeError):
            LiteralString[int]

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class C(type(LiteralString)):
                pass
        with self.assertRaises(TypeError):
            class C(LiteralString):
                pass

    def test_cannot_init(self):
        with self.assertRaises(TypeError):
            LiteralString()
        with self.assertRaises(TypeError):
            type(LiteralString)()

    def test_no_isinstance(self):
        with self.assertRaises(TypeError):
            isinstance(1, LiteralString)
        with self.assertRaises(TypeError):
            issubclass(int, LiteralString)

    def test_alias(self):
        StringTuple = Tuple[LiteralString, LiteralString]
        class Alias:
            def return_tuple(self) -> StringTuple:
                return ("foo", "pep" + "675")

    def test_typevar(self):
        StrT = TypeVar("StrT", bound=LiteralString)
        self.assertIs(StrT.__bound__, LiteralString)

    def test_pickle(self):
        for proto in range(pickle.HIGHEST_PROTOCOL):
            pickled = pickle.dumps(LiteralString, protocol=proto)
            self.assertIs(LiteralString, pickle.loads(pickled))


class SelfTests(BaseTestCase):
    def test_basics(self):
        class Foo:
            def bar(self) -> Self: ...

        self.assertEqual(gth(Foo.bar), {'return': Self})

    def test_repr(self):
        if hasattr(typing, 'Self'):
            mod_name = 'typing'
        else:
            mod_name = 'typing_extensions'
        self.assertEqual(repr(Self), '{}.Self'.format(mod_name))

    def test_cannot_subscript(self):
        with self.assertRaises(TypeError):
            Self[int]

    def test_cannot_subclass(self):
        with self.assertRaises(TypeError):
            class C(type(Self)):
                pass

    def test_cannot_init(self):
        with self.assertRaises(TypeError):
            Self()
        with self.assertRaises(TypeError):
            type(Self)()

    def test_no_isinstance(self):
        with self.assertRaises(TypeError):
            isinstance(1, Self)
        with self.assertRaises(TypeError):
            issubclass(int, Self)

    def test_alias(self):
        TupleSelf = Tuple[Self, Self]
        class Alias:
            def return_tuple(self) -> TupleSelf:
                return (self, self)

    def test_pickle(self):
        for proto in range(pickle.HIGHEST_PROTOCOL):
            pickled = pickle.dumps(Self, protocol=proto)
            self.assertIs(Self, pickle.loads(pickled))


class UnpackTests(BaseTestCase):
    def test_basic_plain(self):
        Ts = TypeVarTuple('Ts')
        self.assertEqual(Unpack[Ts], Unpack[Ts])
        with self.assertRaises(TypeError):
            Unpack()

    def test_repr(self):
        Ts = TypeVarTuple('Ts')
        if TYPING_3_11_0:
            self.assertEqual(repr(Unpack[Ts]), '*Ts')
        else:
            self.assertEqual(repr(Unpack[Ts]), 'typing_extensions.Unpack[Ts]')

    def test_cannot_subclass_vars(self):
        with self.assertRaises(TypeError):
            class V(Unpack[TypeVarTuple('Ts')]):
                pass

    def test_tuple(self):
        Ts = TypeVarTuple('Ts')
        Tuple[Unpack[Ts]]

    def test_union(self):
        Xs = TypeVarTuple('Xs')
        Ys = TypeVarTuple('Ys')
        self.assertEqual(
            Union[Unpack[Xs]],
            Unpack[Xs]
        )
        self.assertNotEqual(
            Union[Unpack[Xs]],
            Union[Unpack[Xs], Unpack[Ys]]
        )
        self.assertEqual(
            Union[Unpack[Xs], Unpack[Xs]],
            Unpack[Xs]
        )
        self.assertNotEqual(
            Union[Unpack[Xs], int],
            Union[Unpack[Xs]]
        )
        self.assertNotEqual(
            Union[Unpack[Xs], int],
            Union[int]
        )
        self.assertEqual(
            Union[Unpack[Xs], int].__args__,
            (Unpack[Xs], int)
        )
        self.assertEqual(
            Union[Unpack[Xs], int].__parameters__,
            (Xs,)
        )
        self.assertIs(
            Union[Unpack[Xs], int].__origin__,
            Union
        )

    def test_concatenation(self):
        Xs = TypeVarTuple('Xs')
        self.assertEqual(Tuple[int, Unpack[Xs]].__args__, (int, Unpack[Xs]))
        self.assertEqual(Tuple[Unpack[Xs], int].__args__, (Unpack[Xs], int))
        self.assertEqual(Tuple[int, Unpack[Xs], str].__args__,
                         (int, Unpack[Xs], str))
        class C(Generic[Unpack[Xs]]): pass
        self.assertEqual(C[int, Unpack[Xs]].__args__, (int, Unpack[Xs]))
        self.assertEqual(C[Unpack[Xs], int].__args__, (Unpack[Xs], int))
        self.assertEqual(C[int, Unpack[Xs], str].__args__,
                         (int, Unpack[Xs], str))

    def test_class(self):
        Ts = TypeVarTuple('Ts')

        class C(Generic[Unpack[Ts]]): pass
        self.assertEqual(C[int].__args__, (int,))
        self.assertEqual(C[int, str].__args__, (int, str))

        with self.assertRaises(TypeError):
            class C(Generic[Unpack[Ts], int]): pass

        T1 = TypeVar('T')
        T2 = TypeVar('T')
        class C(Generic[T1, T2, Unpack[Ts]]): pass
        self.assertEqual(C[int, str].__args__, (int, str))
        self.assertEqual(C[int, str, float].__args__, (int, str, float))
        self.assertEqual(C[int, str, float, bool].__args__, (int, str, float, bool))
        # TODO This should probably also fail on 3.11, pending changes to CPython.
        if not TYPING_3_11_0:
            with self.assertRaises(TypeError):
                C[int]


class TypeVarTupleTests(BaseTestCase):

    def test_basic_plain(self):
        Ts = TypeVarTuple('Ts')
        self.assertEqual(Ts, Ts)
        self.assertIsInstance(Ts, TypeVarTuple)
        Xs = TypeVarTuple('Xs')
        Ys = TypeVarTuple('Ys')
        self.assertNotEqual(Xs, Ys)

    def test_repr(self):
        Ts = TypeVarTuple('Ts')
        self.assertEqual(repr(Ts), 'Ts')

    def test_no_redefinition(self):
        self.assertNotEqual(TypeVarTuple('Ts'), TypeVarTuple('Ts'))

    def test_cannot_subclass_vars(self):
        with self.assertRaises(TypeError):
            class V(TypeVarTuple('Ts')):
                pass

    def test_cannot_subclass_var_itself(self):
        with self.assertRaises(TypeError):
            class V(TypeVarTuple):
                pass

    def test_cannot_instantiate_vars(self):
        Ts = TypeVarTuple('Ts')
        with self.assertRaises(TypeError):
            Ts()

    def test_tuple(self):
        Ts = TypeVarTuple('Ts')
        # Not legal at type checking time but we can't really check against it.
        Tuple[Ts]

    def test_args_and_parameters(self):
        Ts = TypeVarTuple('Ts')

        t = Tuple[tuple(Ts)]
        self.assertEqual(t.__args__, (Unpack[Ts],))
        self.assertEqual(t.__parameters__, (Ts,))

    def test_pickle(self):
        global Ts, Ts_default  # pickle wants to reference the class by name
        Ts = TypeVarTuple('Ts')
        Ts_default = TypeVarTuple('Ts_default', default=Unpack[Tuple[int, str]])

        for proto in range(pickle.HIGHEST_PROTOCOL):
            for typevartuple in (Ts, Ts_default):
                z = pickle.loads(pickle.dumps(typevartuple, proto))
                self.assertEqual(z.__name__, typevartuple.__name__)
                self.assertEqual(z.__default__, typevartuple.__default__)


class FinalDecoratorTests(BaseTestCase):
    def test_final_unmodified(self):
        def func(x): ...
        self.assertIs(func, final(func))

    def test_dunder_final(self):
        @final
        def func(): ...
        @final
        class Cls: ...
        self.assertIs(True, func.__final__)
        self.assertIs(True, Cls.__final__)

        class Wrapper:
            __slots__ = ("func",)
            def __init__(self, func):
                self.func = func
            def __call__(self, *args, **kwargs):
                return self.func(*args, **kwargs)

        # Check that no error is thrown if the attribute
        # is not writable.
        @final
        @Wrapper
        def wrapped(): ...
        self.assertIsInstance(wrapped, Wrapper)
        self.assertIs(False, hasattr(wrapped, "__final__"))

        class Meta(type):
            @property
            def __final__(self): return "can't set me"
        @final
        class WithMeta(metaclass=Meta): ...
        self.assertEqual(WithMeta.__final__, "can't set me")

        # Builtin classes throw TypeError if you try to set an
        # attribute.
        final(int)
        self.assertIs(False, hasattr(int, "__final__"))

        # Make sure it works with common builtin decorators
        class Methods:
            @final
            @classmethod
            def clsmethod(cls): ...

            @final
            @staticmethod
            def stmethod(): ...

            # The other order doesn't work because property objects
            # don't allow attribute assignment.
            @property
            @final
            def prop(self): ...

            @final
            @lru_cache()  # noqa: B019
            def cached(self): ...

        # Use getattr_static because the descriptor returns the
        # underlying function, which doesn't have __final__.
        self.assertIs(
            True,
            inspect.getattr_static(Methods, "clsmethod").__final__
        )
        self.assertIs(
            True,
            inspect.getattr_static(Methods, "stmethod").__final__
        )
        self.assertIs(True, Methods.prop.fget.__final__)
        self.assertIs(True, Methods.cached.__final__)


class RevealTypeTests(BaseTestCase):
    def test_reveal_type(self):
        obj = object()
        self.assertIs(obj, reveal_type(obj))


class DataclassTransformTests(BaseTestCase):
    def test_decorator(self):
        def create_model(*, frozen: bool = False, kw_only: bool = True):
            return lambda cls: cls

        decorated = dataclass_transform(kw_only_default=True, order_default=False)(create_model)

        class CustomerModel:
            id: int

        self.assertIs(decorated, create_model)
        self.assertEqual(
            decorated.__dataclass_transform__,
            {
                "eq_default": True,
                "order_default": False,
                "kw_only_default": True,
                "field_specifiers": (),
                "kwargs": {},
            }
        )
        self.assertIs(
            decorated(frozen=True, kw_only=False)(CustomerModel),
            CustomerModel
        )

    def test_base_class(self):
        class ModelBase:
            def __init_subclass__(cls, *, frozen: bool = False): ...

        Decorated = dataclass_transform(
            eq_default=True,
            order_default=True,
            # Arbitrary unrecognized kwargs are accepted at runtime.
            make_everything_awesome=True,
        )(ModelBase)

        class CustomerModel(Decorated, frozen=True):
            id: int

        self.assertIs(Decorated, ModelBase)
        self.assertEqual(
            Decorated.__dataclass_transform__,
            {
                "eq_default": True,
                "order_default": True,
                "kw_only_default": False,
                "field_specifiers": (),
                "kwargs": {"make_everything_awesome": True},
            }
        )
        self.assertIsSubclass(CustomerModel, Decorated)

    def test_metaclass(self):
        class Field: ...

        class ModelMeta(type):
            def __new__(
                cls, name, bases, namespace, *, init: bool = True,
            ):
                return super().__new__(cls, name, bases, namespace)

        Decorated = dataclass_transform(
            order_default=True, field_specifiers=(Field,)
        )(ModelMeta)

        class ModelBase(metaclass=Decorated): ...

        class CustomerModel(ModelBase, init=False):
            id: int

        self.assertIs(Decorated, ModelMeta)
        self.assertEqual(
            Decorated.__dataclass_transform__,
            {
                "eq_default": True,
                "order_default": True,
                "kw_only_default": False,
                "field_specifiers": (Field,),
                "kwargs": {},
            }
        )
        self.assertIsInstance(CustomerModel, Decorated)


class AllTests(BaseTestCase):

    def test_typing_extensions_includes_standard(self):
        a = typing_extensions.__all__
        self.assertIn('ClassVar', a)
        self.assertIn('Type', a)
        self.assertIn('ChainMap', a)
        self.assertIn('ContextManager', a)
        self.assertIn('Counter', a)
        self.assertIn('DefaultDict', a)
        self.assertIn('Deque', a)
        self.assertIn('NewType', a)
        self.assertIn('overload', a)
        self.assertIn('Text', a)
        self.assertIn('TYPE_CHECKING', a)
        self.assertIn('TypeAlias', a)
        self.assertIn('ParamSpec', a)
        self.assertIn("Concatenate", a)

        self.assertIn('Annotated', a)
        self.assertIn('get_type_hints', a)

        self.assertIn('Awaitable', a)
        self.assertIn('AsyncIterator', a)
        self.assertIn('AsyncIterable', a)
        self.assertIn('Coroutine', a)
        self.assertIn('AsyncContextManager', a)

        self.assertIn('AsyncGenerator', a)

        self.assertIn('Protocol', a)
        self.assertIn('runtime', a)

        # Check that all objects in `__all__` are present in the module
        for name in a:
            self.assertTrue(hasattr(typing_extensions, name))

    def test_all_names_in___all__(self):
        exclude = {
            'GenericMeta',
            'KT',
            'PEP_560',
            'T',
            'T_co',
            'T_contra',
            'VT',
        }
        actual_names = {
            name for name in dir(typing_extensions)
            if not name.startswith("_")
            and not isinstance(getattr(typing_extensions, name), types.ModuleType)
        }
        # Make sure all public names are in __all__
        self.assertEqual({*exclude, *typing_extensions.__all__},
                         actual_names)
        # Make sure all excluded names actually exist
        self.assertLessEqual(exclude, actual_names)

    def test_typing_extensions_defers_when_possible(self):
        exclude = {
            'overload',
            'ParamSpec',
            'Text',
            'TypedDict',
            'TypeVar',
            'TypeVarTuple',
            'TYPE_CHECKING',
            'Final',
            'get_type_hints',
            'is_typeddict',
        }
        if sys.version_info < (3, 10):
            exclude |= {'get_args', 'get_origin'}
        if sys.version_info < (3, 11):
            exclude |= {'final', 'NamedTuple', 'Any'}
        for item in typing_extensions.__all__:
            if item not in exclude and hasattr(typing, item):
                self.assertIs(
                    getattr(typing_extensions, item),
                    getattr(typing, item))

    def test_typing_extensions_compiles_with_opt(self):
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 'typing_extensions.py')
        try:
            subprocess.check_output(f'{sys.executable} -OO {file_path}',
                                    stderr=subprocess.STDOUT,
                                    shell=True)
        except subprocess.CalledProcessError:
            self.fail('Module does not compile with optimize=2 (-OO flag).')


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


class XRepr(NamedTuple):
    x: int
    y: int = 1

    def __str__(self):
        return f'{self.x} -> {self.y}'

    def __add__(self, other):
        return 0


@skipIf(TYPING_3_11_0, "These invariants should all be tested upstream on 3.11+")
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

    @skipUnless(
        (
            TYPING_3_8_0
            or hasattr(CoolEmployeeWithDefault, '_field_defaults')
        ),
        '"_field_defaults" attribute was added in a micro version of 3.7'
    )
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
            class X(NamedTuple, tuple):
                x: int

        with self.assertRaisesRegex(TypeError, 'duplicate base class'):
            class X(NamedTuple, NamedTuple):
                x: int

        class A(NamedTuple):
            x: int
        with self.assertRaisesRegex(
            TypeError,
            'can only inherit from a NamedTuple type and Generic'
        ):
            class X(NamedTuple, A):
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
                self.assertEqual(a.x, 3)

                with self.assertRaisesRegex(TypeError, 'Too many parameters'):
                    G[int, str]

    @skipUnless(TYPING_3_9_0, "tuple.__class_getitem__ was added in 3.9")
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

    @skipIf(TYPING_3_9_0, "Test isn't relevant to 3.9+")
    def test_non_generic_subscript_error_message_py38_minus(self):
        class Group(NamedTuple):
            key: T
            group: List[T]

        with self.assertRaisesRegex(TypeError, 'not subscriptable'):
            Group[int]

        for attr in ('__args__', '__origin__', '__parameters__'):
            with self.subTest(attr=attr):
                self.assertFalse(hasattr(Group, attr))

    def test_namedtuple_keyword_usage(self):
        LocalEmployee = NamedTuple("LocalEmployee", name=str, age=int)
        nick = LocalEmployee('Nick', 25)
        self.assertIsInstance(nick, tuple)
        self.assertEqual(nick.name, 'Nick')
        self.assertEqual(LocalEmployee.__name__, 'LocalEmployee')
        self.assertEqual(LocalEmployee._fields, ('name', 'age'))
        self.assertEqual(LocalEmployee.__annotations__, dict(name=str, age=int))
        with self.assertRaisesRegex(
            TypeError,
            'Either list of fields or keywords can be provided to NamedTuple, not both'
        ):
            NamedTuple('Name', [('x', int)], y=str)

    def test_namedtuple_special_keyword_names(self):
        NT = NamedTuple("NT", cls=type, self=object, typename=str, fields=list)
        self.assertEqual(NT.__name__, 'NT')
        self.assertEqual(NT._fields, ('cls', 'self', 'typename', 'fields'))
        a = NT(cls=str, self=42, typename='foo', fields=[('bar', tuple)])
        self.assertEqual(a.cls, str)
        self.assertEqual(a.self, 42)
        self.assertEqual(a.typename, 'foo')
        self.assertEqual(a.fields, [('bar', tuple)])

    def test_empty_namedtuple(self):
        NT = NamedTuple('NT')

        class CNT(NamedTuple):
            pass  # empty body

        for struct in [NT, CNT]:
            with self.subTest(struct=struct):
                self.assertEqual(struct._fields, ())
                self.assertEqual(struct.__annotations__, {})
                self.assertIsInstance(struct(), struct)
                # Attribute was added in a micro version of 3.7
                # and is tested more fully elsewhere
                if hasattr(struct, "_field_defaults"):
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
        self.assertEqual(NamedTuple.__doc__, typing.NamedTuple.__doc__)
        self.assertIsInstance(NamedTuple.__doc__, str)

    @skipUnless(TYPING_3_8_0, "NamedTuple had a bad signature on <=3.7")
    def test_signature_is_same_as_typing_NamedTuple(self):
        self.assertEqual(inspect.signature(NamedTuple), inspect.signature(typing.NamedTuple))

    @skipIf(TYPING_3_8_0, "tests are only relevant to <=3.7")
    def test_signature_on_37(self):
        self.assertIsInstance(inspect.signature(NamedTuple), inspect.Signature)
        self.assertFalse(hasattr(NamedTuple, "__text_signature__"))

    @skipUnless(TYPING_3_9_0, "NamedTuple was a class on 3.8 and lower")
    def test_same_as_typing_NamedTuple_39_plus(self):
        self.assertEqual(
            set(dir(NamedTuple)),
            set(dir(typing.NamedTuple)) | {"__text_signature__"}
        )
        self.assertIs(type(NamedTuple), type(typing.NamedTuple))

    @skipIf(TYPING_3_9_0, "tests are only relevant to <=3.8")
    def test_same_as_typing_NamedTuple_38_minus(self):
        self.assertEqual(
            self.NestedEmployee.__annotations__,
            self.NestedEmployee._field_types
        )


class TypeVarLikeDefaultsTests(BaseTestCase):
    def test_typevar(self):
        T = typing_extensions.TypeVar('T', default=int)
        self.assertEqual(T.__default__, int)

        class A(Generic[T]): ...
        Alias = Optional[T]

    def test_paramspec(self):
        P = ParamSpec('P', default=(str, int))
        self.assertEqual(P.__default__, (str, int))

        class A(Generic[P]): ...
        Alias = typing.Callable[P, None]

    def test_typevartuple(self):
        Ts = TypeVarTuple('Ts', default=Unpack[Tuple[str, int]])
        self.assertEqual(Ts.__default__, Unpack[Tuple[str, int]])

        class A(Generic[Unpack[Ts]]): ...
        Alias = Optional[Unpack[Ts]]

    def test_pickle(self):
        global U, U_co, U_contra, U_default  # pickle wants to reference the class by name
        U = typing_extensions.TypeVar('U')
        U_co = typing_extensions.TypeVar('U_co', covariant=True)
        U_contra = typing_extensions.TypeVar('U_contra', contravariant=True)
        U_default = typing_extensions.TypeVar('U_default', default=int)
        for proto in range(pickle.HIGHEST_PROTOCOL):
            for typevar in (U, U_co, U_contra, U_default):
                z = pickle.loads(pickle.dumps(typevar, proto))
                self.assertEqual(z.__name__, typevar.__name__)
                self.assertEqual(z.__covariant__, typevar.__covariant__)
                self.assertEqual(z.__contravariant__, typevar.__contravariant__)
                self.assertEqual(z.__bound__, typevar.__bound__)
                self.assertEqual(z.__default__, typevar.__default__)


class TypeVarInferVarianceTests(BaseTestCase):
    def test_typevar(self):
        T = typing_extensions.TypeVar('T')
        self.assertFalse(T.__infer_variance__)
        T_infer = typing_extensions.TypeVar('T_infer', infer_variance=True)
        self.assertTrue(T_infer.__infer_variance__)
        T_noinfer = typing_extensions.TypeVar('T_noinfer', infer_variance=False)
        self.assertFalse(T_noinfer.__infer_variance__)

    def test_pickle(self):
        global U, U_infer  # pickle wants to reference the class by name
        U = typing_extensions.TypeVar('U')
        U_infer = typing_extensions.TypeVar('U_infer', infer_variance=True)
        for proto in range(pickle.HIGHEST_PROTOCOL):
            for typevar in (U, U_infer):
                z = pickle.loads(pickle.dumps(typevar, proto))
                self.assertEqual(z.__name__, typevar.__name__)
                self.assertEqual(z.__covariant__, typevar.__covariant__)
                self.assertEqual(z.__contravariant__, typevar.__contravariant__)
                self.assertEqual(z.__bound__, typevar.__bound__)
                self.assertEqual(z.__infer_variance__, typevar.__infer_variance__)


if __name__ == '__main__':
    main()

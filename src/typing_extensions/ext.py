"""
Protocols and type aliases that are not scheduled for inclusion in typing.
"""

from os import PathLike
from typing import AbstractSet, Awaitable, Container, Iterable, Tuple, TypeVar, Union
from typing_extensions import Protocol, TypeAlias

_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)
_KT = TypeVar("_KT")
_KT_co = TypeVar("_KT_co", covariant=True)
_KT_contra = TypeVar("_KT_contra", contravariant=True)
_VT_co = TypeVar("_VT_co", covariant=True)

#
# Protocols for dunder methods
#


class SupportsAnext(Protocol[_T_co]):
    def __anext__(self) -> Awaitable[_T_co]:
        ...


class SupportsDivMod(Protocol[_T_contra, _T_co]):
    def __divmod__(self, __other: _T_contra) -> _T_co:
        ...


class SupportsGetItem(Container[_KT_contra], Protocol[_KT_contra, _VT_co]):
    def __getitem__(self, __k: _KT_contra) -> _VT_co:
        ...


class SupportsNext(Protocol[_T_co]):
    def __next__(self) -> _T_co:
        ...


class SupportsRDivMod(Protocol[_T_contra, _T_co]):
    def __rdivmod__(self, __other: _T_contra) -> _T_co:
        ...


class SupportsTrunc(Protocol):
    def __trunc__(self) -> int:
        ...


#
# Mapping-like protocols
#


class SupportsKeys(Protocol[_KT_co]):
    def keys(self) -> Iterable[_KT_co]:
        ...


class SupportsItems(Protocol[_KT_co, _VT_co]):
    def items(self) -> AbstractSet[Tuple[_KT_co, _VT_co]]:
        ...


class SupportsKeysAndGetItem(Protocol[_KT, _VT_co]):
    def keys(self) -> Iterable[_KT]:
        ...

    def __getitem__(self, __k: _KT) -> _VT_co:
        ...


#
# I/O protocols
#


class SupportsRead(Protocol[_T_co]):
    def read(self, __length: int = ...) -> _T_co:
        ...


class SupportsReadline(Protocol[_T_co]):
    def readline(self, __length: int = ...) -> _T_co:
        ...


class SupportsWrite(Protocol[_T_contra]):
    def write(self, __s: _T_contra) -> object:
        ...


#
# Path aliases
#

StrPath: TypeAlias = Union[str, PathLike[str]]
BytesPath: TypeAlias = Union[bytes, PathLike[bytes]]

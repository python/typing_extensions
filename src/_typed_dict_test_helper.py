from typing import TypeVar, Generic, Optional, T
from typing_extensions import TypedDict

class FooGeneric(TypedDict, Generic[T]):
    a: Optional[T]

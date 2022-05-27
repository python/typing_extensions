from __future__ import annotations

from typing import Generic, Optional, T
from typing_extensions import TypedDict


class FooGeneric(TypedDict, Generic[T]):
    a: Optional[T]

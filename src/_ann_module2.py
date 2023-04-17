"""
Some correct syntax for variable annotation here.
This file is for testing purposes,
and is vendored from Lib/test/ann_module2.py in CPython.
"""

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

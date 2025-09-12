import math
import streamlit as st
from dataclasses import dataclass
from time import time


def timed_function(fn):
    """
    Usage:
    ```
    @timed_function
    def foo(..):
        ...
    ```
    """

    def wrap_function(*args, **kwargs):
        t1 = time()
        result = fn(*args, **kwargs)
        t2 = time()
        print(f"Function {fn.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_function


def round_increment(number, increment):
    return round(number * (1 / increment)) / (1 / increment)


def tick2sec(tick: int) -> float:
    return round_increment(tick * 0.05, 0.05)


def sec2tick(sec: float) -> int:
    return round(sec * 20)


@dataclass
class Vector:
    x: float
    y: float
    z: float

    def add(self, o):
        return Vector(self.x + o.x, self.y + o.y, self.z + o.z)

    def sub(self, o):
        return Vector(self.x - o.x, self.y - o.y, self.z - o.z)

    def mul(self, f):
        return Vector(self.x * f, self.y * f, self.z * f)

    def div(self, f):
        return Vector(self.x / f, self.y / f, self.z / f)

    def unm(self):
        return Vector(-self.x, -self.y, -self.z)

    def dot(self, o):
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, o):
        return Vector(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def round(self, tol: float = 1):
        return Vector(
            round(self.x / tol), round(self.y / tol), round(self.z / tol)
        ).mul(tol)

    def tostring(self):
        return f"X: {self.x} Y: {self.y} Z: {self.z}"

    def equals(self, o):
        return self.x == o.x and self.y == o.y and self.z == o.z

    def copy(self):
        return Vector(self.x, self.y, self.z)

    def __str__(self):
        return self.tostring()


def safe_extract(key: str, dict: dict):
    return dict[key] if key in dict else None


def ssg(key: str):
    return st.session_state[key]

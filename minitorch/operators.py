"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiply x * y."""
    return x * y


def id(x: float) -> float:
    """Identity: return x."""
    return x


def add(x: float, y: float) -> float:
    """Add x + y."""
    return x + y


def neg(x: float) -> float:
    """Negate x."""
    return -x


def lt(x: float, y: float) -> float:
    """Return 1.0 if x is less than y else 0.0."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Return 1.0 if x is equal to y else 0.0."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return x if x is greater than y else y."""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Return if x or y is within 1e-2 of each other"""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Sigmoid: 1.0 / (1.0 + exp(-x)) if x >= 0 else exp(x) / (1.0 + exp(x)).

    See: https://en.wikipedia.org/wiki/Sigmoid_function

    Args:
    ----
        x: float

    Returns:
    -------
        sigmoid(x)

    """
    # For stability.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def relu(x: float) -> float:
    """Relu: x if x is greater than 0 else 0.

    See: https://en.wikipedia.org/wiki/Rectifier_(neural_networks).
    """
    return max(x, 0)


EPS = 1e-6


def log(x: float) -> float:
    """Natural logarithm log(x)."""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Exponential exp(x)."""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Derivative of log(x) with respect to x."""
    return d / x


def inv(x: float) -> float:
    """Inverse: 1 / x."""
    return 1 / x


def inv_back(x: float, d: float) -> float:
    """Derivative of 1 / x with respect to x times d."""
    return -d / (x**2)


def relu_back(x: float, d: float) -> float:
    """Doing a relu backwards"""
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.

## Task 0.3
# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
    ----
        fn: Function from one value to one value.

    Returns:
    -------
        A function that takes a list, applies `fn` to each element, and returns a new list

    """

    # ASSIGN0.3
    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


# END ASSIGN0.3


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipWith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
    ----
        fn: combine two values

    Returns:
    -------
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by applying fn(x, y) on each pair of elements.

    """

    # ASSIGN0.3
    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


# END ASSIGN0.3


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order reduce.

    Args:
    ----
        fn: combine two values
        start: start value $x_0$

    Returns:
    -------
        Function that takes a list `ls` of elements
        $x_1 ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2, fn(x_1, x_0)))`

    """

    # ASSIGN0.3
    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


# END ASSIGN0.3


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Use `map` and `neg` to negate each element in `ls`"""
    # ASSIGN0.3
    return map(neg)(ls)


# END ASSIGN0.3


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add the elements of `ls1` and `ls2` using `zipWith` and `add`"""
    # ASSIGN0.3
    return zipWith(add)(ls1, ls2)


# END ASSIGN0.3


def sum(ls: Iterable[float]) -> float:
    """Sum up a list using `reduce` and `add`."""
    # ASSIGN0.3
    return reduce(add, 0.0)(ls)


# END ASSIGN0.3


def prod(ls: Iterable[float]) -> float:
    """Product of a list using `reduce` and `mul`."""
    # ASSIGN0.3
    return reduce(mul, 1.0)(ls)


# END ASSIGN0.3

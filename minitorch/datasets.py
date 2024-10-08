import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generates a list of N random points in 2D space.

    Each point is represented as a tuple of two floats (x_1, x_2),
    where both x_1 and x_2 are random numbers between 0 and 1, generated
    using the `random.random()` function.

    Args:
    ----
        N (int): The number of random points to generate.

    Returns:
    -------
        List[Tuple[float, float]]: A list of N tuples, where each tuple
        contains two random float values representing a point in 2D space.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generate a simple dataset with a vertical decision boundary.

    This function creates a dataset of N points in 2D space, where the
    classification is based on whether the x-coordinate (x_1) is less than 0.5.

    Args:
    ----
        N (int): The number of points to generate in the dataset.

    Returns:
    -------
        Graph: A Graph object containing:
            - N: The number of points
            - X: A list of N tuples, where each tuple (x_1, x_2) represents a point in 2D space
            - y: A list of N binary labels (0 or 1)

    The classification rule is:
        - y = 1 if x_1 < 0.5
        - y = 0 if x_1 >= 0.5

    This creates a simple vertical decision boundary at x_1 = 0.5.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generate a dataset with a diagonal decision boundary.

    This function creates a dataset of N points in 2D space, where the
    classification is based on whether the sum of x and y coordinates
    is less than 0.5.

    Args:
    ----
        N (int): The number of points to generate in the dataset.

    Returns:
    -------
        Graph: A Graph object containing:
            - N: The number of points
            - X: A list of N tuples, where each tuple (x_1, x_2) represents a point in 2D space
            - y: A list of N binary labels (0 or 1)

    The classification rule is:
        - y = 1 if x_1 + x_2 < 0.5
        - y = 0 if x_1 + x_2 >= 0.5

    This creates a diagonal decision boundary from (0.5, 0) to (0, 0.5) in the 2D space.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generate a dataset with two vertical decision boundaries.

    This function creates a dataset of N points in 2D space, where the
    classification is based on whether the x-coordinate (x_1) is less than 0.2
    or greater than 0.8.

    Args:
    ----
        N (int): The number of points to generate in the dataset.

    Returns:
    -------
        Graph: A Graph object containing:
            - N: The number of points
            - X: A list of N tuples, where each tuple (x_1, x_2) represents a point in 2D space
            - y: A list of N binary labels (0 or 1)

    The classification rule is:
        - y = 1 if x_1 < 0.2 or x_1 > 0.8
        - y = 0 if 0.2 <= x_1 <= 0.8

    This creates two vertical decision boundaries at x_1 = 0.2 and x_1 = 0.8,
    effectively splitting the space into three regions.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generate a dataset with an XOR-like decision boundary.

    This function creates a dataset of N points in 2D space, where the
    classification is based on an XOR-like pattern: points are labeled 1
    if they are in the top-left or bottom-right quadrants, and 0 otherwise.

    Args:
    ----
        N (int): The number of points to generate in the dataset.

    Returns:
    -------
        Graph: A Graph object containing:
            - N: The number of points
            - X: A list of N tuples, where each tuple (x_1, x_2) represents a point in 2D space
            - y: A list of N binary labels (0 or 1)

    The classification rule is:
        - y = 1 if (x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)
        - y = 0 otherwise

    This creates a decision boundary that resembles the XOR logical operation,
    with four distinct regions in the 2D space.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generate a dataset with a circular decision boundary.

    This function creates a dataset of N points in 2D space, where the
    classification is based on whether a point lies inside or outside
    a circle centered at (0.5, 0.5) with a radius of sqrt(0.1).

    Args:
    ----
        N (int): The number of points to generate in the dataset.

    Returns:
    -------
        Graph: A Graph object containing:
            - N: The number of points
            - X: A list of N tuples, where each tuple (x_1, x_2) represents a point in 2D space
            - y: A list of N binary labels (0 or 1)

    The classification rule is:
        - y = 1 if (x_1 - 0.5)^2 + (x_2 - 0.5)^2 > 0.1
        - y = 0 otherwise

    This creates a circular decision boundary with the center at (0.5, 0.5)
    and radius sqrt(0.1), separating the inner circle from the outer region.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generate a dataset with a spiral-shaped decision boundary.

    This function creates a dataset of N points in 2D space, where the
    classification is based on two intertwining spiral patterns.

    Args:
    ----
        N (int): The number of points to generate in the dataset.
                 The actual number of points will be N rounded down to the nearest even number.

    Returns:
    -------
        Graph: A Graph object containing:
            - N: The number of points
            - X: A list of N tuples, where each tuple (x_1, x_2) represents a point in 2D space
            - y: A list of N binary labels (0 or 1)

    The points are generated using parametric equations for two spirals:
    - One spiral for class 0 (first N/2 points)
    - One spiral for class 1 (second N/2 points)

    The spirals are centered at (0.5, 0.5) and rotate in opposite directions.
    This creates a complex, non-linear decision boundary between the two classes.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}

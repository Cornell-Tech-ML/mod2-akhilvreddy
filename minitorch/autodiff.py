from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.

    if not 0 <= arg < len(vals):
        raise IndexError("Argument 'arg' is out of range.")

    vals_plus = list(vals)
    vals_minus = list(vals)

    vals_plus[arg] += epsilon
    vals_minus[arg] -= epsilon

    f_plus = f(*vals_plus)
    f_minus = f(*vals_minus)

    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Prototype for the accumulate_derivative method."""
        ...

    @property
    def unique_id(self) -> int:
        """Prototype for the accumulate_derivative method."""
        ...

    def is_leaf(self) -> bool:
        """Prototype for the accumulate_derivative method."""
        ...

    def is_constant(self) -> bool:
        """Prototype for the accumulate_derivative method."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Prototype for the accumulate_derivative method."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Prototype for the accumulate_derivative method."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.

    result = []
    visited = []

    def dfs(var: Variable) -> None:
        """Depth-first search to traverse the computation graph.

        Args:
        ----
            var: The current variable to traverse

        """
        if var.unique_id not in visited and not var.is_constant():
            visited.append(var.unique_id)
            for parent in var.parents:
                dfs(parent)
            result.insert(0, var)

    dfs(variable)
    return result

    # raise NotImplementedError("Need to implement for Task 1.4")


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to compute derivatives for the leaf nodes.

    Args:
    ----
        variable (Variable): The right-most variable for which the derivative is being computed.
        deriv (Any): The derivative to propagate backward to the leaves.

    Returns:
    -------
        None: The function does not return anything. It updates the derivative values of each leaf
        through `accumulate_derivative`.

    """
    # TODO: Implement for Task 1.4.

    # A mapping from variable unique_id to its derivative
    grad_table = {variable.unique_id: deriv}
    print(grad_table)

    # Get the list of variables in topological order
    sorted_variables = topological_sort(variable)

    # Traverse the variables in topological order
    for var in sorted_variables:
        d_output = grad_table[var.unique_id]

        if var.is_leaf():
            var.accumulate_derivative(d_output)

        else:
            # Apply the chain rule to compute derivatives with respect to inputs
            for parent, d_input in var.chain_rule(d_output):
                if parent.unique_id in grad_table:
                    grad_table[parent.unique_id] += d_input
                else:
                    grad_table[parent.unique_id] = d_input

        # if var.is_constant():
        #     continue  # Skip constant variables

        # # Step 4: Get the derivative of the current variable
        # d_output = var.accumulate_derivative

        # # Step 5: Compute the chain rule and get the gradients for its inputs
        # for parent, d_parent in var.chain_rule(d_output):
        #     parent.accumulate_derivative(d_parent)  # Accumulate gradient for the parent

    # raise NotImplementedError("Need to implement for Task 1.4")


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensors for the current instance.

        This property provides access to the saved tensors, typically used
        during backpropagation in autograd operations. It retrieves the
        values stored in `self.saved_values` and returns them as a tuple.

        Returns
        -------
            Tuple[Any, ...]: A tuple containing the saved tensors.

        """
        return self.saved_values

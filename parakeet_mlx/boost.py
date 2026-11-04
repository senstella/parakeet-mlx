import unicodedata
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Callable, Generic, Hashable, Iterable, TypeVar, cast

import mlx.core as mx

S = TypeVar("S", bound=Hashable)  # state, gotta be hashable
A = TypeVar("A")
B = TypeVar("B")


def _zero(_) -> float:
    return 0.0


@dataclass(frozen=True)
class Machine(Generic[S, A]):  # mealy
    start: S
    step: Callable[[S, A], tuple[S, float]]  # (next state, weight)
    potential: Callable[[S], float] = _zero
    at_end: Callable[[S], float] = _zero


def advance(machine: Machine[S, A], state: S, symbol: A) -> tuple[S, float]:
    next_state, weight = machine.step(state, symbol)
    delta = machine.potential(next_state) - machine.potential(state) + weight
    return next_state, delta


def settle(machine: Machine[S, A], state: S) -> float:
    return machine.at_end(state) - machine.potential(state)


@lru_cache(maxsize=8)
def tabulate(machine: Machine[S, int], tokens: int) -> Callable[[S], mx.array]:
    @lru_cache(maxsize=1024)
    def deltas(state: S) -> mx.array:
        return mx.array(
            [advance(machine, state, token)[1] for token in range(tokens)],
            dtype=mx.float32,
        )

    return deltas


# primitives
def combine(*machines: Machine[Any, A]) -> Machine[tuple, A]:
    """Combine state machines to run in parallel and aggregate them"""

    def step(states, symbol):
        outcomes = [
            machine.step(state, symbol) for machine, state in zip(machines, states)
        ]
        next_states = tuple(state for state, _ in outcomes)
        return next_states, sum((weight for _, weight in outcomes), 0.0)

    def potential(states):
        return sum(
            (machine.potential(state) for machine, state in zip(machines, states)),
            0.0,
        )

    def at_end(states):
        return sum(
            (machine.at_end(state) for machine, state in zip(machines, states)),
            0.0,
        )

    return Machine(
        tuple(machine.start for machine in machines), step, potential, at_end
    )


def scale(factor: float, machine: Machine[S, A]) -> Machine[S, A]:
    """Scale the weight of state machine"""

    def step(state, symbol):
        next_state, weight = machine.step(state, symbol)
        return next_state, factor * weight

    def potential(state):
        return factor * machine.potential(state)

    def at_end(state):
        return factor * machine.at_end(state)

    return Machine(machine.start, step, potential, at_end)


def premap(f: Callable[[B], A], machine: Machine[S, A]) -> Machine[S, B]:
    """Map each symbol (one char with lift) before it gets passed to the state machine"""

    def step(state, symbol):
        return machine.step(state, f(symbol))

    return cast(Machine[S, B], replace(machine, step=step))


def prefilter(keep: Callable[[A], bool], machine: Machine[S, A]) -> Machine[S, A]:
    """Filter each symbol (one char with lift) and decide what gets passed to the state machine"""

    def step(state, symbol):
        return machine.step(state, symbol) if keep(symbol) else (state, 0.0)

    return replace(machine, step=step)


def prebind(f: Callable[[B], Iterable[A]], machine: Machine[S, A]) -> Machine[S, B]:
    """Expand each symbol (one char with lift) into zero or more symbols and pass it to the state machine"""

    def step(state, symbol):
        total = 0.0
        for inner in f(symbol):
            state, weight = machine.step(state, inner)
            total += weight
        return state, total

    return cast(Machine[S, B], replace(machine, step=step))


def lift(machine: Machine[S, str], vocabulary: list[str]) -> Machine[S, int]:
    lifted: Machine[S, int] = prebind(
        lambda token: vocabulary[token].replace("▁", " "), machine
    )
    return replace(lifted, step=lru_cache(maxsize=2**18)(lifted.step))


EMPTY: Machine[tuple, Any] = combine()


# common usages
def words(
    weights: dict[str, float],
    *,
    boundary: bool = True,
) -> Machine[tuple[frozenset[int], bool], str]:
    """Return a state machine boosting given dict[str, float]"""
    children: dict[tuple[int, str], int] = {}
    node_weights: dict[int, float] = {}
    potentials: dict[int, float] = {}

    for word, weight in weights.items():
        if not word:
            raise ValueError("words cannot be empty")

        node = 0
        for depth, char in enumerate(word, start=1):
            node = children.setdefault((node, char), len(children) + 1)
            if weight > 0:
                potentials[node] = max(
                    potentials.get(node, 0.0),
                    weight * depth / len(word),
                )
        node_weights[node] = weight

    def step(state, char):
        if not isinstance(char, str):
            raise TypeError(
                f"expected a character, got {type(char).__name__}; "
                "perhaps you forgot to wrap the state machine with boost.lift(machine, model.vocabulary)?"
            )

        nodes, after_word = state
        is_word = (
            boundary
            and unicodedata.category(char)[0] in "LMN"
            and unicodedata.east_asian_width(char) not in "WFH"
        )
        at_boundary = not (after_word and is_word)

        weight = (
            sum((node_weights.get(node, 0.0) for node in nodes), 0.0)
            if at_boundary
            else 0.0
        )
        heads = nodes | {0} if at_boundary else nodes
        next_nodes = frozenset(
            children[(node, char)] for node in heads if (node, char) in children
        )
        return (next_nodes, is_word), weight

    def potential(state):
        return max((potentials.get(node, 0.0) for node in state[0]), default=0.0)

    def at_end(state):
        return sum((node_weights.get(node, 0.0) for node in state[0]), 0.0)

    return Machine((frozenset(), False), step, potential, at_end)


def first_uppercase(weight: float) -> Machine[bool, str]:
    """Boost the possibility of the very first letter being uppercase"""

    def step(would, char):
        if not isinstance(char, str):
            raise TypeError(
                f"expected a character, got {type(char).__name__}; "
                "perhaps you forgot to wrap the state machine with boost.lift(machine, model.vocabulary)?"
            )

        if not would or not char.isalnum():
            return would, 0.0
        return False, weight if char.isupper() else 0.0

    return Machine(True, step)

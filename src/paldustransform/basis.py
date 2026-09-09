"""Classical utilities for working with the unitary-group-approach basis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def uga_gen(d: int) -> list[tuple[str, int, float]]:
    """Generate valid AC tableaux and their occupation and spin values."""
    if d < 1:
        raise ValueError("d must be positive")
    states = [("00", 0, 0.0), ("10", 1, 0.5), ("11", 2, 0.0)]
    for _ in range(d - 1):
        new_states = []
        for ac, occupation, spin in states:
            new_states.extend(((ac + "00", occupation, spin), (ac + "10", occupation + 1, spin + 0.5), (ac + "11", occupation + 2, spin)))
            if spin:
                new_states.insert(-1, (ac + "01", occupation + 1, spin - 0.5))
        states = new_states
    return states


def uga_gen_to_basis(states: list[tuple[str, int, float]]) -> list[tuple[str, int, float, float]]:
    """Expand tableaux to include every allowed spin projection."""
    return [(ac, occupation, spin, spin - offset) for ac, occupation, spin in states for offset in range(int(2 * spin + 1))]


def uga_to_comp(ac: str, m: float | int | None = None):
    """Expand an AC tableau into computational-basis states."""
    if len(ac) % 2 or not ac:
        raise ValueError("AC string must contain one two-bit code per orbital")
    first = ac[:2]
    if first in {"00", "11"}:
        states = {(first, 1.0): 0.0}
        spin = 0.0
    elif first == "10":
        states = {("10", 1.0): 0.5, ("01", 1.0): -0.5}
        spin = 0.5
    else:
        raise ValueError("Invalid AC string")
    for i in range(1, len(ac) // 2):
        code = ac[2 * i : 2 * i + 2]
        next_states: dict[tuple[str, float], float] = {}
        for (state, amplitude), projection in states.items():
            if code in {"00", "11"}:
                next_states[(state + code, amplitude)] = projection
            elif code == "10":
                next_states[(state + "10", amplitude * np.sqrt(0.5 + (2 * projection + 1) / (4 * spin + 2)))] = projection + 0.5
                next_states[(state + "01", amplitude * np.sqrt(0.5 - (2 * projection - 1) / (4 * spin + 2)))] = projection - 0.5
            elif code == "01":
                upper = -np.sqrt(0.5 - (2 * projection + 1) / (4 * spin + 2))
                lower = np.sqrt(0.5 + (2 * projection - 1) / (4 * spin + 2))
                if upper:
                    next_states[(state + "10", amplitude * upper)] = projection + 0.5
                if lower:
                    next_states[(state + "01", amplitude * lower)] = projection - 0.5
            else:
                raise ValueError("Invalid AC string")
        if code == "10":
            spin += 0.5
        elif code == "01":
            spin -= 0.5
        states = next_states
    return [key for key, projection in states.items() if projection == m] if m is not None else states


def bitstring_to_vector(bitstring: str) -> NDArray[np.complex128]:
    """Convert a bitstring into a computational-basis row vector."""
    vector = np.array([[1.0 + 0.0j]])
    for bit in bitstring:
        vector = np.kron(vector, np.array([[0, 1]]) if bit == "1" else np.array([[1, 0]]))
    return vector

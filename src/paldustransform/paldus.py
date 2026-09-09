"""Quantum circuits implementing the Paldus transform."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pytket.circuit import MultiplexorBox, Op, OpType, QubitRegister

from .core import QRegMap, RegisterBox, RegisterCircuit
from .increment import IncrementBox


def to_twoscomplement(bits: int, value: int) -> str:
    """Return the fixed-width two's-complement representation of an integer."""
    if bits < 1:
        raise ValueError("bits must be positive")
    if not -(1 << (bits - 1)) <= value < (1 << (bits - 1)):
        raise ValueError(f"{value} cannot be represented in {bits} signed bits")
    return format(value % (1 << bits), f"0{bits}b")


def step_to_op_map(orbital_number: int, step_number: int) -> dict[tuple[bool, ...], Op]:
    """Create the controlled rotation map for one Paldus step."""
    if orbital_number < 1:
        raise ValueError("orbital_number must be positive")
    if not 1 <= step_number <= orbital_number:
        raise ValueError("step_number must be between 1 and orbital_number")
    spin_bits = math.ceil(math.log2(orbital_number + 1))
    spin_z_bits = math.ceil(math.log2(2 * orbital_number + 1))
    result = {}
    for spin in range(step_number):
        for spin_z in range(-spin - 1, spin, 2):
            spin_key = tuple(bool(int(bit)) for bit in format(spin, f"0{spin_bits}b"))
            spin_z_key = tuple(bool(int(bit)) for bit in to_twoscomplement(spin_z_bits, spin_z))
            angle = 2 * math.acos(math.sqrt((spin + spin_z + 1) / (2 * spin + 2))) / np.pi
            result[spin_key + spin_z_key] = Op.create(OpType.PhasedISWAP, [0.25, angle])
    return result


@dataclass
class NCounterQRegs:
    storage: QubitRegister
    state: QubitRegister


class NCounterBox(RegisterBox):
    """Count the Hamming weight of a state register into a binary register."""

    def __init__(self, state_qubits: int):
        if state_qubits < 1:
            raise ValueError("state_qubits must be positive")
        self._storage_size = math.ceil(math.log2(state_qubits + 1))
        circuit = RegisterCircuit(name=f"{state_qubits}CounterBox")
        storage = circuit.add_q_register("n", self._storage_size)
        state = circuit.add_q_register("q", state_qubits)
        increment = IncrementBox(self._storage_size, "a").qcontrol(1, "b")
        for qubit in state:
            circuit.add_registerbox(increment, QRegMap(increment.qubits, [*storage, qubit]))
        super().__init__(NCounterQRegs(storage, state), circuit)

    @property
    def storage_size(self) -> int:
        return self._storage_size


@dataclass
class PaldusStepQRegs:
    spin: QubitRegister
    spin_z: QubitRegister
    state: QubitRegister


class PaldusStepBox(RegisterBox):
    """One Clebsch–Gordan-like step of a Paldus transform."""

    def __init__(self, orbital_number: int, step_number: int):
        if orbital_number < 1:
            raise ValueError("orbital_number must be positive")
        if not 1 <= step_number <= orbital_number:
            raise ValueError("step_number must be between 1 and orbital_number")
        self._orbital_number = orbital_number
        self._step_number = step_number
        self._spin_storage_qubits = math.ceil(math.log2(orbital_number + 1))
        self._spin_z_storage_qubits = math.ceil(math.log2(2 * orbital_number + 1))
        circuit = RegisterCircuit(name=f"PaldusStep{step_number}Box")
        spin = circuit.add_q_register("S", self._spin_storage_qubits)
        spin_z = circuit.add_q_register("m", self._spin_z_storage_qubits)
        state = circuit.add_q_register("q", 2)
        spin_increment = IncrementBox(len(spin), "a").qcontrol(1, "b")
        spin_z_increment = IncrementBox(len(spin_z), "a").qcontrol(1, "b")
        circuit.add_registerbox(spin_z_increment, QRegMap(spin_z_increment.qubits, [*spin_z, state[0]]))
        circuit.add_registerbox(spin_z_increment.dagger, QRegMap(spin_z_increment.qubits, [*spin_z, state[1]]))
        circuit.add_gate(MultiplexorBox(step_to_op_map(orbital_number, step_number)), [*spin, *spin_z, *state])
        circuit.add_registerbox(spin_increment, QRegMap(spin_increment.qubits, [*spin, state[0]]))
        circuit.add_registerbox(spin_increment.dagger, QRegMap(spin_increment.qubits, [*spin, state[1]]))
        super().__init__(PaldusStepQRegs(spin, spin_z, state), circuit)

    def op_map(self) -> dict[tuple[bool, ...], Op]:
        return step_to_op_map(self.orbital_number, self.step_number)

    @property
    def orbital_number(self) -> int:
        return self._orbital_number

    @property
    def step_number(self) -> int:
        return self._step_number

    @property
    def spin_storage_size(self) -> int:
        return self._spin_storage_qubits

    @property
    def spin_z_storage_size(self) -> int:
        return self._spin_z_storage_qubits


@dataclass
class PaldusNSMQRegs:
    occupation: QubitRegister
    spin: QubitRegister
    spin_z: QubitRegister
    state: QubitRegister


@dataclass
class PaldusSMQRegs:
    spin: QubitRegister
    spin_z: QubitRegister
    state: QubitRegister


class PaldusBox(RegisterBox):
    """Construct the Paldus transform for ``orbital_number`` orbitals."""

    def __init__(self, orbital_number: int, occupation_reg: bool = True):
        if orbital_number < 1:
            raise ValueError("orbital_number must be positive")
        self._orbital_number = orbital_number
        self._occupation_reg = occupation_reg
        self._occupation_storage_qubits = math.ceil(math.log2(2 * orbital_number + 1))
        self._spin_storage_qubits = math.ceil(math.log2(orbital_number + 1))
        self._spin_z_storage_qubits = math.ceil(math.log2(2 * orbital_number + 1))
        circuit = RegisterCircuit(name=f"Paldus{orbital_number}Box")
        spin = circuit.add_q_register("S", self._spin_storage_qubits)
        spin_z = circuit.add_q_register("m", self._spin_z_storage_qubits)
        state = circuit.add_q_register("q", 2 * orbital_number)
        occupation = circuit.add_q_register("N", self._occupation_storage_qubits) if occupation_reg else None
        qregs = PaldusNSMQRegs(occupation, spin, spin_z, state) if occupation is not None else PaldusSMQRegs(spin, spin_z, state)
        for i in range(orbital_number):
            step = PaldusStepBox(orbital_number, i + 1)
            circuit.add_registerbox(step, QRegMap(step.qubits, [*spin, *spin_z, state[2 * i], state[2 * i + 1]]))
        if occupation is not None:
            counter = NCounterBox(2 * orbital_number)
            circuit.add_registerbox(counter, QRegMap(counter.qubits, [*occupation, *state]))
        super().__init__(qregs, circuit)

    def spin_to_binary(self, spin: float) -> str:
        encoded = 2 * spin
        if spin < 0 or not float(encoded).is_integer() or encoded >= 2**self.spin_storage_size:
            raise ValueError("spin must be a representable non-negative multiple of 1/2")
        return format(int(encoded), f"0{self.spin_storage_size}b")

    @staticmethod
    def binary_to_spin(value: str) -> float:
        return int(value, 2) / 2

    def occupation_to_binary(self, occupation: int) -> str:
        if not 0 <= occupation < 2**self.occupation_storage_size:
            raise ValueError("occupation is outside the storage range")
        return format(occupation, f"0{self.occupation_storage_size}b")

    @staticmethod
    def binary_to_occupation(value: str) -> int:
        return int(value, 2)

    def spin_z_to_binary(self, spin_z: float) -> str:
        encoded = 2 * spin_z
        if not float(encoded).is_integer():
            raise ValueError("spin_z must be a multiple of 1/2")
        return to_twoscomplement(self.spin_z_storage_size, int(encoded))

    @staticmethod
    def binary_to_spin_z(value: str) -> float:
        integer = int(value, 2)
        if value[0] == "1":
            integer -= 1 << len(value)
        return integer / 2

    def no_occupation(self) -> PaldusBox:
        return PaldusBox(self.orbital_number, False)

    @property
    def orbital_number(self) -> int:
        return self._orbital_number

    @property
    def occupation_storage_size(self) -> int:
        return self._occupation_storage_qubits

    @property
    def spin_storage_size(self) -> int:
        return self._spin_storage_qubits

    @property
    def spin_z_storage_size(self) -> int:
        return self._spin_z_storage_qubits

    @property
    def ancilla_size(self) -> int:
        size = self.spin_storage_size + self.spin_z_storage_size
        return size + self.occupation_storage_size if self._occupation_reg else size

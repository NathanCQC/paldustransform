"""Small register-aware wrappers around pytket circuits."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from copy import copy, deepcopy
from dataclasses import is_dataclass, make_dataclass
from typing import Any, Self

from pytket.circuit import CircBox, QControlBox, Qubit, QubitRegister
from pytket._tket.circuit import Circuit

MapInput = QubitRegister | Qubit | list[Qubit]


def _qubits(items: Sequence[MapInput]) -> list[Qubit]:
    result: list[Qubit] = []
    for item in items:
        result.extend(item.to_list() if isinstance(item, QubitRegister) else item if isinstance(item, list) else [item])
    duplicates = [qubit for qubit, count in Counter(result).items() if count > 1]
    if duplicates:
        raise ValueError(f"Qubits appear more than once: {duplicates}")
    return result


class QRegMap:
    """Map a box's registers or qubits onto circuit registers or qubits."""

    def __init__(self, box_qregs: Sequence[MapInput], circuit_qregs: Sequence[MapInput]):
        if len(box_qregs) != len(circuit_qregs):
            raise ValueError("Box and circuit mappings must have the same length")
        self.box_qubits = _qubits(box_qregs)
        self.circ_qubits = _qubits(circuit_qregs)
        if len(self.box_qubits) != len(self.circ_qubits):
            raise ValueError("Box and circuit mappings must contain the same number of qubits")

    @property
    def qubit_map(self) -> dict[Qubit, Qubit]:
        return dict(zip(self.box_qubits, self.circ_qubits, strict=True))


class RegisterCircuit(Circuit):
    """A pytket circuit that can insert a :class:`RegisterBox`."""

    def add_registerbox(self, box: RegisterBox, qreg_map: QRegMap | None = None) -> Self:
        if qreg_map is None:
            if not set(box.qubits).issubset(self.qubits):
                raise ValueError("Box qubits are not present in the circuit")
            target_qubits = box.qubits
        else:
            if not set(qreg_map.box_qubits).issubset(box.qubits):
                raise ValueError("Mapped box qubits are not present in the box")
            if not set(qreg_map.circ_qubits).issubset(self.qubits):
                raise ValueError("Mapped circuit qubits are not present in the circuit")
            target_qubits = [qreg_map.qubit_map[qubit] for qubit in box.qubits]
        circuit = box.get_circuit().copy()
        circuit.flatten_registers()
        self.add_gate(CircBox(circuit), target_qubits)
        return self

    def copy(self) -> Self:
        return deepcopy(self)


class RegisterBox:
    """A circuit bundled with named quantum registers."""

    def __init__(self, qreg: Any, reg_circuit: RegisterCircuit):
        if not is_dataclass(qreg):
            raise ValueError("qreg must be a dataclass of QubitRegisters")
        self._qreg = qreg
        self._reg_circuit = reg_circuit

    @property
    def qreg(self) -> Any:
        return self._qreg

    @property
    def reg_circuit(self) -> RegisterCircuit:
        return self._reg_circuit

    @property
    def q_registers(self) -> list[QubitRegister]:
        return self.reg_circuit.q_registers

    @property
    def qubits(self) -> list[Qubit]:
        return self.reg_circuit.qubits

    @property
    def n_qubits(self) -> int:
        return self.reg_circuit.n_qubits

    def initialise_circuit(self) -> RegisterCircuit:
        circuit = RegisterCircuit()
        for qreg in self.q_registers:
            circuit.add_q_register(qreg)
        return circuit

    def get_circuit(self) -> RegisterCircuit:
        return self.reg_circuit

    def to_circbox(self) -> CircBox:
        circuit = self.reg_circuit.copy()
        circuit.flatten_registers()
        return CircBox(circuit)

    @property
    def dagger(self) -> RegisterBox:
        result = copy(self)
        result._reg_circuit = self.reg_circuit.dagger()
        result._reg_circuit.name = f"{self.reg_circuit.name}†"
        return result

    def qcontrol(self, n_control: int, control_qreg_str: str = "control") -> RegisterBox:
        circuit = self.initialise_circuit()
        circuit.name = f"C{n_control}{self.reg_circuit.name}"
        control = circuit.add_q_register(control_qreg_str, n_control)
        circuit.add_gate(QControlBox(self.to_circbox(), n_control), [*control, *self.qubits])
        fields = [(name, type(value)) for name, value in self.qreg.__dict__.items()]
        fields.append(("control", QubitRegister))
        qregs_type = make_dataclass("ControlledQRegs", fields)
        return RegisterBox(qregs_type(*self.qreg.__dict__.values(), control), circuit)

"""Modular quantum incrementer."""

from dataclasses import dataclass

from pytket.circuit import Op, OpType, QControlBox, QubitRegister

from .core import RegisterBox, RegisterCircuit


@dataclass
class IncrementQRegs:
    storage: QubitRegister


class IncrementBox(RegisterBox):
    """Increment an unsigned quantum register modulo ``2**storage_qubits``."""

    def __init__(self, storage_qubits: int, storage_qreg_str: str = "q"):
        if storage_qubits < 1:
            raise ValueError("storage_qubits must be positive")
        self._storage_range = 2**storage_qubits
        circuit = RegisterCircuit(name=f"IncMod{self._storage_range}Box")
        storage = circuit.add_q_register(storage_qreg_str, storage_qubits)
        for controls in range(storage_qubits - 1, 0, -1):
            gate = QControlBox(Op.create(OpType.X), controls)
            circuit.add_gate(gate, [storage[storage_qubits - j] for j in range(1, controls + 2)])
        circuit.X(storage[storage_qubits - 1])
        super().__init__(IncrementQRegs(storage), circuit)

    @property
    def storage_range(self) -> int:
        return self._storage_range

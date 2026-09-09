import numpy as np
import pytest

from paldustransform import PaldusBox, PaldusStepBox, bitstring_to_vector, to_twoscomplement, uga_gen, uga_gen_to_basis, uga_to_comp


def test_constructs_transform() -> None:
    box = PaldusBox(2)
    assert box.get_circuit().name == "Paldus2Box"
    assert box.n_qubits == 12
    assert box.ancilla_size == 8


def test_constructs_without_occupation_register() -> None:
    box = PaldusBox(3, occupation_reg=False)
    assert box.n_qubits == 11
    assert box.no_occupation().n_qubits == box.n_qubits


def test_step_validation() -> None:
    with pytest.raises(ValueError):
        PaldusStepBox(2, 3)


def test_encodings() -> None:
    box = PaldusBox(3)
    assert box.spin_to_binary(0.5) == "01"
    assert box.binary_to_spin("10") == 1
    assert box.spin_z_to_binary(-0.5) == "111"
    assert box.binary_to_spin_z("111") == -0.5
    assert to_twoscomplement(3, -2) == "110"


def test_basis_helpers() -> None:
    basis = uga_gen_to_basis(uga_gen(2))
    assert len(basis) == 16
    vector = bitstring_to_vector("10")
    np.testing.assert_array_equal(vector, [[0, 0, 1, 0]])
    states = uga_to_comp("1010", 1)
    assert states == [("1010", 1.0)]

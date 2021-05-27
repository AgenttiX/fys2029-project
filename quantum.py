import typing as tp

import cirq
import numpy as np
import sympy


class CircuitLayerBuilder:
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)


def convert_to_circuit(image: np.ndarray) -> cirq.Circuit:
    """Encode truncated classical image into quantum datapoint.
    In other words, create a circuit containing the gates that transform
    an initial zero-state to the state corresponding to the given array.
    """
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(*values.shape)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit


def create_quantum_model(data_dims: tp.Tuple[int, int]) -> tp.Tuple[cirq.Circuit, cirq.Z]:
    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits = cirq.GridQubit.rect(*data_dims)  # a 4x4 grid.
    readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
    circuit = cirq.Circuit()

    # Prepare the readout qubit in an equal superposition of the states 0 and 1.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = CircuitLayerBuilder(
        data_qubits=data_qubits,
        readout=readout)

    # Then add layers (experiment by adding more).
    builder.add_layer(circuit, cirq.XX, "xx1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)

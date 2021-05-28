import typing as tp

import cirq
import numpy as np
import sympy


class CircuitLayerBuilder:
    """Create a densely connected quantum neural network layer,
    where each input qubit is connected to the output qubit with a gate."""
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit: cirq.Circuit, gate: cirq.Gate, prefix: str):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)


def convert_to_circuit(image: np.ndarray) -> cirq.Circuit:
    """Encode truncated classical image into quantum datapoint.
    In other words, create a circuit containing the gates that transform
    an initial zero-state to the state corresponding to the given array.
    """
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(*image.shape)
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
    # This differs from the network proposed by Farhi et al., as in their network
    # the readout qubit is initially in state 1 instead of a superposition.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = CircuitLayerBuilder(
        data_qubits=data_qubits,
        readout=readout)

    # Then add layers
    # Todo: experiment by adding more
    builder.add_layer(circuit, cirq.XX, "xx1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)


def create_quantum_model2(data_dims: tp.Tuple[int, int]) -> tp.Tuple[cirq.Circuit, cirq.Z]:
    """Create a QNN model circuit and readout operation to go along with it.

    With YY gates as well.
    """
    data_qubits = cirq.GridQubit.rect(*data_dims)  # a 4x4 grid.
    readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
    circuit = cirq.Circuit()

    # Prepare the readout qubit in an equal superposition of the states 0 and 1.
    # This differs from the network proposed by Farhi et al., as in their network
    # the readout qubit is initially in state 1 instead of a superposition.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = CircuitLayerBuilder(
        data_qubits=data_qubits,
        readout=readout)

    # Then add layers
    builder.add_layer(circuit, cirq.XX, "xx1")
    builder.add_layer(circuit, cirq.YY, "yy1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)

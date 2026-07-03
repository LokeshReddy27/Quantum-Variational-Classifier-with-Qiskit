"""
Quantum Variational Classifier
--------------------------------
Hybrid quantum-classical ML model using Qiskit.
Encodes classical data into quantum states, processes via
parameterized variational circuits, and optimizes using
classical COBYLA optimizer.

Author: Gummadi Lokesh S P Reddy
GitHub: github.com/LokeshReddy27/quantum-variational-classifier
"""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler
from scipy.optimize import minimize


# ============================================================
# 1. DATA PREPARATION
# ============================================================

def generate_dataset(n_samples=200, noise=0.2, random_state=42):
    """Generate synthetic 2D binary classification dataset."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X = StandardScaler().fit_transform(X)
    X = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X)
    return X, y


# ============================================================
# 2. QUANTUM CIRCUIT DESIGN
# ============================================================

def angle_embedding(qc, features, qubits):
    """Encode classical features into quantum states via Ry rotations."""
    for i, (qubit, val) in enumerate(zip(qubits, features)):
        qc.ry(val, qubit)


def variational_layer(qc, params, qubits):
    """Single variational layer: parameterized Ry + entangling CNOTs."""
    for i, qubit in enumerate(qubits):
        qc.ry(params[i], qubit)
    for i in range(len(qubits) - 1):
        qc.cx(qubits[i], qubits[i + 1])
    qc.cx(qubits[-1], qubits[0])


def build_variational_circuit(n_qubits=2, n_layers=4):
    """
    Build the full parameterized quantum circuit.
    
    Architecture: Angle Embedding -> [Variational Layer] x n_layers -> Measurement
    """
    n_params = n_qubits * n_layers
    params = ParameterVector("theta", n_params)
    
    qc = QuantumCircuit(n_qubits)
    qubits = list(range(n_qubits))
    
    # Placeholder: features added at runtime
    qc.add_register(type(qc).__class__)
    
    feature_params = ParameterVector("x", n_qubits)
    angle_embedding(qc, feature_params, qubits)
    
    idx = 0
    for layer in range(n_layers):
        layer_params = [params[idx + i] for i in range(n_qubits)]
        variational_layer(qc, layer_params, qubits)
        idx += n_qubits
    
    qc.measure_all()
    return qc, params, feature_params


def circuit_to_parameterized(features, theta, n_qubits=2, n_layers=4):
    """Bind feature values and variational parameters to the circuit."""
    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))
    
    angle_embedding(qc, features, qubits)
    
    idx = 0
    for layer in range(n_layers):
        layer_params = theta[idx: idx + n_qubits]
        variational_layer(qc, layer_params, qubits)
        idx += n_qubits
    
    qc.measure_all()
    return qc


# ============================================================
# 3. CLASSIFIER
# ============================================================

class QuantumVariationalClassifier:
    """
    Variational Quantum Classifier using Qiskit Aer simulator.
    
    Maps binary classification to qubit measurement:
    - P(0) > 0.5 -> class 0
    - P(0) <= 0.5 -> class 1
    """
    
    def __init__(self, n_qubits=2, n_layers=4, shots=1024):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        self.n_params = n_qubits * n_layers
        self.theta_opt = None
        self.sampler = Sampler()
        self.simulator = AerSimulator(method="statevector")
    
    def quantum_prediction(self, features, theta):
        """Return probability of class 0 for a single sample."""
        qc = circuit_to_parameterized(features, theta, self.n_qubits, self.n_layers)
        result = self.simulator.run(qc, shots=self.shots).result()
        counts = result.get_counts()
        prob_0 = counts.get("0" * self.n_qubits, 0) / self.shots
        return prob_0
    
    def cost_function(self, theta, X, y):
        """Cross-entropy like cost: minimize when prediction matches label."""
        cost = 0
        for features, label in zip(X, y):
            prob_0 = self.quantum_prediction(features, theta)
            pred = 0 if prob_0 > 0.5 else 1
            cost += (pred != label)
        return cost / len(X)
    
    def fit(self, X, y, maxiter=100):
        """Train the quantum classifier using COBYLA optimizer."""
        theta_init = np.random.uniform(0, 2 * np.pi, self.n_params)
        result = minimize(
            self.cost_function,
            theta_init,
            args=(X, y),
            method="COBYLA",
            options={"maxiter": maxiter, "disp": True}
        )
        self.theta_opt = result.x
        return self
    
    def predict(self, X):
        """Predict class labels for samples."""
        predictions = []
        for features in X:
            prob_0 = self.quantum_prediction(features, self.theta_opt)
            predictions.append(0 if prob_0 > 0.5 else 1)
        return np.array(predictions)
    
    def score(self, X, y):
        """Return classification accuracy."""
        return np.mean(self.predict(X) == y)


# ============================================================
# 4. ANSATZ COMPARISON (EXTRA CREDIT)
# ============================================================

class CircuitAnsatz:
    """Collection of different circuit architectures for comparison."""
    
    @staticmethod
    def heisenberg_ansatz(features, theta, n_qubits=2, n_layers=4):
        """Heisenberg-model inspired ansatz with XX, YY, ZZ interactions."""
        qc = QuantumCircuit(n_qubits, n_qubits)
        qubits = list(range(n_qubits))
        
        angle_embedding(qc, features, qubits)
        
        idx = 0
        for layer in range(n_layers):
            for i, qubit in enumerate(qubits):
                qc.ry(theta[idx + i], qubit)
            idx += n_qubits
            for i in range(len(qubits) - 1):
                qc.rxx(theta[idx], qubits[i], qubits[i + 1])
                idx += 1
                qc.ryy(theta[idx], qubits[i], qubits[i + 1])
                idx += 1
        
        qc.measure_all()
        return qc


# ============================================================
# 5. MAIN
# ============================================================

def main():
    print("=" * 60)
    print("QUANTUM VARIATIONAL CLASSIFIER")
    print("Hybrid Quantum-Classical Machine Learning with Qiskit")
    print("=" * 60)
    
    # Generate data
    print("\n[1] Generating synthetic dataset (make_moons)...")
    X, y = generate_dataset(n_samples=200, noise=0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"    Training samples: {len(X_train)}")
    print(f"    Test samples:     {len(X_test)}")
    
    # Train
    print("\n[2] Initializing Quantum Variational Classifier...")
    qvc = QuantumVariationalClassifier(n_qubits=2, n_layers=4, shots=1024)
    
    print("\n[3] Training with COBYLA optimizer...")
    print("    (This runs on Qiskit Aer simulator — no quantum hardware needed)\n")
    qvc.fit(X_train, y_train, maxiter=80)
    
    # Evaluate
    train_acc = qvc.score(X_train, y_train)
    test_acc = qvc.score(X_test, y_test)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Training accuracy:  {train_acc:.2%}")
    print(f"  Test accuracy:      {test_acc:.2%}")
    print(f"  Circuit depth:      {4 * qvc.n_layers + 1} gates")
    print(f"  Parameters trained: {qvc.n_params}")
    print("=" * 60)
    
    # Optimized parameters
    print("\nOptimized circuit parameters (theta):")
    print(np.round(qvc.theta_opt, 4))
    
    return qvc


if __name__ == "__main__":
    qvc = main()

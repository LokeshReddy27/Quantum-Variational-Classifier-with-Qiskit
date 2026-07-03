<<<<<<< HEAD
# Quantum Variational Classifier

Hybrid quantum-classical machine learning model built with **Qiskit**. Encodes classical data into quantum states, processes through parameterized variational circuits, and optimizes with COBYLA — no quantum hardware required (runs on Aer simulator).

## Architecture

```
Classical Data → Angle Embedding (Ry) → [Variational Layer × 4] → Measurement → Class Prediction
                                           │
                                    Ry + CNOT entanglement
```

| Component | Description |
|-----------|-------------|
| **Angle Embedding** | Encodes (x, y) features into Ry rotation angles on 2 qubits |
| **Variational Layer** | Parameterized Ry gates + CNOT entanglement across qubits |
| **Optimizer** | COBYLA (gradient-free, suitable for noisy quantum cost landscapes) |
| **Measurement** | Probability of |00⟩ state → class 0 if > 0.5, else class 1 |

## Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~92% |
| Test Accuracy | ~88% |
| Circuit Depth | 17 gates (4 layers) |
| Trainable Parameters | 8 |
| Simulator | Qiskit Aer (statevector) |

## Quick Start

```bash
pip install -r requirements.txt
python quantum_variational_classifier.py
```

## How It Works

1. **Data**: `make_moons()` dataset — 200 samples, standardized and scaled to [0, π]
2. **Embedding**: Each (x, y) pair encoded onto 2 qubits via Ry rotation gates
3. **Circuit**: 4 variational layers, each with 2 Ry gates + CNOT entangling gates forming a ring topology
4. **Measurement**: Qubits measured; probability of |00⟩ determines class
5. **Optimization**: COBYLA minimizes misclassification over 80 iterations

## Why This Matters for Quantum Computing

This project demonstrates the core concepts behind near-term quantum advantage:

- **Parameterized Quantum Circuits** — The same architecture powering VQE (chemistry) and QAOA (optimization)
- **Hybrid Quantum-Classical Workflow** — Classical optimizer + quantum circuit — the standard paradigm for NISQ devices
- **Quantum Encoding** — Mapping real-world data onto quantum states
- **Entanglement as Computation** — CNOT gates create correlations classical circuits cannot efficiently replicate

## Author

**Gummadi Lokesh S P Reddy** — github.com/LokeshReddy27
=======
# Quantum-Variational-Classifier-with-Qiskit
>>>>>>> c799d877562066aca80fefbe8a4dbc4fc5cd4b34

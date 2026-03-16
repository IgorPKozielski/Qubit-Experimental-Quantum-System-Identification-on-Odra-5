# Qubit-Experimental-Quantum-System-Identification-on-Odra-5
Research project investigating the physical dynamics of the Odra-5 quantum device using simulations, experimental measurements, and machine-learning methods for quantum state reconstruction and tomography.

This repository contains an ongoing student research project focused on the characterization and modeling of a noisy quantum system implemented on the Odra-5 quantum device. The goal of the project is to investigate how the physical properties of an open quantum system can be reconstructed from experimental measurements and to develop a practical methodology for analyzing such systems using both physics-based modeling and machine learning approaches.

The project is motivated by the availability of the Odra-5 quantum computer, which provides experimental access to a small-scale quantum processor. Instead of treating the device as a black-box computational resource, the main objective of this work is to study its underlying physical dynamics, including noise processes, time-dependent evolution, and the structure of the effective Hamiltonian governing the system.

The investigation begins with the study of simplified quantum models and numerical simulations of small systems, starting from single-qubit dynamics and spin interactions, and gradually scaling toward a five-qubit system, which corresponds to the architecture accessible on the Odra-5 platform.

The theoretical foundation of the project is based on the time-dependent Schrödinger equation

i ∂ψ(t) / ∂t = H(t) ψ(t)

which is discretized using finite-difference methods to obtain an iterative numerical scheme for time evolution. The Hamiltonian is assumed to be time-dependent, reflecting the driven and noisy nature of experimental quantum hardware.

A key part of the project involves developing tools for quantum state reconstruction from experimental measurements. Measurement data collected from the Odra-5 device will be used to estimate the density matrix describing the system state. In addition to standard quantum tomography techniques, the project explores the possibility of applying modern approaches inspired by recent work on Lindbladian learning with neural differential equations, where neural networks are used to infer the effective dynamics of an open quantum system.

The implementation is written in Python, making use of scientific and quantum-computing libraries such as Qiskit, PyTorch, NumPy, SciPy, and Matplotlib. The repository will contain simulation code, analysis tools, experimental data pipelines, and research notebooks documenting the methodology.

This project is conducted by a student research team of six members, coordinated by the repository author and supervised by a faculty advisor. The long-term goal is to establish a practical experimental workflow that demonstrates the scientific and educational potential of the Odra-5 quantum device, turning it into an active platform for quantum system identification and experimental quantum computing research.

The project is currently work in progress, and the repository will evolve as new methods and experimental results are developed.

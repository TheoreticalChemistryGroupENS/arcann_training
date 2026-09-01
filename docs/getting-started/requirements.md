# ArcaNN Requirements

## Python Package Requirements

You do **not** need to install these yourself: they are all declared as dependencies in ArcaNN's `pyproject.toml`, so a standard installation (see [Installation](./installation.md)) will resolve and install them for you automatically. The list below is provided for reference only.

- **Python**: `>= 3.10`
- **Pip**: `>= 21.3`
- **Setuptools**: `>= 60.0`
- **Wheel**: `>= 0.37`
- **NumPy**: `>= 1.22`
- **ASE** (Atomic Simulation Environment): required for reading and writing atomic configurations, in particular the `extxyz` format used by MACE.
- **PyYAML**: required to read the MACE configuration files.

## External Programs for Trajectories/Structures Manipulation

ArcaNN requires the following external programs for manipulating trajectories and structures:

- **VMD**: `>= 1.9.3`
- **Atomsk**: `>= b0.12.2`

## Supported Programs by Workflow Step

Different steps in the workflow are supported by specific programs. For the **training** step you need **one** of the two supported neural network potential programs — the one matching your chosen architecture:

**Neural network potential program (pick one, used in the training step):**

- **DeePMD-kit**: `>= 2.0` (supported versions currently include `2.0`, `2.1`, `2.2`, and `3.0`). Used when `nnp_program` is set to `"deepmd"` (the default). Also used in the optional **testing** step.
- **MACE**: used when `nnp_program` is set to `"mace"`. Note that the **testing** step does not yet support MACE, and that MACE exploration currently only works with LAMMPS (not i-PI or Sander/EMLE; see [Exploration](../usage/exploration.md)). MACE itself is **not** a default dependency of ArcaNN: you may additionally install one of the `mace`, `mace-mliap`, or `symmetrix` extras — see [Installing MACE Support](./installation.md#installing-mace-support) — matching the `pair_style` below that your LAMMPS build supports.

**Molecular dynamics and supporting programs (used in the exploration step):**

- **LAMMPS**: used for classical-nuclei simulations. It must be built with support for the potential you selected:
    - For DeePMD-kit, LAMMPS must be compatible with DeePMD-kit.
    - For MACE, LAMMPS must provide a compatible MACE interface. ArcaNN recognizes the `mace`, `mliap`, and `symmetrix/mace` pair styles (Kokkos/GPU variants are also accepted); you choose which one to use in your LAMMPS input file.
- **i-PI**: for quantum-nuclei (path-integral) simulations. **Under development**.
- **PLUMED**: for enhanced-sampling/biased simulations. **Only DeepMD supported**.

**Electronic-structure program (pick one, used in the labeling step):**

- **CP2K**: `>= 6.1`. Used when `labeling_program` is set to `"cp2k"` (the default). Supports periodic systems, and a two-step (lower level then reference level) calculation per candidate. Also required for extracting stress tensors and Wannier centers.
- **ORCA**: used when `labeling_program` is set to `"orca"`. Labeling is single-step and always treated as an isolated, non-periodic (molecular) calculation — there is no periodic boundary condition support and no stress tensor/virial or Wannier center extraction; see [ORCA labeling](../usage/labeling.md#orca-labeling).

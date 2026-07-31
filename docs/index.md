<div style="text-align: center;">
<img src="arcann_logo.svg" alt="ArcaNN logo" style="width: 25%; height: auto;" />
</div>

---

[![GNU AGPL v3.0 License](https://img.shields.io/github/license/TheoreticalChemistryGroupENS/arcann_training.svg)](https://github.com/TheoreticalChemistryGroupENS/arcann_training/blob/main/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1039%2FD4DD00209A-004976.svg)](https://doi.org/10.1039/D4DD00209A)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2407.07751-b31b1b.svg)](https://doi.org/10.48550/arXiv.2407.07751)

[![Unit Tests](https://github.com/TheoreticalChemistryGroupENS/arcann_training/actions/workflows/unittests_matrix.yml/badge.svg?branch=main)](https://github.com/TheoreticalChemistryGroupENS/arcann_training/actions/workflows/unittests_matrix.yml)
[![Docs](https://github.com/TheoreticalChemistryGroupENS/arcann_training/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/TheoreticalChemistryGroupENS/arcann_training/actions/workflows/docs.yml)

---

# ArcaNN

ArcaNN proposes an automated enhanced sampling generation of training sets for chemically reactive machine learning interatomic potentials.
It simplifies and automates the iterative training process of a neural network potential for a user-chosen system.
ArcaNN supports two neural network potential architectures, and you choose which one to use at the start of the procedure:

- [**DeePMD-kit**](https://doi.org/10.1063/5.0155600) (the original, default choice), and
- [**MACE**](https://github.com/ACEsuit/mace), an equivariant message-passing architecture (added in this version).

The choice is made once, through a single `nnp_program` setting during initialization (see [Initialization](usage/initialization.md)), and the rest of the workflow adapts automatically. The core concepts of this training procedure are architecture-agnostic and could be extended to other network architectures in the future.

## Key Advantages

- **Modularity**: The code is designed with modularity in mind, allowing users to finely tune the training process to fit their specific system and workflow.
- **Traceability**: Every parameter set during the procedure is recorded, ensuring great traceability.

## Iterative Training Process

During the iterative training process, you will:

1. Train neural network potentials.

2. Use them as reactive force fields in molecular dynamics simulations to explore the phase space.

3. Select and label configurations based on a query by committee approach.

4. Train a new generation of neural network potentials again with the improved training set.

This workflow, often referred to as active or concurrent learning, is inspired by [DP-GEN](https://doi.org/10.1016/j.cpc.2020.107206).
We adopt their naming scheme for the steps in the iterative procedure. Each iteration, or cycle, consists of the following steps:

- **Training**
- **Exploration**
- **Labeling**
- (Optional) **Testing**

Ensure you understand the meaning of each step before using the code.

## GitHub Repository Structure

You will find in our [GitHub repository](https://github.com/TheoreticalChemistryGroupENS/arcann_training) everything you need to set up the ArcaNN software, as well as example files that you can use as an example. The repository contains several folders:

- The `tools/` folder contains helper scripts and files.
- The `arcann_training/` folder contains the `ArcaNN Training` code. **We strongly advise against modifying its contents**. See [ArcaNN Installation Guide](getting-started/installation.md) for the installation.
- The `examples/` folder contains template files to set up the iterative training procedure for your system. Within this folder you can find:
    - The `inputs/` folder with five JSON files, one per `step`.
These files contain all the keywords used to control each step of an iteration (namely **initialization**, **exploration**, **labeling**, **training** and optionally **test**), including their type and the default values taken by the code if a keyword isn't provided by the user.
If the default is a list containing a single value it means that this value will be repeated and used for every **system** (see the corresponding section for each `step`).
Note: For the **exploration** step some keywords have two default values, the first one will be used if the exploration is conducted with classical nuclei MD (*i.e.* LAMMPS) and the second one will be used with quantum nuclei MD (*i.e.* i-PI).
    - The `user_files/` folder with:
        - A `machine.json` template file where all the information about your cluster should be provided (see [HPC Configuration](getting-started/hpc_configuration.md)).
        - An input folder for each `step`, where skeleton files are provided as templates for writing your own inputs for the respective external programs.
        For example, in the `exploration_lammps/`, `labeling_cp2k/`, `training_deepmd/` and `training_mace/` folders, you can find the necessary files to perform the **exploration** with LAMMPS, the **labeling** with CP2K, and the **training** with either DeePMD-kit or MACE (see [Exploration](usage/exploration.md), [Labeling](usage/labeling.md) and [Training](usage/training.md) for a detailed description of the **tunable** keywords). You only need the training template that matches the architecture you selected.
        - A job folder for each step, where skeleton submission files are provided as templates that **ArcaNN** uses to launch the different phases in each `step` when they require a HPC machine.
        For example, in `job_exploration_lammps_slurm/`, `job_labeling_CP2K_slurm`, `job_training_deepmd_slurm` (or `job_training_mace_slurm` if you train with MACE) and the optional `step` `job_test_deepmd_slurm`, you can find basic `Slurm` submission files.

You **must** adapt these files to ensure they work on your machine (see [Usage](usage/iter_prerequisites.md)), but **be careful not to modify the replaceable keywords** (every word starting with `_R_` and ending with `_`) that Arcann will replace with user-defined or auto-generated values (e.g., the wall time for labeling calculations, the cluster partition to be used, etc.).

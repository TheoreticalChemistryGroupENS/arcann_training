# Iterative procedure prerequisites

When training a neural network potential (NNP) for a chemical system (or several systems that you want to describe with the same NNP), you will often want to explore the chemical space as diversely as possible.
In ArcaNN, this is made possible by the use of **systems**.
A **system** corresponds to a particular way of exploring the chemical space that interests you and will be represented by specific *datasets* within the total training set of the NNP.
A *dataset* corresponds to an ensemble of structures (*e.g.*, atomic positions, types of atoms, box size, etc.) and corresponding labels (*e.g.*, energy, forces, virial).

You get the idea: you need a subsystem for every kind of chemical composition, physical state (temperature, density, pressure, cell size, etc.), biased reactive pathway, and more that you wish to include in your final training *dataset*.

**Attention**, **systems** are defined once and for all in the [Initialization](initialization.md) of the procedure.
Because of this, every time you want to include a new subsystem (such as transition state structures, see [SN2](../examples/sn2.md) example), you will need to initialize the procedure again.
This is very simple—you only need to create a new `$WORK_DIR` and include the necessary files in `user_files/` for each extra **system** you want to add.

To initiate the iterative training procedure, you should create in your `$WORK_DIR` two folders: `user_files/` and `data/`.

In `user_files/` you will store all the files needed for each step. You can find some templates to start with in the [GitHub Repository](https://github.com/TheoreticalChemistryGroupENS/arcann_training), now available in your machine at your ArcaNN installation location `arcann_training/examples/user_files/`.

- For the exploration step, you must adapt the template files found in `exploration_lammps/` or `exploration_sander_emle/` depending of your choice. You will need the following files:
    - The input files: `SYSTEM.in` for LAMMPS and `SYSTEM.xml` for i-PI.
    - The plumed files: `plumed_SYSTEM.dat` where `SYSNAME` refers to the **system** name (additional PLUMED files can be used as `plumed_*_SYSNAME.dat`, which will also be taken into account for explorations).

- For the labeling step, use the template files found in `labeling_cp2k/` or `labeling_orca/`. You will need the following files:
    - The input files: `[1-2]_SYSNAME_labeling_XXXXX_[cluster].inp`, where `[cluster]` refers to the short string selected for the labeling cluster in the `machine.json`; see [Labeling](labeling.md).

- For the training step, prepare the file that matches the architecture you chose with `nnp_program` during [Initialization](initialization.md):
    - **If you use DeePMD-kit** (`nnp_program: "deepmd"`), from `training_deepmd/` you need a DeePMD-kit JSON file named `dptrain_VERSION.json`, where `VERSION` is the DeePMD-kit version that you will use (*e.g.*, `2.1`; currently supported versions are `2.0`, `2.1`, `2.2`, and `3.0`).
    - **If you use MACE** (`nnp_program: "mace"`), from `training_mace/` you need a MACE configuration file (in YAML) named `mace_MACEVERSION.yml` or `mace_MACEVERSION.yaml`, where `MACEVERSION` is the MACE version you will use (*e.g.*, `mace_0.3.14.yml`). This file plays the same role for MACE that `dptrain_VERSION.json` plays for DeePMD-kit: it holds the model hyperparameters. If several matching files are present, ArcaNN uses the highest version number by default.

- The SLURM scripts for individual jobs and for job arrays are organised by step in several folders. Prepare your files according to your software choice for each step. You can find them in:
    - `job_exploration_lammps_slurm` and `job_exploration_sander_emle_slurm` for exploration.
    - `job_labeling_orca_slurm` and `job_labeling_cp2k_slurm` for the labeling.
    - `job_training_deepmd_slurm` (DeePMD-kit) or `job_training_mace_slurm` (MACE) for training — use the one matching your chosen architecture.
    - `job_test_deepmd_slurm` for testing (DeePMD-kit only; testing is not yet available for MACE).

We **strongly** advise you to create the previous files starting from the templates, as they contain replaceable strings for the key parameters that will be updated by the procedure.

- A representative file in the [LAMMPS Data Format](https://docs.lammps.org/2001/data_format.html) format for each **system**, named `SYSTEM.lmp`, where `SYSTEM` refers to the **system** name  (we will refer to them as `LMP` files).
This file represent the configuration of the **system**, with the number of atoms and the number of types of atoms, the simulation cell dimension, the atomic masses for each type and the atomic geometry of your **system**. They will be used as starting point for the first exploration.

The order of the atoms in the `LMP` files **must** be identical for every **system** and **must** match the atom-type order defined in your training configuration file — the `"type_map"` keyword of the DeePMD-kit `dptrain_VERSION.json` file, or the corresponding element ordering in the MACE `mace_MACEVERSION.yml` file.

If you train with MACE, note that the `pair_style` line inside each `SYSTEM.in` LAMMPS input file selects how the MACE model is evaluated during exploration. ArcaNN supports `pair_style mace`, `pair_style mliap`, and `pair_style symmetrix/mace` (the corresponding Kokkos/GPU variants are also accepted). Choose the one supported by your LAMMPS build; ArcaNN reads it automatically and prepares the matching model files.

- A `properties` file must be provided and named `properties.txt`.
This file will be used by [atomsk](https://atomsk.univ-lille.fr/tutorial_properties.php). In the case you have several **systems** with different chemical composition, the properties file must be the same for all systems to ensure consistent type mapping.

Then, gather all these files and store them inside the `user_files/` directory. **Don't** create subdirectories inside `user_files`.

Finally, you also need to prepare at least one initial *dataset* which will be used for your first neural network training, that you will store in the `data/` directory. The format of this data must match the `data_format` you set during [Initialization](initialization.md):

- **For DeePMD-kit** (`data_format: "set.000"`): each data set must follow DeePMD-kit standards, containing a `type.raw` file and a `set.000/` folder with `box.npy`, `coord.npy`, `energy.npy` and `force.npy` (see [DeePMD-kit documentation](https://docs.deepmodeling.com/projects/deepmd/en/master/)).
- **For MACE** (`data_format: "extxyz"`): each data set is stored as extended XYZ (`extxyz`) files, a plain-text format that carries the atomic positions together with the reference energies and forces.

You can prepare as many initial *datasets* as you wish and they should all be stored in the `$WORK_DIR/data/` folder with a folder name starting with `init_`. If you already have data in one format and need the other, the [`initialization transition` phase](initialization.md#switching-architectures-the-transition-phase) can convert it for you.

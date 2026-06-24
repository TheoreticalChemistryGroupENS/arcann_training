<div style="text-align: center;">
<img src="./docs/arcann_logo.svg" alt="ArcaNN logo" style="width: 25%; height: auto;" />
</div>

---

[![GNU AGPL v3.0 License](https://img.shields.io/github/license/arcann-chem/arcann_training.svg)](https://github.com/TheoreticalChemistryGroupENS/arcann_training/blob/main/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1039%2FD4DD00209A-004976.svg)](https://doi.org/10.1039/D4DD00209A)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2407.07751-b31b1b.svg)](https://doi.org/10.48550/arXiv.2407.07751)

[![Unit Tests Requirements](https://github.com/arcann-chem/arcann/actions/workflows/unittests_requirements.yml/badge.svg)](https://github.com/arcann-chem/arcann/actions/workflows/unittests_requirements.yml)
[![Unit Tests Matrix](https://github.com/arcann-chem/arcann/actions/workflows/unittests_matrix.yml/badge.svg?branch=main)](https://github.com/arcann-chem/arcann/actions/workflows/unittests_matrix.yml)
[![Docs](https://github.com/arcann-chem/arcann/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/arcann-chem/arcann/actions/workflows/docs.yml)

---

*This is fork of the [original ArcaNN repository](https://github.com/arcann-chem/arcann_training).*

---

# ArcaNN #

ArcaNN proposes an automated enhanced sampling generation of training sets for chemically reactive machine learning interatomic potentials.
In its current version, it aims to simplify and to automate the iterative training process of a [DeePMD-kit](https://doi.org/10.1063/5.0155600) or [MACE](https://doi.org/10.48550/arXiv.2206.07697) neural network potential (NNP) for a user-chosen system.
The main advantages of this code are its modularity, the ability to finely tune the training process to adapt to your system and workflow, and great traceability, as the code records every parameter set during the procedure.
During the iterative training process, you will iteratively train neural network potentials, use them as reactive force fields for molecular dynamics simulations (to explore the phase space), select and label some configurations based on a query by committee approach, and then train neural network potentials again with an improved training set, and so forth.
This workflow, sometimes referred to as active or concurrent learning, was heavily inspired by [DP-GEN](https://doi.org/10.1016/j.cpc.2020.107206), and we use their naming scheme for the steps of the iterative procedure.

We refer the reader to the [documentation](https://arcann-chem.github.io/arcann_training/) and the accompanying paper [ArcaNN: automated enhanced sampling generation of training sets for chemically reactive machine learning interatomic potentials](https://doi.org/10.1039/D4DD00209A). (*The documentation has not been updated to include changes from this fork.*)

## Installation ##

Preferably, install ArcaNN in an exclusive python environment. To install it, follow these instructions:

```bash
git clone https://github.com/TheoreticalChemistryGroupENS/arcann_training.git
pip install ./arcann_training
```

Automatic conversion of MACE model files during `exploration prepare` requires the following extra packages. Model conversion can also be done using `training compress` phase through a submitted job, not requiring the installation of extra packages.

- [mace-torch](https://mace-docs.readthedocs.io/en/latest/guide/installation.html), for all LAMMPS pair styles (libtorch, MLIAP, and Symmetrix)
- [cuequivariance-torch](https://github.com/nvidia/cuequivariance), for the MLIAP LAMMPS pair style
- [symmetrix](https://github.com/wcwitt/symmetrix), for the symmetrix LAMMPS pair style

These can be installed following the instructions of each package, or using the optional dependencies of ArcaNN:

```bash
pip install ./arcann_training[mace] # To install mace-torch
pip install ./arcann_training[mace-mliap] # To install mace-torch and cuequivariance-torch
pip install ./arcann_training[symmetrix] # To install mace-torch and symmetrix
```

In the case of symmetrix, `--config-settings` can also be passed.

## Usage ##

The [original documentation](https://arcann-chem.github.io/arcann_training/) should be read. Changes to usage of this version are explained bellow.

### Dataset ###

Support for [ExtendedXYZ](https://www.ovito.org/docs/current/reference/file_formats/input/xyz.html#extended-xyz-format) has been added, and can be used interchangeably with Set.000 format ([DeePMD format](https://docs.deepmodeling.com/projects/deepmd/en/master/data/data-conv.html#numpy-format)).

Validation datasets can be used, and are mandatory for MACE. They are defined using the folder name style: `valid_SYSNAME` or `init_valid_SYSNAME`. Where the last is the validation dataset for the initial dataset. An additional setting for the `initialization start` phase has been added to control the percentage of candidates that should be added to validation datasets: `validation/training_split` (default: `0.2`).

### Initialization ###

The `initialization start` phase sets up the NNP software that will be used, by default DeeMPD-kit is used. To change to MACE, create a `input.json` file, with:

```json
{
    "nnp_software": "mace",
    "data_format": "extxyz"
}
```

`data_format` sets the format of the dataset, it has to possible values `extxyz`  and `set.000`. Change of `validation/training_split` should also be done in this file. With the file created and the python environment with ArcaNN activated, run:

```bash
python -m arcann_training initialization start
```

### Training ###

Use the [example scripts](https://github.com/TheoreticalChemistryGroupENS/arcann_training/tree/main/examples/user_files/job_training_mace_slurm) provided to create the scripts for your HPC. Create a `mace_MACEVERSION.yaml` file (change `MACEVERSION` to the version of MACE that you want to use, e.g. `0.3.15`) configuring the model parameters. The `max_num_epochs` (default: `500`) set up in the `XXX-training/input.json` will overwrite the one in the `mace_MACEVERSION.yaml` file, it also replaces the `numb_steps` from the previous ArcaNN version. The variable `mace_model_version` (default: `0.3.14`) controls which `mace_MACEVERSION.yaml` file is expected. The `XXX-training/input.json` file can look like:

```json
{
    "mace_model_version": "0.3.14",
    "max_num_epochs": 500
}
```

With this file you can run:

```bash
python -m arcann_training training prepare
```

The order of the steps for the `training` phase for MACE is:

```mermaid
flowchart LR
    A(["prepare"]) --> B(["launch"]) --> C(["check_launch"]) --> D(["compress (optional)"]) --> E(["check_compress (optional)"])
```

The phases `compress` and `check_compress` are only mandatory when the environment with ArcaNN does not have the necessary extra packages installed, see [**Installation**](#installation). They will be used to convert the model files created on training to the appropriate format depending on the identified `pair_style` in the LAMMPS exploration inputs.

### Exploration ###

*MACE is not supported on i-PI and sander.*

Use the [example scripts](https://github.com/TheoreticalChemistryGroupENS/arcann_training/tree/main/examples/user_files/job_exploration_lammps_slurm) provided to create the scripts for your HPC.

This phase will convert the model files, if they were not during `training compress`. All MACE pair styles in LAMMPS don't do automatic energy and force deviation. Therefore, ArcaNN will modify the provided LAMMPS inputs to do reruns using the other models, and calculate deviation on the `deviate` step.

The order of the steps for the `exploration` phase is:

```mermaid
flowchart LR
    A(["prepare"]) --> B(["launch"]) --> C(["check"]) --> D(["deviate"]) --> E(["extract"])
```

### Labeling ###

Labeling can be done using [ORCA](https://www.faccts.de/docs/orca/6.1/manual/index.html) and [CP2K](https://www.cp2k.org/). No changes for the user were done in this phase. However, we are aware of bugs when using CP2K 2024, that also affects the original ArcaNN code.

The order of the steps for the `labeling` phase is:

```mermaid
flowchart LR
    A(["prepare"]) --> B(["launch"]) --> C(["check"]) --> D(["extract"])
```

### Iterations ###

After an active learning iteration is done and you want to move to the next one, you should run:

```bash
python -m arcann_training training increment
```

## License ##

Distributed under the GNU Affero General Public License v3.0. See `LICENSE` for more information.

## How to cite ##

If you use this code, please cite the following publication:

David, R.; de la Puente, M.; Gomez, A.; Anton, O.; Stirnemann, G.; Laage, D. ArcaNN: automated enhanced sampling generation of training sets for chemically reactive machine learning interatomic potentials. Digital Discovery, 2024, DOI: [10.1039/D4DD00209A](https://doi.org/10.1039/D4DD00209A).

## Fundings & HPC Allocations ##

- Idex ANR-10-IDEX-0001-02PSL
- ERC Grant Agreement No. 757111
- GENCI Grant 2023-A0130707156
- FAPESP Grant 2025/15166-9

## Acknowledgments & Sources ##

- [Stackoverflow](https://stackoverflow.com/)

### Beta-testers ###

- Olaia Anton, Zakarya Benayad, Miguel de la Puente, Axel Gomez
- Oscar Gayraud, Pierre Girard, Anne Milet
- Meritxell Malagarriga Perez, Adrián García
- Ashley Borkowski, Pauf Neupane, Ward Thompson
- Hadi Dinpajooh

### Atomsk ###

- Hirel, P. Atomsk: A Tool for Manipulating and Converting Atomic Data Files. Comput. Phys. Commun. 2015, 197, 212–219. [https://doi.org/10.1016/j.cpc.2015.07.012](https://doi.org/10.1016/j.cpc.2015.07.012).

### VMD ###

- Humphrey, W.; Dalke, A.; Schulten, K. VMD: Visual Molecular Dynamics. J. Mol. Graph. 1996, 14 (1), 33–38. [https://doi.org/10.1016/0263-7855(96)00018-5](https://doi.org/10.1016/0263-7855(96)00018-5).

### DeePMD-kit ###

- Zeng, J.; Zhang, D.; Lu, D.; Mo, P.; Li, Z.; Chen, Y.; Rynik, M.; Huang, L.; Li, Z.; Shi, S.; Wang, Y.; Ye, H.; Tuo, P.; Yang, J.; Ding, Y.; Li, Y.; Tisi, D.; Zeng, Q.; Bao, H.; Xia, Y.; Huang, J.; Muraoka, K.; Wang, Y.; Chang, J.; Yuan, F.; Bore, S. L.; Cai, C.; Lin, Y.; Wang, B.; Xu, J.; Zhu, J.-X.; Luo, C.; Zhang, Y.; Goodall, R. E. A.; Liang, W.; Singh, A. K.; Yao, S.; Zhang, J.; Wentzcovitch, R.; Han, J.; Liu, J.; Jia, W.; York, D. M.; E, W.; Car, R.; Zhang, L.; Wang, H. DeePMD-Kit v2: A Software Package for Deep Potential Models. J. Chem. Phys. 2023, 159 (5), 054801. [https://doi.org/10.1103/PhysRevMaterials.3.023804](https://doi.org/10.1063/5.0155600).
- Wang, H.; Zhang, L.; Han, J.; E, W. DeePMD-Kit: A Deep Learning Package for Many-Body Potential Energy Representation and Molecular Dynamics. Comput. Phys. Commun. 2018, 228, 178–184. [https://doi.org/10.1016/j.cpc.2018.03.016](https://doi.org/10.1016/j.cpc.2018.03.016).

### MACE ###

- Batatia, I.; Kovacs, D. P.; Simm, G.; Ortner, C.; Csanyi, G. MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields. In Advances in neural information processing systems; Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., Oh, A., Eds.; Curran Associates, Inc., 2022; Vol. 35, pp 11423–11436. [https://doi.org/10.48550/arXiv.2206.07697](https://doi.org/10.48550/arXiv.2206.07697).

### DP-Compress ###

- Lu, D.; Jiang, W.; Chen, Y.; Zhang, L.; Jia, W.; Wang, H.; Chen, M. DP Compress: A Model Compression Scheme for Generating Efficient Deep Potential Models. J. Chem. Theory Comput. 2022, 18 (9), 5559–5567. [https://doi.org/10.1021/acs.jctc.2c00102](https://doi.org/10.1021/acs.jctc.2c00102).

### Concurrent Learning ###

- Zhang, L.; Lin, D.-Y.; Wang, H.; Car, R.; E, W. Active Learning of Uniformly Accurate Interatomic Potentials for Materials Simulation. Phys. Rev. Materials 2019, 3 (2), 023804. [https://doi.org/10.1103/PhysRevMaterials.3.023804](https://doi.org/10.1103/PhysRevMaterials.3.023804)
- Zhang, Y.; Wang, H.; Chen, W.; Zeng, J.; Zhang, L.; Wang, H.; E, W. DP-GEN: A Concurrent Learning Platform for the Generation of Reliable Deep Learning Based Potential Energy Models. Comput. Phys. Commun. 2020, 253, 107206. [https://doi.org/10.1016/j.cpc.2020.107206](https://doi.org/10.1016/j.cpc.2020.107206).

### LAMMPS ###

- Thompson, A. P.; Aktulga, H. M.; Berger, R.; Bolintineanu, D. S.; Brown, W. M.; Crozier, P. S.; In ’T Veld, P. J.; Kohlmeyer, A.; Moore, S. G.; Nguyen, T. D.; Shan, R.; Stevens, M. J.; Tranchida, J.; Trott, C.; Plimpton, S. J. LAMMPS - a Flexible Simulation Tool for Particle-Based Materials Modeling at the Atomic, Meso, and Continuum Scales. Comput. Phys. Commun. 2022, 271, 108171. [https://doi.org/10.1016/j.cpc.2021.108171](https://doi.org/10.1016/j.cpc.2021.108171).

### i-PI ###

- Kapil, V.; Rossi, M.; Marsalek, O.; Petraglia, R.; Litman, Y.; Spura, T.; Cheng, B.; Cuzzocrea, A.; Meißner, R. H.; Wilkins, D. M.; Helfrecht, B. A.; Juda, P.; Bienvenue, S. P.; Fang, W.; Kessler, J.; Poltavsky, I.; Vandenbrande, S.; Wieme, J.; Corminboeuf, C.; Kühne, T. D.; Manolopoulos, D. E.; Markland, T. E.; Richardson, J. O.; Tkatchenko, A.; Tribello, G. A.; Van Speybroeck, V.; Ceriotti, M. I-PI 2.0: A Universal Force Engine for Advanced Molecular Simulations. Comput. Phys. Commun. 2019, 236, 214–223. [https://doi.org/10.1016/j.cpc.2018.09.020](https://doi.org/10.1016/j.cpc.2018.09.020).

### CP2K ###

- Kühne, T. D.; Iannuzzi, M.; Del Ben, M.; Rybkin, V. V.; Seewald, P.; Stein, F.; Laino, T.; Khaliullin, R. Z.; Schütt, O.; Schiffmann, F.; Golze, D.; Wilhelm, J.; Chulkov, S.; Bani-Hashemian, M. H.; Weber, V.; Borštnik, U.; Taillefumier, M.; Jakobovits, A. S.; Lazzaro, A.; Pabst, H.; Müller, T.; Schade, R.; Guidon, M.; Andermatt, S.; Holmberg, N.; Schenter, G. K.; Hehn, A.; Bussy, A.; Belleflamme, F.; Tabacchi, G.; Glöß, A.; Lass, M.; Bethune, I.; Mundy, C. J.; Plessl, C.; Watkins, M.; VandeVondele, J.; Krack, M.; Hutter, J. CP2K: An Electronic Structure and Molecular Dynamics Software Package - Quickstep: Efficient and Accurate Electronic Structure Calculations. J. Chem. Phys. 2020, 152 (19), 194103. [https://doi.org/10.1063/5.0007045](https://doi.org/10.1063/5.0007045).

### ORCA ###

- Neese, F.; Wennmohs, F.; Becker, U.; Riplinger, C. The ORCA Quantum Chemistry Program Package. The Journal of Chemical Physics 2020, 152 (22). [https://doi.org/10.1063/5.0004608](https://doi.org/10.1063/5.0004608).


# ArcaNN Installation Guide

## Installation on Machines with Internet Access

ArcaNN is a standard Python package: its dependencies (`ase`, `numpy`, `pyyaml`, ...) are declared in `pyproject.toml` and installed automatically by `pip`, so no separate Conda environment file is needed. To install it, follow these steps:

- **Clone or Download the Repository:**

Use the green `Code` button on the [repository](https://github.com/TheoreticalChemistryGroupENS/arcann_training)'s main page to either clone or download the repository.
While it's recommended to keep a local copy of this repository on any computer that will be used for preparing, running, or analyzing the iterative training process, this is not mandatory.

- **Navigate to the Repository Folder:**

After downloading or cloning, navigate to the main folder of the repository.

- **(Recommended) Create an Isolated Python Environment:**

ArcaNN requires Python `>= 3.10` (see [Requirements](requirements.md)). Any isolation tool works — Conda, `venv`, or [`uv`](https://docs.astral.sh/uv/) — for example:

```bash
conda create --name <ENVNAME> python=3.10
conda activate <ENVNAME>
```

(If you're developing ArcaNN itself rather than just running it, see [Contributions](../contributions/contributions.md) for the `uv`-based setup used in CI instead.)

- **Install the Package:**

```bash
pip install .
```

This resolves and installs all required dependencies automatically.

- **Verify the Installation:**
To ensure that `ArcaNN` has been installed correctly, run the following command:

```bash
python -m arcann_training --help
```

This command should display the basic usage message of the code.

- **Optional:**
If you wish, you can delete the repository folder after installation is complete.

**Note:** Alternatively, you can install the program in "editable" mode with:

```bash
pip install -e .
```

This method allows any modifications to the source files to take effect immediately during program execution. It is only recommended if you plan to modify the source files and requires you to keep the repository folder on your machine.

## Installing MACE Support

The base install above install the basics for what DeePMD-kit and MACE needs. There are optional MACE extras: whether you need them at all, and in which environment, depends on how MACE models get converted to their LAMMPS-ready format (see below). If you plan to set `nnp_program` to `"mace"` (see [Initialization](../usage/initialization.md)) and do end up needing them, install the extra dependencies that match the LAMMPS `pair_style` you will use for exploration (see [Requirements](requirements.md)) with one of:

```bash
# pair_style mace
pip install ".[mace]"

# pair_style mliap (adds cuequivariance-torch)
pip install ".[mace-mliap]"

# pair_style symmetrix/mace
pip install ".[symmetrix]"
```

(For `symmetrix`, `pip`'s `--config-settings` flag can also be passed through if your build needs it.) Each extra pulls in `mace-torch` plus whatever additional package that `pair_style` requires.

**Where you need this installed** depends on how MACE models get converted to their LAMMPS-ready format:

- If you run the `training compress` phase every iteration (recommended, and the default expectation — see [Training](../usage/training.md)), the conversion happens inside the submitted `Slurm` job, so you **do not** need these extras.
- If you skip `training compress`, the conversion happens wherever you run `exploration prepare`, for which you **do need** to have the matching extra installed. ArcaNN will otherwise warn you and ask you to run `training compress` instead.

## Installation on Machines without Internet Access

If your machine does not have access to the internet, download ArcaNN's dependencies on a machine that does, then transfer them over. Since dependencies are declared in `pyproject.toml`, `pip download` handles this without any extra tooling:

- **Download the Repository and the Packages it Needs:**

On a machine with internet access, download the `ArcaNN` repository, then download the wheels for ArcaNN and all its dependencies into a local folder (append `[mace]`, `[mace-mliap]`, or `[symmetrix]` to `./arcann_training` below if you also need [MACE support](#installing-mace-support)):

```bash
pip download ./arcann_training -d arcann_offline_packages
```

- **Transfer Files to the Offline Machine:**

Use rsync to transfer the downloaded packages and the ArcaNN repository to your offline machine:

```bash
rsync -rvu arcann_offline_packages USER@WORKMACHINE:/PATH/TO/INSTALLATION/FOLDER/.
rsync -rvu arcann_training USER@WORKMACHINE:/PATH/TO/INSTALLATION/FOLDER/.
```

- **Install ArcaNN on the Offline Machine:**

On your offline machine, install ArcaNN and its dependencies directly from the downloaded folder, without contacting any package index:

```bash
pip install --no-index --find-links=/PATH/TO/INSTALLATION/FOLDER/arcann_offline_packages ./arcann_training
```

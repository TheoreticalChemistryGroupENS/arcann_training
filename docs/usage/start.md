# Using ArcaNN

## Iterations, Steps and Phases of the Iterative Procedure

At this stage, ArcaNN is installed in your machine, and you have made the necessary changes to adapt it (see [HPC Configuration](../getting-started/hpc_configuration.md)). As in the [GitHub Repository](https://github.com/TheoreticalChemistryGroupENS/arcann_training/), you can now find in the location where you installed ArcaNN, an `arcann_training/` folder containing several files, as well as the `arcann_training/` scripts, a `tools/` directory and a `examples/` directory.

To start the procedure, create an empty directory anywhere you like that will be your iterative training working directory.
We will refer to this directory by the variable name `$WORK_DIR`.

We will describe the **prerequisites**, and then the **initialization**, **training**, **exploration**, **labeling** steps and, the optional **test**.
At the end of each step description, we include an example.

Before you begin, note that ArcaNN can build the potential with either of two architectures, **DeePMD-kit** or **MACE**. You select this once during initialization with the `nnp_program` setting, and every step below then adapts automatically. The workflow (the steps and their phases) is the same for both; where a phase behaves differently for MACE, this is pointed out in the tables and step descriptions.

As described in more detail below, training the NNP proceeds in iterations consisting of three **steps**: exploration, labeling, and training.
Each **step** is broken down into elementary tasks, which we call *phases*.
Each iteration will have three folders: XXX-exploration, XXX-labeling, and XXX-training (*e.g.*, `XXX` is `003` for the 3rd iteration).
Each **step** is executed in its corresponding folder by running, in order, the relevant *phases* with the following command:

```bash
python -m arcann_training STEP_NAME PHASE_NAME
```

where `STEP_NAME` refers to the current **step** (`initialization`, `exploration`, `labeling`, `training`, or `test`) and `PHASE_NAME` is the specific task that needs to be performed within that **step**.
This will become clearer with examples in the sections below, where each **step** is explained.
The following tables provide a brief description of the *phases* in each **step**, in the correct order.

### Command-line options

`python -m arcann_training STEP_NAME PHASE_NAME` accepts a few optional flags:

| Flag | Default | Description |
| --- | --- | --- |
| `-v`, `--verbose` | `0` | Logging verbosity: `0` for normal output, `1` for debug-level logging (see [Unexpected Behavior](../contributions/unexpected_behavior.md)). |
| `-i`, `--input` | `input.json` | Name of the input file to read for this phase, if you want to use something other than `input.json` (e.g. to keep several named presets in the same folder). |
| `-c`, `--cluster` | *(none)* | Overrides automatic cluster detection with the given machine keyword from `machine.json`, instead of matching it against the `hostname` entries. Mainly useful for testing a configuration, or running phases from a machine that isn't itself the target HPC machine (e.g. a login/front-end node with a different hostname). |

For example, to run the `exploration prepare` phase in debug mode with a custom input file:

```bash
python -m arcann_training exploration prepare -v 1 -i input_strict.json
```

### Initialization

| Phase | Description |
| --- | --- |
| `start` | Sets up the procedure: reads your chosen systems and options (including `nnp_program`), creates the `control/` folder and the first `000-training` folder. This is the phase you run to begin (detailed in the [Initialization](initialization.md) example). |
| `transition` | (Optional) Converts your existing data set from one storage format to the other (`set.000` ↔ `extxyz`) so you can switch between DeePMD-kit and MACE without re-labeling your configurations. See [Initialization](initialization.md#switching-architectures-the-transition-phase). |

### Exploration

| Phase | Description |
| --- | --- |
| `prepare` | Prepares the folders for running the exploration MDs of all **systems** (automatically generating input files required for each simulation). |
| `launch` | Submits the MD simulation to the specified partition of the cluster, usually with a SLURM array. |
| `check` | Verifies whether the exploration simulations have completed successfully. If any simulations ended abruptly, it indicates which ones, allowing the user to `skip` or `force` them (see [Exploration](#exploration)). |
| `deviate` | Reads the model deviation (maximum deviation between atomic forces predicted by the committee of NN) along the trajectories of each system and identifies configurations that are candidates (deviations within specified boundaries; see [Exploration](#exploration)). |
| `extract` | Extracts a user-defined number of candidate configurations per **system**, saving them to a `SYSNAME/candidates_SYSNAME.xyz` file for labeling and addition to the NNP training set. |
| `clean` | Removes files that are no longer required (optional). |

### Labeling

| Phase | Description |
| --- | --- |
| `prepare` | Prepares folders and files to run electronic structure calculations on identified candidates of each **system**, obtaining the energies and forces required to train the NNP. |
| `launch` | Submits the calculations with one SLURM array per **system**. |
| `check` | Verifies that calculations have completed successfully. If any calculations finished abruptly, it writes their index to a text file in the corresponding `SYSNAME/` folder. The user must decide whether to `skip` or resubmit manually each failed calculation before proceeding. |
| `extract` | Extracts the energies and forces from the CP2K outputs and builds the training data sets for each **system** (stored in the `$WORK_DIR/data/` folder), in the format matching your chosen architecture — `set.000` for DeePMD-kit or `extxyz` for MACE. |
| `clean` | Removes files that are no longer required and compresses the calculation outputs into an archive (optional). |

### Training

| Phase | Description |
| --- | --- |
| `prepare` | Prepares folders and files for training the user-defined number of independent NNPs to be used in the next iteration. |
| `launch` | Submits the training calculations to the cluster. With DeePMD-kit this uses `dp train`; with MACE it uses the MACE training program. |
| `check` | Verifies whether the training has completed successfully. If any training ended abruptly, it needs to be resubmitted manually to ensure the training finishes correctly. |
| `freeze` | (DeePMD-kit) Freezes the NN parameters into a binary file (`.pb` extension for the TensorFlow back-end) usable with LAMMPS and Python, via `dp freeze`. **With MACE this phase does nothing** — MACE models do not require freezing — but you still run it so the workflow stays consistent; ArcaNN simply reports that no job was needed. |
| `check_freeze` | Verifies that the freezing completed successfully. With MACE, it confirms the (no-op) phase is complete. |
| `compress` | (DeePMD-kit) [Compresses](https://doi.org/10.1021/acs.jctc.2c00102) the NNP by modifying the `.pb` file to enhance performance with minimal loss of accuracy, via `dp compress` (optional). With MACE, this phase instead produces the ready-to-use model file(s) for LAMMPS, in the format matching the `pair_style` you selected. |
| `check_compress` | Verifies that the compression (DeePMD-kit) or model preparation (MACE) completed successfully. |
| `increment` | Changes the iteration number in `control` and creates new `exploration`, `labeling`, and `training` folders for the next iteration. |
| `clean` | Removes files that are no longer required (optional). |

### Test

**Availability:** the optional **test** step is currently supported only for DeePMD-kit. If `nnp_program` is set to `"mace"`, ArcaNN will stop with a message that MACE evaluation is not yet supported.

| Phase | Description |
| --- | --- |
| `prepare` | Prepares folders and files for testing the performance of the current iteration's NNP on each *dataset* included in the training set. |
| `launch` | Submits the testing calculations using the `dp test` code from DeePMD-kit. If you need ["detail files"](https://docs.deepmodeling.com/projects/deepmd/en/r2/test/test.html) generated by `dp test`, include this directly in the `job_deepmd_test_...` file. |
| `check` | Verifies whether the calculations have completed successfully. |
| `clean` | Removes files that are no longer required (optional). If "detail files" weren't requested, the `XXX-test/` folder will be removed, as all the step information is consolidated in the `control/testing_XXX.json` file. Otherwise, the "detail files" will be compressed into .npy format and stored in `XXX-test/`. |

## Parameters

Parameters will need to be defined for most *phases* of each **step** (*e.g.*, length of MD simulations, temperature, number of CPU tasks for labeling calculations, etc.).
This is done via input files in the JSON format.
Executing *phase* without an input file will use all the default values (see the `exploration.json` file in `examples/inputs` for all `exploration` phases) and write them to a `default_json.json` file.
This file serves as a reminder of what the default values are.
After successfully executing a *phase*, a `used_input.json` file will be created, indicating which parameters `ArcaNN` used for that *phase*.
It will be appended with additional parameters after a subsequent *phase* (*e.g.*, after `exploration prepare`, a `used_input.json` is created and appended after `exploration deviate` with parameters specific to the `deviate` phase).
If you want to override the default values for a phase, simply create an `input.json` file with the parameters you want to change.
For example, to override the number of picoseconds for the `exploration prepare` phase, add an `input.json` file like this:

```JSON
{
    "exp_time_ps": 100
}
```

And run or rerun  `exploration prepare`: you will see within the  `used_input.json` that the requested parameters have been read, and that dependent parameters (*e.g.*, walltime) have been adjusted accordingly.

The parameters indicated in an `input.json` file will **always override** the default or auto-calculated ones, and for some of them, the values **will persist**.
For instance, if you provided an override for `max_candidates` in iteration `003`, it will be maintained in iteration `004` without requiring another `input.json`.

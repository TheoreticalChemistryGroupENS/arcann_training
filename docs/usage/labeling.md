# Labeling

In the labeling phase we compute the electronic energies, atomic forces and (sometimes) the stress tensor of the candidate configurations obtained in the exploration phase, using an electronic-structure program. ArcaNN supports **CP2K** (the default, described below) and **ORCA** (see [ORCA labeling](#orca-labeling) at the end of this page); you select the program with the `"labeling_program"` keyword of the `prepare` phase (`"cp2k"` or `"orca"`).

In the case you are performing the labeling step in a different HPC machine, don't forget to copy the data (you must also install the ArcaNN software and create a python environment!) beforehand:

```bash
rsync -rvu $WORK_DIR USER@OTHER_HPC_MACHINE:PATH_TO_WORKDIR
```

For this we need to go to the `XXX-labeling` folder and as usual run the `prepare` phase. It is very important to have a look at the `default_input.json` of the `prepare` phase to choose the computational resources to be used in the electronic structure calculations (number of nodes and MPI/OpenMP tasks). Note that the default values are insufficient for most condensed systems (due to the large number of atoms), so you should have previously determined the resources required by your specific system(s).

```JSON
{
    "step_name": "labeling",
    "user_machine_keyword_label": "mykeyword1",
    "job_email": "",
    "labeling_program": "cp2k",
    "walltime_first_job_h": [0.5, 0.5, 0.5],
    "walltime_second_job_h": [1.0, 1.0, 1.5],
    "nb_nodes": [1, 1, 1],
    "nb_mpi_per_node": [32, 32, 64],
    "nb_threads_per_mpi": [2, 2, 2]
}
```

The `"user_machine_keyword_label"` keyword corresponds to the partition in the HPC machine, The `"nb_mpi_per_node"` and `"nb_nodes"` keywords set the number of CPU nodes used for the labeling. The wall times should be set for the first iteration but can be guessed automatically later using the average time per CP2K calculation measured in the previous iteration.

Once you have executed this phase, folders will have been created for each subsystem within which there will be as many folders as candidate configurations (maximum number of 99999 per iteration), containing all required files to run CP2K. Make sure that you have prepared (and correctly named!) Slurm submission files for your machine in the `$WORK_DIR/user_files/` folder (see [Initialization](initialization.md)), from the template files.

## CP2K labeling

### Input files

Use the templates in `user_files/labeling_cp2k/`. CP2K labeling can run in **one or two steps**, and ArcaNN detects which one you want per system, automatically, from which template files are present:

- **Two steps (the typical/recommended setup, used in the examples):** provide both `1_SYSNAME_labeling_XXXXX_[cluster].inp` and `2_SYSNAME_labeling_XXXXX_[cluster].inp`. The first, quick calculation at a lower level of theory (smaller basis set, looser SCF settings) generates a wavefunction, which is then used as the SCF restart guess (`SCF_GUESS RESTART`) for the second calculation at your reference level of theory (larger basis set, tighter SCF settings). `"walltime_first_job_h"` and `"walltime_second_job_h"` set the wall time of each step respectively.
- **One step:** provide a single un-prefixed `SYSNAME_labeling_XXXXX_[cluster].inp` template (no `1_`/`2_` prefix) and no `2_...` template; ArcaNN then runs only that calculation (using `"walltime_first_job_h"`) and `"walltime_second_job_h"` is unused, similar to ORCA below.

A minimal two-step, reference-level (`2_...`) template looks like:

```text
&FORCE_EVAL
 METHOD QS
 &DFT
  WFN_RESTART_FILE_NAME labeling__R_PADDEDSTEP_-SCF.wfn
  CHARGE -2
  MULTIPLICITY 1
  &SCF
   SCF_GUESS RESTART
   ...
  &END SCF
  ...
 &END DFT
 STRESS_TENSOR DIAGONAL_ANALYTICAL
 &SUBSYS
  &CELL
   ABC [angstrom] _R_CELL_
  &END CELL
  &TOPOLOGY
   COORD_FILE_FORMAT XYZ
   COORD_FILE_NAME labeling__R_PADDEDSTEP_.xyz
  &END TOPOLOGY
  ...
 &END SUBSYS
 &PRINT
  &FORCES
   FILENAME =2_labeling__R_PADDEDSTEP_-Forces.for
   ...
  &END FORCES
  &STRESS_TENSOR
   FILENAME =2_labeling__R_PADDEDSTEP_-Stress_Tensor.st
   ...
  &END STRESS_TENSOR
 &END PRINT
&END FORCE_EVAL
```

A few points to keep in mind when adapting these templates to your system:

- `CHARGE`/`MULTIPLICITY` must be set to the correct values for each of your systems, just as for ORCA (see below); ArcaNN does not infer them.
- The `&PRINT`/`&FORCES` block is mandatory: ArcaNN parses forces from the `{step}_labeling_XXXXX-Forces.for` file it produces, not from the main `.out` file (which is only used for the energy).
- `STRESS_TENSOR DIAGONAL_ANALYTICAL` plus the `&PRINT`/`&STRESS_TENSOR` block are only needed if you want stress/virial data in your training set; if present, ArcaNN reads them from `{step}_labeling_XXXXX-Stress_Tensor.st`. **Stress tensor extraction is only supported for CP2K < 2024** — for CP2K `2024.x` and above ArcaNN currently cannot parse the stress tensor at all, so stick to an older CP2K version if you need virials.
- Whether a frame is stored as periodic or not is inferred from the `PERIODIC` keyword inside `&CELL`/`&POISSON` (if both are set, they must agree); `_R_CELL_` is filled in automatically by ArcaNN from the system's box.
- `_R_WALLTIME_`, `_R_NB_MPI_`, and `_R_PADDEDSTEP_` are placeholders that ArcaNN fills in automatically at `prepare` time from `"walltime_first_job_h"`/`"walltime_second_job_h"`, `"nb_mpi_per_node"`/`"nb_nodes"`, and the candidate index; keep them as-is in the templates.
- Wannier centers can optionally be extracted too (only relevant for DeePMD-kit models that use them): if a `{step}_labeling_XXXXX-Wannier.xyz` file is produced (via a `LOCALIZE`/Wannier-centers `&PRINT` setup in your input), ArcaNN will pick it up automatically.

### Job scripts

Use the templates in `job_labeling_cp2k_slurm/` (see [Iterative procedure prerequisites](./iter_prerequisites.md)). For two-step labeling, a single job script runs **both** CP2K steps sequentially: it executes the first (quick) calculation, saves its wavefunction, then executes the second (reference) calculation using that wavefunction as the SCF restart guess. For one-step labeling, adapt the script to only run that single calculation.

### Checking convergence

A candidate step is considered converged when its `{step}_labeling_XXXXX.out` file contains `"SCF run converged in"`; it is reported as **not converged** (rather than failed) if it instead contains `"SCF run NOT converged"`. For two-step labeling, `check` treats the two steps differently: failures in the **first** step alone only produce a warning (since that step is just used to generate a wavefunction guess for the second one — what actually matters for your training data is that the reference-level, second step converges), whereas failures in the **second** step (or in the only step, for one-step labeling) make `check` abort. As with ORCA, you can either rerun a candidate manually with a different setup or create an empty `skip` file in its folder to ignore it; keep running `check` until you get a "Success!" message.

### Extracting data

The `extract` phase reads the energy from the main `.out` file, forces from the `-Forces.for` file, and (if present) stress/virial from the `-Stress_Tensor.st` file and Wannier centers from the `-Wannier.xyz` file, all using the reference-level (second, for two-step labeling) step's output. Use the `extract` phase to set up everything for the training phase.

### Cleaning up

Run the `clean` phase to clean up your folder: it removes symlinks, job scripts, XYZ/input files, and CP2K's job stdout/stderr files (`CP2K.*`), then compresses everything else (except wavefunction files) into a `labeling_XXXXX_noWFN.tar.bz2` archive. CP2K wavefunctions (`.wfn` files) are **not** included in that archive and are not deleted automatically; `clean` prints ready-to-use commands to archive them separately if you want to keep them, e.g. as starting points for higher-level calculations later:

```bash
# Only the reference (2nd step) wavefunctions:
find ./ -name '2_*.wfn' | tar -cf labeling_XXXXX_WFN.tar --files-from -
# All wavefunctions (both steps):
find ./ -name '*.wfn' | tar -cf labeling_XXXXX_WFN.tar --files-from -
```

You can then delete the per-candidate subfolders once you are satisfied with the `_noWFN` archive (and have saved or don't need the wavefunction files). We have now augmented our total training set and might do a new training iteration and keep iterating until convergence is reached!

## ORCA labeling

If you set `"labeling_program"` to `"orca"`, ArcaNN drives [ORCA](https://www.faccts.de/docs/orca/6.1/manual/index.html) instead of CP2K for the electronic structure calculations. The workflow (`prepare`, `launch`, `check`, `extract`, `clean`) and the `default_input.json` keywords are the same as for CP2K, with the differences described below.

**Important limitation:** ORCA labeling in ArcaNN is always treated as an **isolated (non-periodic) molecular calculation**. Even though each system still has a cell/box defined in its configuration (for bookkeeping and dataset consistency), ArcaNN does not pass periodic boundary conditions to ORCA and the resulting training frames are flagged as non-periodic. If you need PBC in your reference calculations, use CP2K instead.

### Input files

Use the templates in `user_files/labeling_orca/` instead of `labeling_cp2k/`. Unlike CP2K, ORCA labeling is **single-step**: you only need one input file per candidate, named `1_SYSNAME_labeling_XXXXX_[cluster].inp` (there is no second, reference-level calculation, so `"walltime_second_job_h"` is unused for ORCA systems). A minimal template looks like:

```text
! B3LYP D3BJ 6-31G* TightSCF NoFrozenCore KeepDens Engrad

%MaxCore 1024

%pal nprocs _R_NB_MPI_ end

*xyzfile -2 1 labeling_XXXXX.xyz
```

A few points to keep in mind when adapting this template to your system:

- The **`Engrad`** simple-input keyword is mandatory: ArcaNN parses the energy and forces from the `.engrad` file that ORCA produces when this keyword is set, not from the main `.out` file.
- The `-2 1` on the `*xyzfile` line are the total **charge** and **spin multiplicity** of the system; set them to the correct values for each of your systems (ArcaNN does not infer or set them for you).
- `_R_NB_MPI_` is a placeholder that ArcaNN fills in automatically from `"nb_mpi_per_node"`/`"nb_nodes"` at `prepare` time; keep it as-is in the template.
- `KeepDens` (keep the density/`.gbw` file) is optional but convenient if you plan to reuse the wavefunctions, e.g. as a starting guess for higher-level calculations later.

### Job scripts

Use the templates in `job_labeling_orca_slurm/` instead of `job_labeling_cp2k_slurm/` (see [Iterative procedure prerequisites](./iter_prerequisites.md)).

### Checking convergence

A candidate is considered converged when its `1_labeling_XXXXX.out` file contains both `"ORCA TERMINATED NORMALLY"` and `"SCF CONVERGED"`. Unlike CP2K, ORCA does not distinguish between "not converged" and "failed" calculations — any candidate missing either string is simply reported as failed. Also, the `check` phase is **stricter for ORCA than for CP2K**: any failed/not-converged/still-running candidate makes `check` abort immediately (rather than just warning), so you must resolve or `skip` every problematic candidate before `check` can succeed. As with CP2K, you can either rerun the calculation manually with a different setup or create an empty `skip` file in the candidate's folder to ignore it; keep running `check` until you get a "Success!" message.

### Extracting data

The `extract` phase reads energies and forces for ORCA from the `.engrad` file. Two things are **not** available for ORCA, unlike CP2K:

- **Stress tensor/virial:** ORCA calculations produce no virial, so ORCA-labeled configurations never contribute stress/virial targets to the training set.
- **Wannier centers:** not extracted for ORCA (only relevant if you are training models that use Wannier centers/dipoles).

### Cleaning up

The `clean` phase behaves as for CP2K, except that ORCA's wavefunction files use the `.gbw` extension (instead of `.wfn`) and are named `1_*.gbw` (since there is only one step). ArcaNN also removes ORCA's job stdout/stderr files (`ORCA.*`) instead of CP2K's (`CP2K.*`). The archive/cleanup command it prints for you to keep only the reference wavefunctions is:

```bash
find ./ -name '1_*.gbw' | tar -cf labeling_XXXXX_WFN.tar --files-from -
```

Everything else — the overall folder structure, the maximum of 99999 candidates per iteration, and iterating until convergence — works exactly as described above for CP2K.

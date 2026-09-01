# Exploration

In the exploration phase we will generate new configurations (referred to as **candidates**) to include in the training set. For this we will perform MD simulations with either the LAMMPS (classical nuclei) or i-PI (quantum nuclei) softwares. Go to the current iteration exploration folder `XXX-exploration` created at the end of the previous training phase and execute the `prepare` phase to initialize the exploration.

The exploration phase is driven by the neural network potential you trained, so it uses the architecture you chose with `nnp_program` (DeePMD-kit or MACE). The commands and phases are the same in both cases. If you train with MACE, the only thing to be aware of is that the `pair_style` line in each `SYSTEM.in` LAMMPS input file selects how the MACE model is evaluated: ArcaNN supports `pair_style mace`, `pair_style mliap`, and `pair_style symmetrix/mace` (with the corresponding Kokkos/GPU variants also accepted). Choose the one supported by your LAMMPS build (see [Iterative procedure prerequisites](./iter_prerequisites.md)); ArcaNN reads it automatically and uses the matching model files prepared during training. Note that converting the trained MACE model(s) into the files LAMMPS needs happens on-demand during `prepare`, which requires the `mace`/`symmetrix` Python package(s) to be importable in the environment you run `prepare` from — this is a separate requirement from your LAMMPS build itself supporting the pair style.

Also, unlike DeePMD-kit — where a single LAMMPS run natively evaluates the whole committee of NNPs to get the deviation — MACE/`mliap`/`symmetrix` pair styles can only evaluate one model per dynamics run. To still get a committee deviation, ArcaNN runs the dynamics with one model and then automatically appends extra "rerun" passes that replay the resulting trajectory through each of the other trained models. This is handled for you, but it means MACE exploration jobs do noticeably more compute work per trajectory than the LAMMPS run time alone would suggest.

**MACE currently requires `exploration_type: "lammps"`.** The `i-PI` and `sander_emle` exploration types described below are only implemented for DeePMD-kit; if `nnp_program` is `"mace"` and a system's `exploration_type` is not `"lammps"`, ArcaNN stops with an error.

For the first exploration phase we might want to generate only a few candidate configurations to check whether our initial NNP are stable enough to give physically meaningful configurations. We might as well want to use a relatively strict error criterion for candidate selection.
To change these parameters you can create a `input.json` file , indicating the values to be updated, and run the `prepare` phase again. If you want to keep the default values you only need to run the `prepare` phase once. As for the Initialization and Training steps, this will generate a `used_input.json` file:

```JSON
{
    "step_name": "exploration",
    "user_machine_keyword_exp": "mykeyword1",
    "job_email": "",
    "atomsk_path": "PATH_TO_THE_ATOMSK_BINARY",
    "vmd_path": "PATH_TO_THE_VMD_BINARY",
    "exploration_type": ["lammps", "lammps", "lammps"],
    "traj_count": [2, 2, 2],
    "temperature_K": [273.0, 300.0, 300.0],
    "timestep_ps": [0.0005, 0.0005, 0.0005],
    "previous_start": [true, true, true],
    "disturbed_start": [false, false, false],
    "print_interval_mult": [0.01, 0.01, 0.01],
    "job_walltime_h": [-1, -1, -1],
    "exp_time_ps": [10, 10, 10],
    "max_exp_time_ps": [400, 400, 400],
    "max_candidates": [50, 50, 100],
    "sigma_low": [0.1, 0.1, 0.1],
    "sigma_high": [0.8, 0.8, 0.8],
    "sigma_high_limit": [1.5, 1.5, 1.5],
    "ignore_first_x_ps": [0.5, 0.5, 0.5],
    "init_exp_time_ps": [-1, -1, -1],
    "init_job_walltime_h": [-1, -1, -1],
    "disturbed_candidate_value": [0.5, 0, 0],
    "disturbed_start_value": [0.0, 0.0, 0.0],
    "disturbed_start_indexes": [[], [], []],
    "disturbed_candidate_indexes": [[], [], []]
}
```

- `"traj_count"`: number of independent MD trajectories launched per **system** and per trained NNP (so the total number of runs for a system is `traj_count × nnp_count`).
- `"temperature_K"` and `"timestep_ps"`: temperature (K) and integration timestep (ps) of the MD run. As noted above, if you give two values they are used for classical (LAMMPS) and quantum-nuclei (i-PI) exploration respectively.
- `"exp_time_ps"`: the requested simulation length, in ps, for this iteration (`-1` lets ArcaNN pick a value automatically from previous iterations). `"max_exp_time_ps"` is a hard ceiling on how long any single run is allowed to run for, regardless of `exp_time_ps`.
- `"job_walltime_h"`: wall time requested for the MD `Slurm` job, in hours. `-1` (the default) lets ArcaNN estimate it automatically — 1 h for the very first exploration of a system, then from the measured time-per-step of the previous iteration's runs, scaled to the new `exp_time_ps`.
- `"print_interval_mult"`: fraction of the total number of MD steps between two saved frames/deviation evaluations (e.g. `0.01` means a frame is written roughly every 1% of the trajectory). Lower values sample the trajectory more finely (more candidate opportunities, larger output files).
- `"ignore_first_x_ps"`: an equilibration/burn-in period, in ps, at the start of each trajectory that is excluded from candidate selection in the `deviate` phase.
- `"previous_start"`: whether this iteration's runs may start from a configuration extracted from the *previous* iteration's exploration (rather than always from the original `SYSTEM.lmp`/`SYSTEM.xml`). `"disturbed_start"`: same, but starting from a *disturbed* (randomly perturbed) configuration produced by a previous iteration that used a non-zero `disturbed_start_value` (see below). Both are ignored on the very first iteration.
- `"sigma_low"`, `"sigma_high"` and `"sigma_high_limit"` indicate the deviation acceptance criteria in eV/Ang for the candidate selection: below `sigma_low` a frame is considered accurate enough to discard; between `sigma_low` and `sigma_high` it is a good candidate; between `sigma_high` and `sigma_high_limit` it is a candidate but flagged as more borderline/"disturbed"-worthy. `sigma_high_limit` is not simply a fourth bucket, though: as soon as the deviation of a frame *crosses* `sigma_high_limit`, ArcaNN treats the trajectory as having diverged from that point on — all subsequent frames are discarded outright (not even considered as candidates), and if the crossing happens within `ignore_first_x_ps`, the *entire* trajectory is discarded from the candidate statistics.
- `"max_candidates"` indicates the maximum number of candidates that can be selected.
- The values in `disturbed_start_value` are used to disturb the starting structures for the next iteration. A non-zero value sets the maximal amplitude of the random translation vector that will be applied to each atom (a different vector for each atom) in Å. `"disturbed_start_indexes"` restricts this disturbance to specific (zero-based) atomic indices instead of the whole configuration, the same way `"disturbed_candidate_indexes"` does for `disturbed_candidate_value` (see below).

**Note:** the `vmd_path` keyword is not needed if `vmd` is immediately available in our path when executing the `extract` phase (loaded as a module for example). Similarly, we can remove `atomsk_path` if `atomsk` is already in the path.

Some of the phases are slightly different if you use LAMMPS or i-PI, both phase workflows are detailed below.

## LAMMPS: classical nuclei simulations

Once you are satisfied with your exploration parameters (see example below) you can execute the next exploration phases: `launch` to run MD trajectories with each subsystem  and `check` (once the `Slurm` MD jobs are done!). If the `check` phase is successful, you can move on to the `deviate` phase, where you can set important parameters for candidate selection. Once again you can modify these keywords by the creation (or modification if you already created one for a previous phase) of a `default_input.json` file and re-executing the `deviate` phase. In the `extract` phase an important choice can be made: whether to include "disturbed" candidates in the training set or not. This is done by changing the `disturbed_start_value` and `disturbed_candidate_value` variables from the defaults (0.0) and will include a set of candidates generated by applying a random perturbation to those obtained in the MD trajectories (this will multiply by 2 the number of selected candidates, make sure that the `disturbed_start_value` that you choose will still give physically meaningful configurations, otherwise you will deteriorate your NNP!). Once you execute this phase a `candidates_XXX_SUBSYS.xyz` file will be created in each subsystem directory containing the candidate configurations that will be added to the training set (you might want to check that they make sense!). You can also disturb only some atoms in the configuration in which case you will need to write their (zero-based) atomic indices in the `disturbed_candidate_indexes` variable. The `clean` phase can be executed to clean up all the temporary files. A `control/exploration_XXX.json` file will be written recording all the exploration parameters. You can now move on to the labeling phase! (Don't forget to keep your local folder updated so that you can analyze all these results)

## i-PI quantum nuclei simulations (Under development)

Simulations explicitly including nuclear quantum effects by path-integral molecular dynamics with i-PI are quite similar to classical nuclei simulations with LAMMPS. Although the i-PI input files are different (see [i-PI](https://ipi-code.org/)), the `prepare`, `launch` and `check` phases can be done exactly as previously (see [LAMMPS classical nuclei simulations](#lammps-classical-nuclei-simulations) above). **Available with DeePMD-kit only** (see the MACE note above).

**Note:** support for the phases past `check` (`deviate`/`extract`) for i-PI exploration is still under development.

## Sander/EMLE: QM/MM exploration

ArcaNN can also drive QM/MM exploration through [Sander](https://ambermd.org/) coupled to [EMLE](https://github.com/chemle/emle-engine) (`exploration_type: "sander_emle"`). The `prepare`, `launch`, `check`, `deviate`, `extract` and `clean` phases are run in the same order as for LAMMPS; no `select_beads`/`rerun` step is needed. **Available with DeePMD-kit only** (see the MACE note above).

For each **system** using this mode, place the following in `user_files/` (in addition to the usual `SYSTEM.lmp`; see [Iterative procedure prerequisites](./iter_prerequisites.md)), using templates from `examples/user_files/exploration_sander_emle/`:

- `SYSTEM.in`: the Sander MD input file.
- `SYSTEM.yaml`: the EMLE backend configuration (read/updated by ArcaNN — the `deepmd_model`, `deepmd_deviation`, `model`, `energy_file`, `log_file`, `qm_xyz_file` and `qm_xyz_frequency` keys are filled in automatically, the rest is yours to set).
- `SYSTEM.prmtop`: the Amber topology file for the system.
- `SYSTEM.mat`: the trained EMLE model file.
- `SYSTEM.ncrst`: the starting restart coordinates (Amber `ncrst` format).

Optional `plumed_SYSTEM.dat` / `plumed_*_SYSTEM.dat` files are supported exactly as for LAMMPS if referenced from `SYSTEM.in`.

**Limitation:** unlike LAMMPS exploration, Sander/EMLE does not yet pick a new starting structure from the previous iteration's candidates — every iteration restarts from the same `SYSTEM.ncrst`, and `previous_start` is ignored for these systems (ArcaNN logs a warning). This is a known gap in the current implementation.

## Handling failed or borderline runs: `skip` and `force`

During `check`, a run that ArcaNN cannot validate is handled in one of two ways. This applies the same way regardless of `exploration_type` (LAMMPS, i-PI, or Sander/EMLE):

- **`skip`**: ArcaNN itself creates an empty `skip` file in a run's folder when the trajectory output (the `.dcd`/`.nc` file) exists but is unreadable/corrupted. You can also create this file yourself (`touch RUNFOLDER/skip`) to discard a run you know is unusable; skipped runs are excluded from candidate selection and counted as `skipped_count` in `control/exploration_XXX.json`.
- **`force`**: create this file yourself (`touch RUNFOLDER/force`) when a run looks like it failed (e.g., no "Total wall time" line found in the LAMMPS log, no "Average timings for all steps" line in the `sander_emle` log, or no "SIMULATION: Exiting cleanly" line in the i-PI log) but you have checked it and consider it usable anyway. `check` then accepts it as completed instead of reporting it as failed, and counts it under `forced_count`.

Re-run `check` after adding either file; it will report how many runs were skipped/forced.

**MACE caveat:** with `nnp_program: "mace"`, `check` only inspects the main dynamics LAMMPS log, not the per-model rerun trajectories used to build the committee deviation (see the note on reruns above). Forcing a run whose main log looks incomplete does not guarantee those rerun passes finished for every model in the committee; if one is missing or truncated, the later `deviate` phase can fail or produce an inconsistent deviation for that run. Check that all `SYSTEM_mace_forces_model*.lammpstrj` files exist and have matching frame counts before forcing a MACE run.

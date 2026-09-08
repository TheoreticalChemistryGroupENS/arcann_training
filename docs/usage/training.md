# Training

During the training procedure you will train neural networks on the data sets that you have generated so far (or on the initial ones only for the `000-training`). ArcaNN uses the architecture you selected during initialization with `nnp_program`: **DeePMD-kit** (default) or **MACE**. In order to do this go to the current iteration training folder `XXX-training`.

There are 9 phases (see [Iterations, Steps and Phases of the Iterative Procedure](start.md)) that you must now execute in order after having optionally modified the `input.json` file to define the relevant parameters (in case you want something different from the defaults, which are written to `default_input.json` in the `prepare` phase). The input keywords that you should check the most carefully are those related to the first phase `prepare`, as this sets all the important parameters for the training. Some phases will simply submit `Slurm` jobs. You must wait for the jobs to finish before executing the next phase (generally this will be a check phase that will tell you whether jobs have failed or are currently running). Once you have executed the first 8 phases — `prepare`, `launch`, `check`, `freeze`, `check_freeze`, `compress`, `check_compress`, and `increment` — the training iteration is done! Executing the 9th phase, `clean`, is optional, as this only removes intermediary files.

**Note on the phases for MACE:** the sequence of phases is identical for both architectures, so you run the same commands in the same order. Two of them behave differently, though: the `freeze` phase does nothing for MACE (these models do not need freezing — you still run it, and ArcaNN reports that no job was launched; note that on the *first* run of `freeze` for a MACE model you may still be prompted to confirm before it marks the phase done, even though no job is actually submitted), and the `compress` phase prepares the LAMMPS-ready MACE model file(s) rather than compressing a DeePMD-kit graph — this requires the `.in` LAMMPS input templates to already be present in `user_files/` so ArcaNN can detect which `pair_style` to produce files for; if none are found, `compress` (and `check_compress`) aborts with an error.

After running the `initialization` step described in the previous example, you must now perform the first training phase. Update (or copy for the first time) the full `$WORK_DIR` from your local machine to your HPC machine (where you must have also a copy of this repository and an environment in which it is installed):

```bash
rsync -rvu $WORK_DIR USER@HPC-MACHINE:/PATH/TO/WORK_DIR
```

This step is not mandatory: if you run ArcaNN directly on the HPC machine (rather than driving it from your local machine), you can skip the `rsync` and work in `$WORK_DIR` in place.

Now go to the empty `000-training` folder created by the script execute the `prepare` phase:

```bash
python -m arcann_training training prepare
```

This will create three folders `1/`, `2/` and `3/` and a copy of your `data/` folder, as well as a `default_input.json` file containing the default training parameters. If you want to modify some of the default values you can create an `input.json` file from the `default_input.json` file. A typical example looks like this:

```JSON
{
    "step_name": "training",
    "user_machine_keyword_train": "mykeyword1",
    "user_machine_keyword_freeze": "mykeyword2",
    "user_machine_keyword_compress": "mykeyword2",
    "job_email": "",
    "use_initial_datasets": true,
    "use_extra_datasets": false,
    "deepmd_model_version": "2.1",
    "mace_model_version": "0.3.14",
    "job_walltime_train_h": 4,
    "mean_s_per_step": 0.10,
    "start_lr": 0.001,
    "stop_lr": 1e-06,
    "decay_rate": 0.9172759353897796,
    "decay_steps": 5000,
    "decay_steps_fixed": false,
    "numb_steps": 400000,
    "max_num_epochs": 500,
    "numb_test": 0
}
```

Here the `"user_machine_keyword_*"` values should match a keyword defined in your `machine.json` (see [HPC Configuration](../getting-started/hpc_configuration.md)). Note that the more performant GPUs should ideally be used for training, while the other steps could be allocated to less performant GPUs or even to CPUs. In this example we used a user-chosen walltime of 4 h (instead of the default `-1`, which calculates the job walltime automatically based on your previous trainings).

`"mean_s_per_step"` is the estimated training time per step (in seconds) used to auto-compute `job_walltime_train_h` when it is left at `-1`: ArcaNN multiplies it by `numb_steps`, rounds up to the next hour, and (from the second training iteration onward) adds a 50% safety margin — but only if `mean_s_per_step` was *not* explicitly set by you; if you do set it, ArcaNN uses `ceil(numb_steps * mean_s_per_step)` directly, with no hour-rounding and no safety margin. **For DeePMD-kit**, you normally never need to set it: the default (`0.10` s/step) is only used for the very first training (`000-training`); every later iteration measures the real per-step time from the previous training run and uses that instead. **For MACE, this automatic measurement currently does not happen** — the per-step timing is only ever recorded for DeePMD-kit runs, and MACE's is unconditionally recorded as `0` s/step after every iteration. In practice this means the very first MACE training (`000-training`) uses the `0.10` s/step default, but **every MACE training from the second iteration onward gets an auto-computed walltime of 0 hours** unless you override it. If you are training with MACE, always set `"job_walltime_train_h"` (or a realistic `"mean_s_per_step"`) explicitly rather than relying on the automatic estimate.

You only need the keywords that correspond to the architecture you selected with `nnp_program` for their architecture-specific effect, but note that `"numb_steps"` is used by **both**: besides being DeePMD-kit's number of training steps, it also feeds directly into the walltime formula above regardless of `nnp_program`, so leave it at a sensible value even when training with MACE.

- **DeePMD-kit parameters:** `deepmd_model_version` (the DeePMD-kit version to use, given as text, e.g. `"2.1"`; supported versions are `2.0`–`3.0`, anything else aborts), together with the DeePMD-kit training parameters `start_lr`, `stop_lr`, `decay_rate`, `decay_steps`, `decay_steps_fixed`, `numb_steps` (the number of training steps) and `numb_test`. If you leave `deepmd_model_version` out, ArcaNN picks the highest version for which it finds a `dptrain_VERSION.json` file in `user_files/`.
- **MACE parameters:** `mace_model_version` (the MACE version to use, e.g. `"0.3.14"`, which must match a `mace_MACEVERSION.yml`/`.yaml` file in `user_files/`; supported versions are `0.3.x`, anything `>= 0.4.0` or `< 0.3.0` aborts) and `max_num_epochs` (the maximum number of training epochs, 500 by default). Most of the model's behavior is defined inside your `mace_MACEVERSION.yml` file, but note that ArcaNN always overwrites the `energy_key`/`forces_key`/`virials_key` entries in it to its own `REF_energy`/`REF_forces`/`REF_virials` convention, regardless of what you set there. If you leave `mace_model_version` out, ArcaNN picks the highest available version. A `foundation_model` key is also supported for fine-tuning from a pretrained MACE model.

The fraction of data used for validation is controlled by `validation/training_split`, which you set during [Initialization](./initialization.md) — for MACE, note that this validation subset is itself split in half again internally (one half used as the actual training-time validation set, the other set aside as a held-out test set), so the effective MACE validation fraction is half of what `validation/training_split` specifies. We can then execute all the other phases in order (waiting for `Slurm` jobs to finish!).

**Notes:**

- At some point during the iterative procedure we might want to get rid of our initial data sets, we would only need to set the `use_initial_datasets` variable to `False`.
- We might also have generated some data independently from the iterative procedure that we might want to start using, this can be done by copying the corresponding DeePMD-kit systems to `data/`, prefixing their names by `extra_` and setting the `use_extra_datasets` variable to `True`.
- **Ad-hoc datasets:** you can also add data for a brand-new **system**—one that was never declared in `systems_auto` during [Initialization](initialization.md)—without reinitializing the procedure. Simply place a dataset folder under `$WORK_DIR/data/` named `SYSTEM_ITERATION` (e.g. `newTS_0`), matching the `data_format` you chose during Initialization. Since its name does not match any entry in `systems_auto`, ArcaNN automatically classifies it as an ad-hoc dataset (tracked internally as `systems_adhoc`) the next time `training prepare` runs, and it is picked up for training like any other dataset—no `input.json` flag needed. The counts are reported in `control/training_XXX.json` as `added_adhoc_count` and `added_adhoc_iter_count`. Keep in mind this only adds static, already-labeled data to the training set: unlike a **system** declared in `systems_auto`, an ad-hoc system does *not* get automated exploration or labeling in later iterations—both of those steps only ever consider `systems_auto`. If you need the full explore-label-train loop for a new subsystem, you must reinitialize as described in [Iterative procedure prerequisites](iter_prerequisites.md).
- At the end of the step the last phase `increment` will create the folders needed for the next iteration, save the current NNPs into the `$WORK_DIR/NNP` folder, and write a `control/training_XXX.json` file with all parameters used during training. The saved file naming depends on `nnp_program`: for DeePMD-kit these are graph files `graph_[nnp_count]_XXX[_compressed].pb`; for MACE these are `model_[nnp_count]_XXX.model`, plus (once compressed) the LAMMPS-ready `.model-lammps.pt`/`-mliap_lammps.pt`/`.model.json` files matching your `pair_style`.
- The `clean` phase currently only removes DeePMD-kit artifacts (`.pb` files and related temporary files); it does not yet clean up MACE's model files, so if you use MACE you may need to remove leftover files from the training folders manually.

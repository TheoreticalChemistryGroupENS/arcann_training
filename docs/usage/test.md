# Test

**Availability:** the **test** step is currently supported only for DeePMD-kit. If you selected MACE (`nnp_program: "mace"`) during [Initialization](./initialization.md), `prepare` will stop with a message that MACE evaluation is not yet supported — in the current version this surfaces as an unhandled-exception traceback rather than a clean one-line error, but the outcome is the same: the phase does not run. Support for evaluating MACE models is planned for a future version.

It is possible to perform tests at every iteration of the learning procedure (the code will create `XXX-test/` folders at every `increment` phase of a `training` step). However, doing this at every iteration is rather time consuming and is not really necessary (although you should obviously test your converged NNP thoroughly).

**Prerequisites:** `prepare` requires that `training increment` has already been run for the iteration you want to test (it aborts otherwise), and that the corresponding NNP graph file(s) already exist in `$WORK_DIR/NNP/`. It also expects the test folder's own `data/` subfolder (`XXX-test/data/`, distinct from the top-level `$WORK_DIR/data/`) — this is normally set up automatically, but if it is ever missing or empty, later phases will fail rather than report a clear error, so check it if `check` behaves unexpectedly.

To run it, go to the `XXX-test/` folder of the iteration you want to test and run the phases in order (see [Iterations, Steps and Phases](./start.md#test)): `prepare`, `launch`, `check`, then optionally `clean`. As with the other steps, running `prepare` without an `input.json` writes the defaults to `default_input.json`:

```JSON
{
    "step_name": "test",
    "user_machine_keyword_test": "mykeyword2",
    "job_email": "",
    "job_walltime_h": 2.0,
    "is_compressed": false,
    "deepmd_model_version": 0.0
}
```

- `"user_machine_keyword_test"` should match a keyword defined in your `machine.json` (see [HPC Configuration](../getting-started/hpc_configuration.md)); testing is generally light enough to run on CPU or a less performant GPU partition.
- `"job_walltime_h"` sets the wall time for the `dp test` job.
- `"is_compressed"` selects whether the compressed or the uncompressed graph of that iteration's NNP is tested.
- `"deepmd_model_version"` is recorded for bookkeeping in `used_input.json`/`control/testing_XXX.json`, but **does not currently select which DeePMD-kit version is actually used to run the test** — regardless of what you set here, ArcaNN always loads the version the graph was originally trained with (from that iteration's `control/training_XXX.json`). Do not rely on this keyword to test a graph with a different DeePMD-kit version than the one it was trained with.

**Note on `"is_compressed"` / `"deepmd_model_version"`:** the `0.0`/`false` shown above are placeholders, not meaningful defaults. If you run `prepare` with **no `input.json` file at all**, ArcaNN ignores them and instead copies `"is_compressed"` and `"deepmd_model_version"` straight from that iteration's `control/training_XXX.json` — i.e. it tests the graph exactly as it was produced (compressed or not, same DeePMD-kit version), which is what you want in most cases. However, as soon as you create an `input.json` for **any** reason (even just to change `"job_walltime_h"`), that automatic inheritance is skipped for both keys: any of the two you didn't explicitly set will fall back to the literal `0.0`/`false` placeholders instead of the training values. So if you use an `input.json` at all, explicitly set `"is_compressed"` (and `"deepmd_model_version"`, for the record — though as noted above it has no effect on which version actually runs) to match your training run.

`launch` submits the testing job using `dp test` (see [DeePMD-kit test documentation](https://docs.deepmodeling.com/projects/deepmd/en/r2/test/test.html)); if you need the ["detail files"](https://docs.deepmodeling.com/projects/deepmd/en/r2/test/test.html) it can generate, add the corresponding flag directly to your `job_deepmd_test_ARCHTYPE_myHPCkeyword.sh` submission file (from the `job_test_deepmd_slurm/` templates — note the file itself is named `job_deepmd_test_...`, not `job_test_deepmd_...`). `check` verifies the job completed successfully. `clean` removes temporary files; if detail files were not requested it also removes the whole `XXX-test/` folder, since all the results are already consolidated into `control/testing_XXX.json`.

A full worked example (equivalent to the [SN2](../examples/sn2.md) walkthrough for the main steps) is not available yet, sorry!

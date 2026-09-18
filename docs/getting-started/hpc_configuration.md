# HPC Configuration

ArcaNN is designed for use on one or several HPC machines, whose specific configurations must be specified by the user through a `machine.json` file.
A general example file can be found in the [GitHub Repository](https://github.com/TheoreticalChemistryGroupENS/arcann_training/blob/main/examples/user_files/machine.json).
You should modify this file to suit your setup and then copy it to the `user_files/` folder in your working directory (see later in [Usage](../usage/iter_prerequisites.md)).

## Structure of the `machine.json` File

The `machine.json` file is organized as a JSON dictionary with one or more keys that designate different HPC machines. The typical structure looks like this:

```json
{
    "myHPCkeyword1": {ENTRIES THAT DESCRIBE HPC 1},
    "myHPCkeyword2": {ENTRIES THAT DESCRIBE HPC 2},
    // Additional machines can be added here
}
```

Each key in the JSON file is a short string designating the name of the machine (e.g., `"myHPCkeyword1"`, `"myHPCkeyword2"` are the names of 2 different HPC machines).
The value associated with each key is a dictionary indicating the configuration entries for running jobs on the corresponding HPC machine.

Below is an example of the initial entries for an HPC machine using a SLURM job scheduler:

```json
{
    "myHPCkeyword1": {
        "hostname": "myHPC1",
        "walltime_format": "hours",
        "job_scheduler": "slurm",
        "launch_command": "sbatch",
        "max_jobs": 200,
        "max_array_size": 500,
        "mykeyword1": {
            "project_name": "myproject",
            "allocation_name": "myallocationgpu1",
            "arch_name": "a100",
            "arch_type": "gpu",
            "partition": "mypartitiongpu1",
            "subpartition": "mysubpartitiongpu1",
            "qos": {
                "myqosgpu1": 72000,
                "myqosgpu2": 360000
            },
            "valid_for": ["training"],
            "default": ["training"]
        },
        "mykeyword2": { /* Additional partition configurations */ }
    },
    /* Additional HPC machines can be added here */
}
```

## HPC Entry

Each HPC machine entry contains a JSON dictionary where each key corresponds to a configuration entry.

- **hostname**: A substring contained in the output of the command `hostname` or `python -c "import socket ; print(socket.gethostname())"`. This should match your machine's name.
- **walltime_format**: The unit of time (e.g., hours) used to specify wall time on the cluster.
- **job_scheduler**: The job scheduler used by your HPC machine. **In the current version, only `"slurm"` is actually supported** — job submission, array jobs, and all job-file templates in `examples/user_files/job*` are Slurm-specific. Other values are accepted by the config parser but have no corresponding logic, so set this to `"slurm"`.
- **launch_command**: The command for submitting jobs, i.e. `sbatch` for `Slurm`.
- **max_jobs**: Maximum number of jobs per user allowed by the scheduler. Can also be a user-defined safety limit.
- **max_array_size**: Maximum number of jobs in a single job array. This is important for `Slurm` as ArcaNN relies heavily on job arrays.

## Resource Configuration

Several resources can be available for calculation within the same HPC machine.
Each available resource in the HPC machine is represented by a key (e.g., `"mykeyword1"`) and includes:

- **project_name**: Name of the project using the HPC resources.
It will correspond to the `_R_PROJECT_` keyword in the `#SBATCH --account=_R_PROJECT_` line of the slurm job.
- **allocation_name**: Allocation or account name, typically used in large HPC facilities.
It will correspond to the `_R_ALLOC_` keyword in the `#SBATCH --account=_R_PROJECT_@_R_ALLOC_` line of the slurm job.
- **arch_name**: Architecture name (e.g., `a100` for GPU nodes).
- **arch_type**: Architecture type (e.g., `gpu` or `cpu`).
- **partition**: The partition on the HPC machine.
- **subpartition**: (Optional) Subpartition within the main partition.
- **qos**: Quality of Service settings, with corresponding time limits in seconds.
- **valid_for**: Specifies the steps this partition is valid for (e.g., `["training", "freezing", "compressing", "exploration", "test", "labeling"]`).
- **default**: Indicates the default partition for specific steps.

## Customization and Submission Files

You can add multiple partition configurations as needed. For example, `"mykeyword1"` could represent a GPU partition using A100 GPU nodes, which is used for training unless a different partition is specified.

If your HPC setup does not include projects, allocations, partitions, or subpartitions, you can omit the corresponding keywords.

To run ArcaNN on your HPC machine, you must provide example submission files tailored to your system.
These files should be modeled after the `examples/user_files/job*/*.sh` files and **must include the replaceable strings** indicated by a `_R_` prefix and suffix.
Place these files in the `$WORK_DIR/user_files/` folder, which you must create to use ArcaNN for a specific system (see [Usage](../usage/iter_prerequisites.md)).

## Common `_R_` placeholders (every `Slurm` job script)

Whatever the step (`exploration`, `labeling`, `training`, `test`) or phase (`prepare`, `freeze`, `compress`, ...), every `Slurm` job/job-array script generated by ArcaNN goes through the same header-replacement routine, which fills in the following placeholders from the resource entry selected in `machine.json` (see [Resource Configuration](#resource-configuration) above) and from that step's `"job_email"` keyword. Keep them all in your submission-file templates:

| Placeholder | Filled from | Notes |
| --- | --- | --- |
| `_R_PROJECT_` | `"project_name"` | |
| `_R_ALLOC_` | `"allocation_name"` | |
| `_R_PARTITION_` | `"partition"` | If `"partition"` is omitted/`null` for the resource, the line containing this placeholder is removed entirely instead of being filled in. |
| `_R_SUBPARTITION_` | `"subpartition"` | Same as `_R_PARTITION_`: the line is removed if `"subpartition"` is not set. |
| `_R_QOS_` | `"qos"` | ArcaNN picks the smallest QoS in the dictionary whose time limit is `>=` the job's estimated wall time; if none is large enough, it uses the largest available QoS instead and logs a warning. |
| `_R_WALLTIME_` | the step's computed/requested wall time | Formatted as `HH:MM:SS` if the machine's `"walltime_format"` contains `"hours"`, otherwise as a plain number of seconds. |
| `_R_EMAIL_` | `"job_email"` (from the step's `input.json`) | If `"job_email"` is empty, this placeholder's line **and** every other line containing the substring `mail` (e.g. `--mail-type`) are removed instead of being filled in. |

Every phase also fills in additional, step-specific placeholders (resource counts, input/output file names, program versions, etc.) — see the "template placeholders" reference at the end of each step's page in [Usage](../usage/iter_prerequisites.md) (Exploration, Labeling, Training, Test).

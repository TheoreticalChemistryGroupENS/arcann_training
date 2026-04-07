"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2026/02/06
"""

# Standard library modules
import logging
import os
import random
import shutil
import sys
from copy import deepcopy
from pathlib import Path

# Non-standard library imports
import numpy as np

# Local imports
from arcann_training.common.check import validate_step_folder
from arcann_training.common.dataset import Dataset
from arcann_training.common.filesystem import check_directory
from arcann_training.common.json import (
    backup_and_overwrite_json_file,
    get_key_in_dict,
    load_default_json_file,
    load_json_file,
    replace_values_by_key_name,
    write_json_file,
)
from arcann_training.common.list import (
    replace_substring_in_string_list,
    string_list_to_textfile,
    textfile_to_string_list,
)
from arcann_training.common.machine import (
    get_machine_keyword,
    get_machine_spec_for_step,
)
from arcann_training.common.slurm import replace_in_slurm_file_general
from arcann_training.common.yaml import load_yaml_file, write_yaml_file
from arcann_training.training.utils import (
    calculate_decay_rate,
    calculate_decay_steps,
    generate_training_json,
    validate_deepmd_config,
    validate_mace_config,
)


def main(
    current_step: str,
    current_phase: str,
    deepmd_iterative_path: Path,
    fake_machine=None,
    user_input_json_filename: str = "input.json",
):
    # Get the logger
    arcann_logger = logging.getLogger("ArcaNN")

    # Get the current path and set the training path as the parent of the current path
    current_path = Path().resolve()
    training_path = current_path.parent

    # Log the step and phase of the program
    arcann_logger.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()}."
    )
    arcann_logger.debug(f"Current path :{current_path}")
    arcann_logger.debug(f"Training path: {training_path}")
    arcann_logger.debug(f"Program path: {deepmd_iterative_path}")
    arcann_logger.info("-" * 88)

    # Check if the current folder is correct for the current step
    validate_step_folder(current_step)

    # Get the current iteration number
    padded_curr_iter = Path().resolve().parts[-1].split("-")[0]
    curr_iter = int(padded_curr_iter)
    arcann_logger.debug(f"curr_iter, padded_curr_iter: {curr_iter}, {padded_curr_iter}")

    # Load the default input JSON
    default_input_json = load_default_json_file(
        deepmd_iterative_path / "assets" / "default_config.json"
    )[current_step]
    default_input_json_present = bool(default_input_json)
    if (
        default_input_json_present
        and not (current_path / "default_input.json").is_file()
    ):
        write_json_file(
            default_input_json, (current_path / "default_input.json"), read_only=True
        )
    arcann_logger.debug(f"default_input_json: {default_input_json}")
    arcann_logger.debug(f"default_input_json_present: {default_input_json_present}")

    # Load the user input JSON
    if (current_path / user_input_json_filename).is_file():
        user_input_json = load_json_file((current_path / user_input_json_filename))
    else:
        user_input_json = {}
    user_input_json_present = bool(user_input_json)
    arcann_logger.debug(f"user_input_json: {user_input_json}")
    arcann_logger.debug(f"user_input_json_present: {user_input_json_present}")

    # Make a deepcopy of it to create the used input JSON
    current_input_json = deepcopy(user_input_json)

    # Get control path and load the main JSON
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))

    nnp_program: str = main_json["nnp_program"]

    arcann_logger.info(f"Using {nnp_program} as NNP software")
    arcann_logger.info("-" * 88)

    # Load the previous training JSON
    if curr_iter > 0:
        prev_iter = curr_iter - 1
        padded_prev_iter = str(prev_iter).zfill(3)
        previous_training_json = load_json_file(
            (control_path / f"training_{padded_prev_iter}.json")
        )
        del prev_iter, padded_prev_iter
    else:
        previous_training_json = {}

    # Get the machine keyword (Priority: user > previous > default)
    # And update the merged input JSON
    user_machine_keyword = get_machine_keyword(
        current_input_json, previous_training_json, default_input_json, "train"
    )
    # Set it to None if bool, because: get_machine_spec_for_step needs None
    user_machine_keyword = (
        None if isinstance(user_machine_keyword, bool) else user_machine_keyword
    )
    arcann_logger.debug(f"user_machine_keyword: {user_machine_keyword}")

    # From the keyword (or default), get the machine spec (or for the fake one)
    (
        machine,
        machine_walltime_format,
        machine_job_scheduler,
        machine_launch_command,
        machine_max_jobs,
        machine_max_array_size,
        user_machine_keyword,
        machine_spec,
    ) = get_machine_spec_for_step(
        deepmd_iterative_path,
        training_path,
        "training",
        fake_machine,
        user_machine_keyword,
    )
    arcann_logger.debug(f"machine: {machine}")
    arcann_logger.debug(f"machine_walltime_format: {machine_walltime_format}")
    arcann_logger.debug(f"machine_job_scheduler: {machine_job_scheduler}")
    arcann_logger.debug(f"machine_launch_command: {machine_launch_command}")
    arcann_logger.debug(f"machine_max_jobs: {machine_max_jobs}")
    arcann_logger.debug(f"machine_max_array_size: {machine_max_array_size}")
    arcann_logger.debug(f"user_machine_keyword: {user_machine_keyword}")
    arcann_logger.debug(f"machine_spec: {machine_spec}")

    current_input_json["user_machine_keyword_train"] = user_machine_keyword
    arcann_logger.debug(f"current_input_json: {current_input_json}")

    if fake_machine is not None:
        arcann_logger.info(f"Pretending to be on: '{fake_machine}'.")
    else:
        arcann_logger.info(f"Machine identified: '{machine}'.")
    del fake_machine

    # Check if we can continue
    if curr_iter > 0:
        labeling_json = load_json_file(
            (control_path / f"labeling_{padded_curr_iter}.json")
        )
        if not labeling_json["is_extracted"]:
            arcann_logger.error("Lock found. Please execute 'labeling extract' first.")
            arcann_logger.error("Aborting...")
            return 1
        # exploration_json = load_json_file((control_path / f"exploration_{padded_curr_iter}.json"))
    else:
        # exploration_json = {}
        labeling_json = {}

    if "deepmd_model_version" not in user_input_json and nnp_program == "deepmd":
        dptrain_list = []
        for file in (current_path.parent / "user_files").iterdir():
            if file.suffix != ".json":
                continue
            if "dptrain" not in file.stem:
                continue
            dptrain_list.append(file)
        arcann_logger.debug(f"dptrain_list: {dptrain_list}")
        del file

        if not dptrain_list:
            arcann_logger.error(
                f"No dptrain_DEEPMDVERSION.json files found in {(current_path.parent / 'user_files')}"
            )
            arcann_logger.error("Aborting...")
            return 1

        dptrain_max_version = 0
        for dptrain in dptrain_list:
            dptrain_max_version = max(
                dptrain_max_version, float(dptrain.stem.split("_")[-1])
            )
        del dptrain

        arcann_logger.debug(f"dptrain_max_version: {dptrain_max_version}")
        current_input_json["deepmd_model_version"] = dptrain_max_version
        del dptrain_list, dptrain_max_version

        arcann_logger.info(
            f"Using DeePMD version: {current_input_json['deepmd_model_version']}"
        )

    elif "mace_model_version" not in user_input_json and nnp_program == "mace":
        macetrain_list = list(
            (current_path.parent / "user_files").glob("*.yml")
        ) + list((current_path.parent / "user_files").glob("*.yaml"))
        arcann_logger.debug(f"macetrain_list: {macetrain_list}")

        if not macetrain_list:
            arcann_logger.error(
                f"No macetrain_MACEVERSION.yaml files found in {(current_path.parent / 'user_files')}"
            )
            arcann_logger.error("Aborting...")
            return 1

        mace_max_version = "0"
        for mace in macetrain_list:
            mace_max_version = max(mace_max_version, mace.stem.split("_")[-1])

        arcann_logger.debug(f"mace_max_version: {mace_max_version}")
        current_input_json["mace_model_version"] = mace_max_version

        arcann_logger.info(
            f"Using MACE version: {current_input_json['mace_model_version']}"
        )

    # Generate/update both the training JSON and the merged input JSON
    # Priority: user/current > previous > default
    training_json, current_input_json = generate_training_json(
        current_input_json, previous_training_json, default_input_json
    )

    arcann_logger.debug(f"training_json: {training_json}")
    arcann_logger.debug(f"current_input_json: {current_input_json}")

    # Check if the job file exists
    job_file_name = f"job_{nnp_program}_train_{machine_spec['arch_type']}_{machine}.sh"
    if (current_path.parent / "user_files" / job_file_name).is_file():
        master_job_file = textfile_to_string_list(
            current_path.parent / "user_files" / job_file_name
        )
    else:
        arcann_logger.error(
            f"No JOB file provided for '{current_step.capitalize()} / {current_phase.capitalize()}' for this machine."
        )
        arcann_logger.error("Aborting...")
        return 1

    arcann_logger.debug(
        f"master_job_file: {master_job_file[0:5]}, {master_job_file[-5:-1]}"
    )
    current_input_json["job_email"] = get_key_in_dict(
        "job_email", user_input_json, previous_training_json, default_input_json
    )
    del job_file_name

    if nnp_program == "deepmd":
        # Check DeePMD version
        validate_deepmd_config(training_json)

        # Check if the default input json file exists
        dp_train_input_path = (
            training_path
            / "user_files"
            / f"dptrain_{training_json['deepmd_model_version']}.json"
        ).resolve()

        nnp_input = load_json_file(dp_train_input_path)
        if "type_map" not in main_json:
            type_map = [
                main_json["properties"][element]["symbol"]
                for element in main_json["properties"]
            ]
            main_json["type_map"] = type_map

        # Make sure they are the same
        if nnp_input["model"]["type_map"] != main_json["type_map"]:
            arcann_logger.error(
                f"Type map in {dp_train_input_path} does not match the one in config.json."
            )
            arcann_logger.error("Aborting...")
            return 1

        del dp_train_input_path
        arcann_logger.debug(f"dp_train_input: {nnp_input}")

    elif nnp_program == "mace":
        validate_mace_config(training_json)

        mace_input_path = (
            training_path
            / "user_files"
            / f"mace_{training_json['mace_model_version']}.yml"
        ).resolve()

        if not mace_input_path.exists():
            mace_input_path = (
                training_path
                / "user_files"
                / f"mace_{training_json['mace_model_version']}.yaml"
            ).resolve()

        nnp_input = load_yaml_file(mace_input_path)

        # control the key to align with MACE references
        nnp_input["energy_key"] = "REF_energy"
        nnp_input["forces_key"] = "REF_forces"
        nnp_input["virials_key"] = "REF_virials"

        if "E0s" not in nnp_input or not isinstance(nnp_input["E0s"], dict):
            arcann_logger.critical(
                "It is HIGHLY recommanded when training MACE model, to do it by providing the energies of the isolated atoms."
                "To do so, please provide these E0s by computing the energies of the isolated atoms with the same level of theory as the one used for the training dataset."
            )
        else:
            e0_atoms = set(nnp_input["E0s"].keys())
            system_atoms = [
                main_json["properties"][element]["symbol"]
                for element in main_json["properties"]
            ]
            elements = load_json_file(
                deepmd_iterative_path / "assets" / "elements.json"
            )
            system_atoms = [
                elm["atomic_number"]
                for elm in elements.values()
                if elm["symbol"] in system_atoms
            ]
            system_atoms = set(system_atoms)
            if e0_atoms != system_atoms:
                arcann_logger.error(
                    f"The atoms provided for E0s don't match with the one on your systems. E0 atoms: {e0_atoms}, system atoms: {system_atoms}."
                )
                arcann_logger.error("Aborting...")
                return 1
        arcann_logger.debug(f"mace_input: {nnp_input}")

        if "foundation_model" in nnp_input:
            fondation_path = (
                training_path / "user_files" / f"{nnp_input['foundation_model']}"
            ).resolve()
            if not fondation_path.is_file():
                arcann_logger.error(
                    f"Foundation model file {nnp_input['foundation_model']} not found in user_files."
                )
                arcann_logger.error("Aborting...")
                return 1

    arcann_logger.debug(f"main_json: {main_json}")

    # Load the datasets that were previously processed
    dataset = Dataset(training_dir=training_path, config_file=main_json)
    dataset.read_dataset()  # all datasets from control/dataset.json

    # Initial dataensemble may not be used
    if not training_json["use_initial_datasets"]:
        # in case we don't want to train with the initial datasets
        dataset.remove_datasets(init_dataset=True)

    auto_iter_count, adhoc_iter_count, val_auto_iter_count, val_adhoc_iter_count = (
        dataset.load_dataset(
            extra_dataset=training_json["use_extra_datasets"],
            init_dataset=training_json["use_initial_datasets"],
        )
    )
    dataset.update_control_file()  # update and save the control/dataset.json file

    main_json["extra_datasets"] = [list(dataset.control_file["extra_datasets"].keys())]
    main_json["systems_adhoc"] = [list(dataset.control_file["adhoc_datasets"].keys())]

    # Total of points in the datasets
    initial_count = sum(
        de.size for de in dataset.training_dataset.values() if de.step == "initial"
    )
    auto_count = sum(
        de.size for de in dataset.training_dataset.values() if de.step == "system_auto"
    )
    adhoc_count = sum(
        de.size for de in dataset.training_dataset.values() if de.step == "system_adhoc"
    )
    extra_count = sum(
        de.size for de in dataset.training_dataset.values() if de.step == "extra"
    )
    trained_count = initial_count + auto_count + adhoc_count + extra_count
    arcann_logger.debug(
        f"trained_count: {trained_count} = {initial_count} + {auto_count} + {adhoc_count} + {extra_count}"
    )
    arcann_logger.debug(f"training_dataset: {dataset.training_dataset}")
    training_json |= {
        "training_datasets": dataset.training_paths,
        "training_count": {
            "total": trained_count,
            "initial_count": initial_count,
            "added_auto_count": auto_count,
            "added_adhoc_count": adhoc_count,
            "extra_count": extra_count,
            "added_auto_iter_count": auto_iter_count,
            "added_adhoc_iter_count": adhoc_iter_count,
        },
    }

    initial_count = sum(
        de.size for de in dataset.validation_dataset.values() if de.step == "initial"
    )
    auto_count = sum(
        de.size
        for de in dataset.validation_dataset.values()
        if de.step == "system_auto"
    )
    adhoc_count = sum(
        de.size
        for de in dataset.validation_dataset.values()
        if de.step == "system_adhoc"
    )
    extra_count = sum(
        de.size for de in dataset.validation_dataset.values() if de.step == "extra"
    )
    validation_count = initial_count + auto_count + adhoc_count + extra_count
    arcann_logger.debug(
        f"validation_count: {validation_count} = {initial_count} + {auto_count} + {adhoc_count} + {extra_count}"
    )
    arcann_logger.debug(f"validation_dataset: {dataset.validation_dataset}")
    training_json |= {
        "validation_datasets": dataset.validation_paths,
        "validation_count": {
            "total": validation_count,
            "initial_count": initial_count,
            "added_auto_count": auto_count,
            "added_adhoc_count": adhoc_count,
            "extra_count": extra_count,
            "added_auto_iter_count": val_auto_iter_count,
            "added_adhoc_iter_count": val_adhoc_iter_count,
        },
    }
    arcann_logger.debug(f"training_json: {training_json}")

    if nnp_program == "deepmd":
        # Update the inputs with the sets
        nnp_input["training"]["training_data"]["systems"] = [
            "data/" + ds for ds in dataset.training_paths
        ]
        nnp_input["training"]["validation_data"]["systems"] = [
            "data/" + ds for ds in dataset.validation_paths
        ]
    elif nnp_program == "mace":
        nnp_input["train_file"] = "data/training_dataset.extxyz"
        nnp_input["valid_file"] = "data/validation_dataset.extxyz"
        nnp_input["test_file"] = "data/test_dataset.extxyz"

    # Here calculate the parameters
    # decay_steps it auto-recalculated as funcion of trained_count only for DeepMD
    if nnp_program == "deepmd":
        arcann_logger.debug(
            f"training_json - decay_steps: {training_json['decay_steps']}"
        )
        arcann_logger.debug(
            f"current_input_json - decay_steps: {current_input_json['decay_steps']}"
        )
        if not training_json["decay_steps_fixed"]:
            decay_steps = calculate_decay_steps(
                training_json["training_count"]["total"], training_json["decay_steps"]
            )
            arcann_logger.debug("Recalculating decay_steps")
            # Update the training JSON and the merged input JSON
            training_json["decay_steps"] = decay_steps
            current_input_json["decay_steps"] = decay_steps
        else:
            decay_steps = training_json["decay_steps"]
        arcann_logger.debug(f"decay_steps: {decay_steps}")
        arcann_logger.debug(
            f"training_json - decay_steps: {training_json['decay_steps']}"
        )
        arcann_logger.debug(
            f"current_input_json - decay_steps: {current_input_json['decay_steps']}"
        )

        # numb_steps and decay_rate
        arcann_logger.debug(
            f"training_json - numb_steps / decay_rate: {training_json['numb_steps']} / {training_json['decay_rate']}"
        )
        arcann_logger.debug(
            f"current_input_json - numb_steps / decay_rate: {current_input_json['numb_steps']} / {current_input_json['decay_rate']}"
        )
        numb_steps = training_json["numb_steps"]
        decay_rate_new = calculate_decay_rate(
            numb_steps,
            training_json["start_lr"],
            training_json["stop_lr"],
            training_json["decay_steps"],
        )
        while decay_rate_new < training_json["decay_rate"]:
            arcann_logger.debug(
                f"numb_steps is too small to allow for the decay_rate, increasing numb_steps: {decay_rate_new} < {training_json['decay_rate']}"
            )
            numb_steps = numb_steps + 10000
            decay_rate_new = calculate_decay_rate(
                numb_steps,
                training_json["start_lr"],
                training_json["stop_lr"],
                training_json["decay_steps"],
            )
        # Update the training JSON and the merged input JSON
        training_json["numb_steps"] = int(numb_steps)
        training_json["decay_rate"] = decay_rate_new
        current_input_json["numb_steps"] = int(numb_steps)
        current_input_json["decay_rate"] = decay_rate_new
        arcann_logger.debug(f"numb_steps: {numb_steps}")
        arcann_logger.debug(f"decay_rate: {decay_rate_new}")
        arcann_logger.debug(
            f"training_json - numb_steps / decay_rate: {training_json['numb_steps']} / {training_json['decay_rate']}"
        )
        arcann_logger.debug(
            f"current_input_json - numb_steps / decay_rate: {current_input_json['numb_steps']} / {current_input_json['decay_rate']}"
        )

        del decay_steps, numb_steps, decay_rate_new
        nnp_input["training"]["numb_steps"] = training_json["numb_steps"]
        nnp_input["learning_rate"]["decay_steps"] = training_json["decay_steps"]
        nnp_input["learning_rate"]["stop_lr"] = training_json["stop_lr"]

    elif nnp_program == "mace":
        nnp_input["max_num_epochs"] = training_json["max_num_epochs"]
        # TODO se pencher sur le lr, est-ce que c'est possible dans mace et comment?

    # Set booleans in the training JSON
    training_json = {
        **training_json,
        "is_prepared": True,
        "is_launched": False,
        "is_checked": False,
        "is_freeze_launched": False,
        "is_frozen": False,
        "is_compress_launched": False,
        "is_compressed": False,
        "is_incremented": False,
    }

    # Rsync data to local data
    # TODO not rsync, maybe just symlink + compress everything into one big dataset + possibly convert depending on the NNP

    localdata_path = current_path / "data"
    localdata_path.mkdir(exist_ok=True)
    if nnp_program == "deepmd":
        for train_dataset in dataset.training_paths:
            target_path = localdata_path / train_dataset
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(
                training_path / "data" / train_dataset,
                target_path,
                copy_function=os.link,
            )
        for valid_dataset in dataset.validation_paths:
            target_path = localdata_path / valid_dataset
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(
                training_path / "data" / valid_dataset,
                target_path,
                copy_function=os.link,
            )
        del train_dataset, valid_dataset, localdata_path

    elif nnp_program == "mace":
        dataset.prepare_for_mace_train(data_path=localdata_path)
        if nnp_input["foundation_model"]:
            foundation_model_path = (
                training_path / "user_files" / f"{nnp_input['foundation_model']}"
            ).resolve()
            shutil.copy(
                foundation_model_path, current_path / foundation_model_path.name
            )
            del foundation_model_path

    if nnp_program == "deepmd":
        # Change some inside output
        nnp_input["training"]["disp_file"] = "lcurve.out"
        nnp_input["training"]["save_ckpt"] = "model.ckpt"

    arcann_logger.debug(f"training_json: {training_json}")
    arcann_logger.debug(f"user_input_json: {user_input_json}")
    arcann_logger.debug(f"current_input_json: {current_input_json}")
    arcann_logger.debug(f"default_input_json: {default_input_json}")
    arcann_logger.debug(f"previous_training_json: {previous_training_json}")

    # Create the inputs/jobfiles for each NNP with random SEED

    # Walltime
    if (
        "job_walltime_train_h" in user_input_json
        and user_input_json["job_walltime_train_h"] > 0
    ):
        walltime_approx_s = int(user_input_json["job_walltime_train_h"] * 3600)
        mean_s_per_step = walltime_approx_s / training_json["numb_steps"]
        arcann_logger.debug(
            f"job_walltime_train_h: {user_input_json['job_walltime_train_h']}"
        )
    elif (
        "mean_s_per_step" in user_input_json and user_input_json["mean_s_per_step"] > 0
    ):
        walltime_approx_s = int(
            np.ceil((training_json["numb_steps"] * user_input_json["mean_s_per_step"]))
        )
        mean_s_per_step = walltime_approx_s / training_json["numb_steps"]
        arcann_logger.debug(f"mean_s_per_step: {user_input_json['mean_s_per_step']}")
    else:
        if curr_iter == 0:
            # This is rounded up to the next hour
            walltime_approx_s = int(
                np.ceil(
                    training_json["numb_steps"]
                    * default_input_json["mean_s_per_step"]
                    / 3600
                )
                * 3600
            )
            mean_s_per_step = walltime_approx_s / training_json["numb_steps"]
        else:
            walltime_approx_s = int(
                np.ceil(
                    training_json["numb_steps"]
                    * previous_training_json["mean_s_per_step"]
                    * 1.5
                    / 3600
                )
                * 3600
            )
            mean_s_per_step = walltime_approx_s / training_json["numb_steps"]

    current_input_json["job_walltime_train_h"] = float(walltime_approx_s / 3600)
    current_input_json["mean_s_per_step"] = mean_s_per_step
    training_json["job_walltime_train_h"] = float(walltime_approx_s / 3600)
    training_json["mean_s_per_step"] = mean_s_per_step
    arcann_logger.debug(f"walltime_approx_s: {walltime_approx_s}")
    arcann_logger.debug(f"mean_s_per_step: {mean_s_per_step}")

    for nnp in range(1, main_json["nnp_count"] + 1):
        local_path = current_path / f"{nnp}"
        local_path.mkdir(exist_ok=True)
        check_directory(local_path)

        random.seed()
        random_0_1000 = random.randrange(0, 1000)  # noqa: S311
        if nnp_program == "deepmd":
            replace_values_by_key_name(
                nnp_input, "seed", int(f"{nnp}{random_0_1000}{padded_curr_iter}")
            )

            dp_train_input_file = (Path(f"{nnp}") / "training.json").resolve()

            write_json_file(
                nnp_input,
                dp_train_input_file,
                enable_logging=False,
                read_only=True,
            )
        elif nnp_program == "mace":
            nnp_input["seed"] = int(f"{nnp}{random_0_1000}{padded_curr_iter}")
            nnp_input["name"] = f"model_{nnp}_{padded_curr_iter}"
            nnp_input["model_dir"] = "MACE_models"
            mace_input_file = (Path(f"{nnp}") / "training.yaml").resolve()
            write_yaml_file(
                nnp_input, mace_input_file, enable_logging=False, read_only=True
            )

        job_file = replace_in_slurm_file_general(
            master_job_file,
            machine_spec,
            walltime_approx_s,
            machine_walltime_format,
            training_json["job_email"],
        )

        if nnp_program == "deepmd":
            # Replace the inputs/variables in the job file
            job_file = replace_substring_in_string_list(
                job_file,
                "_R_DEEPMD_VERSION_",
                f"{training_json['deepmd_model_version']}",
            )
            job_file = replace_substring_in_string_list(
                job_file, "_R_DEEPMD_INPUT_FILE_", "training.json"
            )
            job_file = replace_substring_in_string_list(
                job_file, "_R_DEEPMD_LOG_FILE_", "training.log"
            )
            job_file = replace_substring_in_string_list(
                job_file, "_R_DEEPMD_OUTPUT_FILE_", "training.out"
            )

        elif nnp_program == "mace":
            # Replace the inputs/variables in the job file
            job_file = replace_substring_in_string_list(
                job_file, "_R_MACE_VERSION_", f"{training_json['mace_model_version']}"
            )  # maybe not a version but a repo
            job_file = replace_substring_in_string_list(
                job_file, "_R_MACE_INPUT_FILE_", "training.yaml"
            )
            job_file = replace_substring_in_string_list(
                job_file, "_R_MACE_LOG_FILE_", "training.log"
            )
            job_file = replace_substring_in_string_list(
                job_file, "_R_MACE_OUTPUT_FILE_", "training.out"
            )
            if nnp_input["foundation_model"]:
                job_file = replace_substring_in_string_list(
                    job_file,
                    "_R_MACE_FONDATION_FILE_",
                    f"../{nnp_input['foundation_model']}",
                )

        string_list_to_textfile(
            local_path
            / f"job_{nnp_program}_train_{machine_spec['arch_type']}_{machine}.sh",
            job_file,
            read_only=True,
        )
        del job_file, local_path, random_0_1000

    del nnp, walltime_approx_s, mean_s_per_step

    # Dump the JSON files (main, training and current input)
    arcann_logger.info("-" * 88)
    arcann_logger.debug(f"main_json: {main_json}")
    write_json_file(main_json, (control_path / "config.json"), read_only=True)
    write_json_file(
        training_json,
        (control_path / f"training_{padded_curr_iter}.json"),
        read_only=True,
    )
    backup_and_overwrite_json_file(
        current_input_json, (current_path / "used_input.json"), read_only=True
    )

    # End
    arcann_logger.info("-" * 88)
    arcann_logger.info(
        f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
    )

    # Cleaning
    del current_path, control_path, training_path
    del (
        default_input_json,
        default_input_json_present,
        user_input_json,
        user_input_json_present,
        user_input_json_filename,
    )
    del (
        main_json,
        current_input_json,
        training_json,
        previous_training_json,
        labeling_json,
    )
    del user_machine_keyword
    del curr_iter, padded_curr_iter
    del (
        machine,
        machine_spec,
        machine_walltime_format,
        machine_job_scheduler,
        machine_launch_command,
        machine_max_jobs,
        machine_max_array_size,
    )
    del master_job_file

    arcann_logger.debug("LOCAL")
    arcann_logger.debug(f"{locals()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "prepare",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass

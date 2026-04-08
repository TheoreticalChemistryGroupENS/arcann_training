"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2025 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2025/12/16
"""

# Standard library modules
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

# Local imports
from arcann_training.common.dataset import Dataset, ExtXYZEnsemble, Set000Ensemble
from arcann_training.common.filesystem import check_directory
from arcann_training.common.json import (
    backup_and_overwrite_json_file,
    load_default_json_file,
    load_json_file,
    write_json_file,
)
from arcann_training.common.utils import natural_sort_key
from arcann_training.initialization.utils import (
    check_dptrain_properties,
    check_lmp_properties,
    check_properties_file,
    generate_main_json,
)


# Main function
def main(
    current_step: str,
    current_phase: str,
    deepmd_iterative_path,
    fake_machine=None,
    user_input_json_filename: str = "input.json",
):
    # Get the logger
    arcann_logger = logging.getLogger("ArcaNN")

    # Get the current path and set the training path as the current path
    current_path = Path().resolve()
    training_path = current_path
    user_files_path = current_path / "user_files"

    # Log the step and phase of the program
    arcann_logger.info(f"Step: {current_step.capitalize()}.")
    arcann_logger.debug(f"Current path: {current_path}")
    arcann_logger.debug(f"Training path: {training_path}")
    arcann_logger.debug(f"Program path: {deepmd_iterative_path}")
    arcann_logger.info("-" * 88)

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
    user_input_json = load_json_file(
        (current_path / user_input_json_filename), abort_on_error=False
    )
    user_input_json_present = bool(user_input_json)
    arcann_logger.debug(f"user_input_json: {user_input_json}")
    arcann_logger.debug(f"user_input_json_present: {user_input_json_present}")

    # Check if a data folder is present in the training path
    check_directory(
        (training_path / "data"), error_msg=f"No data folder found in: {training_path}"
    )

    # Check the properties file
    properties_dict = check_properties_file(user_files_path / "properties.txt")
    arcann_logger.debug(f"properties_dict: {properties_dict}")

    # Auto-populate the systems_auto
    if "systems_auto" not in user_input_json:
        list_of_lmp = [file.stem for file in user_files_path.glob("*.lmp")]
        if not list_of_lmp:
            arcann_logger.error(f"No lmp found in {user_files_path}")
            arcann_logger.error("Aborting...")
            return 1
        list_of_lmp.sort(key=natural_sort_key)
        user_input_json["systems_auto"] = list_of_lmp
        arcann_logger.info(
            f"Auto-populated 'systems_auto' with: {user_input_json['systems_auto']}"
        )
    elif "systems_auto" in user_input_json and user_input_json["systems_auto"]:
        if not isinstance(user_input_json["systems_auto"], list):
            arcann_logger.error("'systems_auto' in the input JSON is not a list.")
            arcann_logger.error("Aborting...")
            return 1
        for system_auto in user_input_json["systems_auto"]:
            if not (user_files_path / f"{system_auto}.lmp").is_file():
                arcann_logger.error(
                    f"File not found: {user_files_path / f'{system_auto}.lmp'} but requested as system"
                )
                arcann_logger.error("Aborting...")
                return 1
        arcann_logger.info(
            f"Using 'systems_auto' from the input JSON: {user_input_json['systems_auto']}"
        )
    else:
        arcann_logger.error("Empty 'systems_auto' in the input JSON.")
        arcann_logger.error("Aborting...")
        return 1
    arcann_logger.debug(f"user_input_json: {user_input_json}")

    # Generate the main JSON, the merged input JSON and the padded current iteration
    main_json, merged_input_json, padded_curr_iter = generate_main_json(
        user_input_json, default_input_json
    )
    arcann_logger.debug(f"main_json: {main_json}")
    arcann_logger.debug(f"merged_input_json : {merged_input_json}")
    arcann_logger.debug(f"padded_curr_iter : {padded_curr_iter}")

    nnp_program = main_json["nnp_program"]

    arcann_logger.info(f"Using {nnp_program} as NNP software")
    arcann_logger.info("-" * 88)

    # Add the properties dictionary to the main JSON
    main_json["properties"] = properties_dict

    # Check the lmp against the properties
    for system_auto in main_json["systems_auto"]:
        check_lmp_properties(
            user_files_path / f"{system_auto}.lmp", main_json["properties"]
        )

    if nnp_program == "deepmd":
        # Check the dptrain against the properties
        check_dptrain_properties(user_files_path, main_json["properties"])
    elif nnp_program == "mace":
        if (
            len(
                list(user_files_path.glob("mace_*.yml"))
                + list(user_files_path.glob("mace_*.yaml"))
            )
            == 0
        ):
            arcann_logger.error(
                "MACE config file (mace_MACEVERSION.yml) is missing from user_files."
            )
            arcann_logger.error("Aborting...")
            raise FileNotFoundError("MACE config file is missing from user_files.")
    else:
        arcann_logger.error(
            f"NNP program: {nnp_program} not recognized. ArcaNN supports 'deepmd' or 'mace'."
        )
        arcann_logger.error("Aborting...")
        raise ValueError(
            f"NNP program: {nnp_program} not recognized. ArcaNN supports 'deepmd' or 'mace'."
        )

    # Create the control directory
    control_path = training_path / "control"
    arcann_logger.debug(f"control_path: {control_path}")

    # DEBUG: Print the JSON files
    arcann_logger.debug(f"main_json: {main_json}")
    arcann_logger.debug(f"user_input_json: {user_input_json}")
    arcann_logger.debug(f"merged_input_json: {merged_input_json}")

    try:
        dataset = Dataset(
            training_dir=training_path,
            config_file=main_json,
        )
    except Exception as e:
        arcann_logger.exception(f"Error in initializing the Datasets: {e}")
        arcann_logger.error("Aborting...")
        return 1

    # !!!! these will be horrible code but it's just for the sake of transitionning between one kind of dataset management to another !!!!
    is_extxyz, is_set000 = False, False
    for ens in (training_path / "data").iterdir():
        if ens.is_dir():
            if any(ens.glob("*.extxyz")):
                is_extxyz = True
            elif (ens / "set.000").is_dir():
                is_set000 = True

    if is_extxyz and is_set000:
        arcann_logger.error(
            "Both extxyz files and set.000 folders found in the data directory. Please keep only one format."
        )
        arcann_logger.error("Aborting...")
        return 1

    elif is_extxyz and main_json["data_format"] != "extxyz":
        dataset.data_format = "extxyz"
        dataset.data_ensemble = ExtXYZEnsemble
    elif is_set000 and main_json["data_format"] != "set.000":
        dataset.data_format = "set.000"
        dataset.data_ensemble = Set000Ensemble
    # Populate the dataset with existing data
    dataset.load_dataset(
        extra_dataset=True,
        init_dataset=True,
    )
    arcann_logger.debug(
        f"Loaded training dataset: {dataset.training_dataset}, {dataset.validation_dataset}"
    )
    if not dataset.validation_dataset:
        # there is no validation set, we'll create it
        subprocess.run(
            ["cp", "-a", str(training_path / "data"), str(training_path / "old_data")],
            check=True,
        )
        (training_path / "converted_data").mkdir(exist_ok=True)

        trainings, validations = {}, {}
        for ensemble in dataset.training_dataset.values():
            ensemble.load()
            arcann_logger.debug(f"Loading ensemble: {ensemble.path}")

            # create the data ensemble in the new directory, one for training and one for validation
            common_kwargs = {
                "step": ensemble.step,
                "system_name": ensemble.system_name
                if ensemble.system_name is not None
                else None,
                "iteration": int(ensemble.iteration)
                if ensemble.iteration is not None
                else None,
                "data_format": dataset.data_format,
                "properties": dataset.config_file["properties"],
            }
            if ensemble.system_name:
                training_dir = training_path / "converted_data" / ensemble.path.name
                validation_dir = (
                    training_path
                    / "converted_data"
                    / f"{ensemble.system_name}_valid_{str(ensemble.iteration).zfill(3)}"
                )
            else:
                training_dir = training_path / "converted_data" / ensemble.path.name
                names = ensemble.path.name.split("_")
                validation_dir = (
                    training_path
                    / "converted_data"
                    / f"{names[0]}_valid_{'_'.join(names[1:])}"
                )
            training_dir.mkdir(exist_ok=True)
            validation_dir.mkdir(exist_ok=True)
            training_data_ensemble = dataset.data_ensemble(
                path=training_dir, training_type="training", **common_kwargs
            )
            validation_data_ensemble = dataset.data_ensemble(
                path=validation_dir, training_type="validation", **common_kwargs
            )

            # populate these data ensembles with the data from the original
            nb_of_frames = ensemble.coord.shape[0]
            indices = np.random.permutation(nb_of_frames)
            split_idx = int(nb_of_frames * (1 - dataset.split))
            train_idx = indices[:split_idx]
            val_idx = indices[split_idx:]
            arrays = {
                "energy": ensemble.energy,
                "coord": ensemble.coord,
                "box": ensemble.box,
                "force": ensemble.force,
                "virial": ensemble.virial,
                "wannier": ensemble.wannier,
            }
            train_arrays = {
                name: arr[train_idx] if arr is not None else None
                for name, arr in arrays.items()
            }
            training_data_ensemble.load_from_raw_arrays(
                type=ensemble.type,
                **train_arrays,
                wannier_not_cvg=ensemble.wannier_not_cvg,
                is_periodic=ensemble.is_periodic,
            )
            trainings[training_data_ensemble.path.name] = training_data_ensemble
            val_arrays = {
                name: arr[val_idx] if arr is not None else None
                for name, arr in arrays.items()
            }
            validation_data_ensemble.load_from_raw_arrays(
                type=ensemble.type,
                **val_arrays,
                wannier_not_cvg=ensemble.wannier_not_cvg,
                is_periodic=ensemble.is_periodic,
            )
            validations[validation_data_ensemble.path.name] = validation_data_ensemble

            # do the actual convertion:
            if main_json["data_format"] == "extxyz":
                training_data_ensemble.to_extxyz(save=True)
                validation_data_ensemble.to_extxyz(save=True)
            else:
                training_data_ensemble.to_set000()
                validation_data_ensemble.to_set000()

        dataset.training_dataset |= trainings
        dataset.validation_dataset |= validations
        subprocess.run(["rm", "-rf", str(training_path / "data")], check=True)
        subprocess.run(
            ["mv", str(training_path / "converted_data"), str(training_path / "data")],
            check=True,
        )

    else:
        dataset.convert_dataset(
            to_extxyz=(main_json["data_format"] == "extxyz"),
            to_set000=(main_json["data_format"] == "set000"),
        )
    dataset.update_control_file()

    arcann_logger.debug(
        f"initial_dataset_paths: {dataset.training_paths + dataset.validation_paths}"
    )
    arcann_logger.debug(f"dataset_json: {dataset.control_file['initial_datasets']}")

    # Dump the JSON files (main, initial datasets and merged input)
    arcann_logger.info("-" * 88)
    write_json_file(main_json, (control_path / "config.json"), read_only=True)
    backup_and_overwrite_json_file(
        merged_input_json, (current_path / "used_input.json"), read_only=True
    )

    # End
    arcann_logger.info("-" * 88)
    arcann_logger.info(f"Step: {current_step.capitalize()} is a success!")

    # Cleaning
    del current_path, control_path, training_path
    del (
        default_input_json,
        default_input_json_present,
        user_input_json,
        user_input_json_present,
        user_input_json_filename,
    )
    del padded_curr_iter
    del main_json, merged_input_json
    del dataset

    arcann_logger.debug("LOCAL")
    arcann_logger.debug(f"{locals()}")
    return 0


# Standalone part
if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "initialization",
            "transition",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass

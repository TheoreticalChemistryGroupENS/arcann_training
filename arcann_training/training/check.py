"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/07/14
"""

# Standard library modules
import logging
import re
import sys
from pathlib import Path

# Non-standard imports
import numpy as np
from packaging import version

# Local imports
from arcann_training.common.check import validate_step_folder
from arcann_training.common.json import (
    find_key_in_dict,
    load_json_file,
    write_json_file,
)
from arcann_training.common.list import textfile_to_string_list
from arcann_training.common.yaml import load_yaml_file


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

    # Get control path, load the main JSON and the training JSON
    control_path = training_path / "control"
    main_json = load_json_file((control_path / "config.json"))
    training_json = load_json_file((control_path / f"training_{padded_curr_iter}.json"))
    nnp_program: str = main_json["nnp_program"]

    arcann_logger.info(f"Using {nnp_program} as NNP software")

    # Check if we can continue
    if not training_json["is_launched"]:
        arcann_logger.error("Lock found. Please execute 'training launch' first.")
        arcann_logger.error("Aborting...")
        return 1

    # Check the normal termination of the training phase
    # Counters
    # s_per_step_per_step_size = []
    training_times = []
    step_sizes = []
    completed_count = 0
    min_nbor_dist = None
    max_nbor_size = None
    if nnp_program == "deepmd":
        nnp_version = version.parse(training_json["deepmd_model_version"])
        training_input = (
            load_json_file(current_path / "1" / "training.json")
            if (current_path / "1" / "training.json").is_file()
            else None
        )

    elif nnp_program == "mace":
        nnp_version = version.parse(training_json["mace_model_version"])
        training_input = (
            load_yaml_file(current_path / "1" / "training.yaml")
            if (current_path / "1" / "training.yaml").is_file()
            else None
        )

    for nnp in range(1, main_json["nnp_count"] + 1):
        local_path = current_path / f"{nnp}"
        if (local_path / "training.log").is_file():
            training_out = textfile_to_string_list((local_path / "training.log"))
        elif (local_path / "training.out").is_file():
            training_out = textfile_to_string_list((local_path / "training.out"))
        else:
            training_out = []
        if training_out:
            # Finished correctly
            if nnp_program == "deepmd" and any(
                "finished training" in s for s in training_out
            ):
                if nnp_program == "deepmd" and nnp_version >= version.parse("3.0.0"):
                    training_out_time = [s for s in training_out if "wall time" in s]
                    batch_pattern = r"batch\s*(\d+)\b"
                    time_pattern = r"wall time = (\d+\.\d+) s"

                else:
                    training_out_time = [
                        s for s in training_out if "training time" in s
                    ]
                    batch_pattern = r"batch\s*(\d+)\s"
                    time_pattern = r"training time (\d+\.\d+) s"

                if min_nbor_dist is None or max_nbor_size is None:
                    for log_text in training_out:
                        if "min nbor dist" in log_text:
                            min_nbor_dist_match = re.search(
                                r"min nbor dist: ([\d\.]+)", log_text
                            )
                            if min_nbor_dist_match:
                                min_nbor_dist = float(min_nbor_dist_match.group(1))
                        elif "max nbor size" in log_text:
                            max_nbor_size_match = re.search(
                                r"max nbor size: \[([ \d]+)\]", log_text
                            )
                            if max_nbor_size_match:
                                max_nbor_size = [
                                    int(n) for n in max_nbor_size_match.group(1).split()
                                ]

                batch_numbers = []

                for entry in training_out_time:
                    batch_match = re.search(batch_pattern, entry)
                    time_match = re.search(time_pattern, entry)

                    if batch_match and time_match:
                        batch_number = int(batch_match.group(1))
                        training_time = float(time_match.group(1))

                        batch_numbers.append(batch_number)
                        training_times.append(training_time)

                del (
                    entry,
                    batch_match,
                    time_match,
                    batch_number,
                    training_time,
                    training_out_time,
                )
                del time_pattern, batch_pattern

                for suffix in ["index", "meta", "data-00000-of-00001"]:
                    if (
                        local_path / f"model.ckpt-{batch_numbers[-1]}.{suffix}"
                    ).is_file():
                        (
                            local_path / f"model.ckpt-{batch_numbers[-1]}.{suffix}"
                        ).rename(local_path / f"model.ckpt.{suffix}")
                del suffix

                step_sizes.extend(np.diff(batch_numbers))
                del batch_numbers
                completed_count += 1

            elif nnp_program == "mace" and any(
                "Training complete" in s for s in training_out
            ):
                completed_count += 1
            else:
                arcann_logger.critical(f"DP Train - '{nnp}' not finished/failed.")
            del training_out
        else:
            arcann_logger.critical(f"DP Train - '{nnp}' still running/no outfile.")
        del local_path
    del nnp
    arcann_logger.debug(f"completed_count: {completed_count}")

    # Infos
    if min_nbor_dist is not None:
        training_json["min_nbor_dist"] = min_nbor_dist
        arcann_logger.info(f"Your minimum neighbor distance is: {min_nbor_dist:.3f}")
        if min_nbor_dist < 0.1:
            arcann_logger.warning(
                "Your minimum neighbor distance is lower than 0.1 Angstrom."
            )
            arcann_logger.warning("You might have a funky system.")

    if max_nbor_size is not None:
        training_json["max_nbor_size"] = max_nbor_size
        arcann_logger.info(
            f"In the training datasets, the maximum number of type-i neighbors of an atom is: {max_nbor_size}"
        )
        arcann_logger.info(f"Your type map was: {main_json['type_map']}")
        arcann_logger.info(f"The total is: {sum(max_nbor_size)}")
        selection_list = find_key_in_dict(training_input, "sel")
        arcann_logger.info(
            f"In the training parameters, the expected maximum number of type-i neighbors of an atom was: {selection_list[0]} (keyword 'sel')."
        )
        if sum(max_nbor_size) > sum(selection_list[0]):
            arcann_logger.warning(
                "The maximum number of type-i neighbors of an atom is higher than the expected maximum number of type-i neighbors of an atom (keyword 'sel')."
            )
            arcann_logger.warning("Please correct this.")
        if sum(selection_list[0]) > 2.0 * sum(max_nbor_size):
            arcann_logger.warning(
                "The expected maximum number of type-i neighbors of an atom is at least 100% larger that the ones present in the training datasets."
            )
            arcann_logger.warning(
                "You may want to decrease the expected maximum number of type-i neighbors of an atom (keyword 'sel')."
            )

    arcann_logger.info("-" * 88)
    # Update the boolean in the training JSON
    if completed_count == main_json["nnp_count"]:
        training_json["is_checked"] = True

    # If not empty
    if training_times and step_sizes:
        training_json["mean_s_per_step"] = np.average(training_times) / np.average(
            step_sizes
        )
        training_json["median_s_per_step"] = np.median(training_times) / np.average(
            step_sizes
        )
        training_json["stdeviation_s_per_step"] = np.std(training_times) / np.average(
            step_sizes
        )
    else:
        training_json["mean_s_per_step"] = -1
        training_json["median_s_per_step"] = -1
        training_json["stdeviation_s_per_step"] = -1

    arcann_logger.debug(f"mean_s_per_step: {training_json['mean_s_per_step']}")
    arcann_logger.debug(f"median_s_per_step: {training_json['median_s_per_step']}")
    arcann_logger.debug(
        f"stdeviation_s_per_step: {training_json['stdeviation_s_per_step']}"
    )

    del training_times, step_sizes
    # Dump the JSON files (training)
    write_json_file(
        training_json,
        (control_path / f"training_{padded_curr_iter}.json"),
        read_only=True,
    )

    # End
    arcann_logger.info("-" * 88)
    if completed_count == main_json["nnp_count"]:
        arcann_logger.info(
            f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
        )
    else:
        arcann_logger.critical(
            f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a failure!"
        )
        arcann_logger.critical("Some DP Train did not finished correctly.")
        arcann_logger.critical("Please check manually before re-exectuing this step.")
        arcann_logger.critical("Aborting...")
        return 1
    del completed_count

    # Cleaning
    del current_path, control_path, training_path
    del user_input_json_filename
    del main_json, training_json
    del curr_iter, padded_curr_iter

    arcann_logger.debug("LOCAL")
    arcann_logger.debug(f"{locals()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "check",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass

"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/05/15
"""

# Standard library modules
import logging
import sys
from pathlib import Path

# Local imports
from arcann_training.common.check import validate_step_folder
from arcann_training.common.json import load_json_file, write_json_file
from arcann_training.common.lammps import LAMMPSInputHandler, LAMMPSPair


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
    if not training_json["is_compress_launched"]:
        arcann_logger.error("Lock found. Please execute 'training compress' first.")
        arcann_logger.error("Aborting...")
        return 1

    completed_count = 0
    needed_mace_styles: set[LAMMPSPair] = set()  # TODO: this is bad here
    if nnp_program == "deepmd":
        for nnp in range(1, main_json["nnp_count"] + 1):
            local_path = current_path / f"{nnp}"
            if (local_path / f"graph_{nnp}_{padded_curr_iter}_compressed.pb").is_file():
                completed_count += 1
            else:
                arcann_logger.critical(f"DP Compress - '{nnp}' not finished/failed.")
            del local_path
        del nnp
    elif nnp_program == "mace":
        for lmp_input in (training_path / "user_files").glob("*.in"):
            needed_mace_styles.add(
                LAMMPSInputHandler(
                    lmp_input,
                    [
                        main_json["properties"][element]["symbol"]
                        for element in main_json["properties"]
                    ],
                ).lmp_pair
            )

        style_ext = {
            LAMMPSPair.SYMMETRIX: ".model.json",
            LAMMPSPair.MACE: "-lammps.pt",
            LAMMPSPair.MLIAP: "-mliap_lammps.pt",
        }

        for nnp in range(1, main_json["nnp_count"] + 1):
            local_path = current_path / f"{nnp}" / "MACE_models"
            for style in needed_mace_styles:
                if (
                    local_path / f"model_{nnp}_{padded_curr_iter}{style_ext[style]}"
                ).is_file():
                    completed_count += 1
                else:
                    arcann_logger.critical(
                        f"MACE Compress - '{nnp}-{str(style)}' not finished/failed."
                    )

    arcann_logger.debug(f"completed_count: {completed_count}")

    # Dump the JSON files (training)
    write_json_file(
        training_json,
        (control_path / f"training_{padded_curr_iter}.json"),
        read_only=True,
    )

    # End
    arcann_logger.info("-" * 88)
    if completed_count == main_json["nnp_count"] or (
        nnp_program == "mace"
        and completed_count == main_json["nnp_count"] * len(needed_mace_styles)
    ):
        arcann_logger.info(
            f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a success!"
        )
    else:
        arcann_logger.error(
            f"Step: {current_step.capitalize()} - Phase: {current_phase.capitalize()} is a failure!"
        )
        arcann_logger.error("Some DP Compress did not finished correctly.")
        arcann_logger.error("Please check manually before relaunching this step.")
        arcann_logger.error("Aborting...")
    del completed_count

    # Cleaning
    del current_path, control_path, training_path
    del user_input_json_filename
    del main_json, training_json
    del curr_iter, padded_curr_iter

    arcann_logger.debug("LOCAL")
    arcann_logger.debug("{locals()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(
            "training",
            "check_compress",
            Path(sys.argv[1]),
            fake_machine=sys.argv[2],
            user_input_json_filename=sys.argv[3],
        )
    else:
        pass

"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2025 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2025/12/16
Last modified: 2026/01/30
"""

import logging
from pathlib import Path
from typing import Dict

import yaml

from arcann_training.common.utils import catch_errors_decorator


@catch_errors_decorator
def load_yaml_file(
    file_path: Path, abort_on_error: bool = True, enable_logging: bool = True
) -> Dict:
    """
    Load a YAML file from the given file path and return its contents as a dictionary.

    Parameters
    ----------
    file_path: Path
        The path to the YAML file to be loaded.
    abort_on_error: bool
        Whether to abort the program if the file cannot be found. If True, an error message is logged and the program exits with an error code. If False, an empty dictionary is returned. Defaults is True.
    enable_logging: bool
        Whether to log information about the loading process. Defaults is True.

    Returns
    -------
    Dict
        A dictionary containing the contents of the YAML file.

    Raises
    ------
    TypeError
        If file_path is not a Path object.
    FileNotFoundError
        If the file cannot be found and abort_on_error is True.
    """
    if not isinstance(file_path, Path):
        error_msg = f"'{file_path}' must be a '{type(Path())}'."
        raise TypeError(error_msg)

    # Check if the file exists and is a file
    if file_path.is_file():
        # If logging is enabled, log information about the loading process
        if enable_logging:
            logging.info(f"Loading '{file_path.name}' from '{file_path.parent}'.")
        # Open the file and load the contents as a dictionary
        with file_path.open(encoding="UTF-8") as yaml_file:
            # Check if the file is empty
            file_content = yaml_file.read().strip()
            if len(file_content) == 0:
                return {}
            return yaml.safe_load(file_content)
    else:
        # If the file cannot be found and abort_on_error is True, log an error message and exit with an error code
        if abort_on_error:
            error_msg = f"File '{file_path.name}' not found in '{file_path.parent}'."
            raise FileNotFoundError(error_msg)
        # If abort_on_error is False, return an empty dictionary
        else:
            # If logging is enabled, log information about the creation of the empty dictionary
            if enable_logging:
                logging.info(
                    f"Creating an empty dictionary: '{file_path.name}' in '{file_path.parent}'."
                )
            return {}


@catch_errors_decorator
def write_yaml_file(
    yaml_dict: Dict,
    file_path: Path,
    enable_logging: bool = True,
    read_only: bool = False,
) -> None:
    """
    Writes a dictionary to a YAML file, optionally logging the action and setting the file to read-only.

    This function serializes `YAML_dict` to a YAML-formatted string (with pretty-printing) and writes it to the file
    specified by `file_path`. It can optionally log the write operation and modify the file's permissions to read-only.

    Parameters
    ----------
    yaml_dict : dict
        The dictionary to serialize and write to the YAML file.
    file_path : Path
        The file path where the YAML data should be written. This must be an instance of `Path`, otherwise, a
        `TypeError` will be raised.
    enable_logging : bool, optional
        If True (the default), logs a message indicating the file path where the YAML data is being written.
    read_only : bool, optional
        If True, sets the file's permissions to read-only after writing. If False (the default), the file's
        permissions are not modified.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If `file_path` is not an instance of `Path`.
    Exception
        If there is an issue writing to the file (e.g., permissions issue, disk full, file locked).
    """
    if not isinstance(file_path, Path):
        error_msg = f"'{file_path}' must be a '{type(Path())}'."
        raise TypeError(error_msg)

    if file_path.is_file():
        current_permissions = file_path.stat().st_mode
        new_permissions = current_permissions | 0o200
        file_path.chmod(new_permissions)

    try:
        # Open the file specified by the file_path argument in write mode
        with file_path.open("w", encoding="UTF-8") as yaml_file:
            # Convert dictionary to formatted YAML string
            yaml_str = yaml.dump(yaml_dict, indent=4)
            yaml_file.write(yaml_str)

        # If log_write is True, log a message indicating the file and path that the YAML data is being written to
        if enable_logging:
            logging.info(f"YAML data written to '{file_path.absolute()}'.")
        if read_only:
            current_permissions = file_path.stat().st_mode
            # Remove the write permission (0222) while keeping others intact
            new_permissions = current_permissions & ~0o222
            # Update the file permissions
            file_path.chmod(new_permissions)

    except (OSError, IOError) as e:
        # Raise an exception if the file path is not valid or the file cannot be written
        error_msg = f"Error writing YAML data to file '{file_path}': '{e}'."
        raise Exception(error_msg) from e

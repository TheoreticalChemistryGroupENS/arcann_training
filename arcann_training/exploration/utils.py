"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2022/01/01
Last modified: 2024/08/06

Functions
---------
generate_input_exploration_json(user_input_json: Dict, previous_json: Dict, default_input_json: Dict, merged_input_json: Dict, main_json: Dict) -> Dict
    Update and complete input JSON by incorporating values from the user input JSON, the previous JSON, and the default JSON.

get_system_exploration(merged_input_json: Dict, system_auto_index: int) -> Tuple[Union[float, int],Union[float, int],Union[float, int],Union[float, int],Union[float, int],Union[float, int],Union[float, int],Union[float, int],bool]
    Returns a tuple of system exploration parameters based on the input JSON and system number.

generate_input_exploration_deviation_json(user_input_json: Dict, previous_json: Dict, default_input_json: Dict, merged_input_json: Dict, main_json: Dict,) -> Dict
    Update and complete input JSON by incorporating values from the user input JSON, the previous JSON, and the default JSON.

get_system_deviation(merged_input_json: Dict, system_auto_index: int) -> Tuple[Union[float, int], Union[float, int], Union[float, int], Union[float, int], Union[float, int]]
    Return a tuple of system exploration parameters based on the input JSON and system index.

generate_input_exploration_disturbed_json(user_input_json: Dict, previous_json: Dict, default_input_json: Dict, merged_input_json: Dict, main_json: Dict,) -> Dict
    Update and complete input JSON by incorporating values from the user input JSON, the previous JSON, and the default JSON.

get_system_disturb(merged_input_json: Dict, system_auto_index: int) -> Tuple[Union[float, int], Union[float, int], List[int]]
    Return a tuple of system exploration parameters based on the input JSON and system index.

generate_starting_points(exploration_type: int, system_auto: int, training_path: str, padded_prev_iter: str, previous_json: Dict, input_present: bool, disturbed_start: bool) -> Tuple[List[str], List[str], bool]
    Generates a list of starting point file names.

create_models_list(main_json: Dict, previous_json: Dict, it_nnp: int, padded_prev_iter: str, training_path: Path, local_path: Path) -> Tuple[List[str], str]
    Generate a list of model file names and create symbolic links to the corresponding model files.

get_last_frame_number(model_deviation: np.ndarray, sigma_high_limit: float, disturbed_start: bool) -> int
    Returns the index of the last frame to be processed based on the given parameters.

update_system_nb_steps_factor(previous_json: Dict, system_auto_index: int) -> int
    Calculates a ratio based on information from a dictionary and returns a multiplying factor for system_nb_steps.
"""

# TODO: Homogenize the docstrings for this module

# Standard library modules
import logging
import re
import subprocess
from copy import deepcopy
from enum import StrEnum, auto
from pathlib import Path
from typing import Dict, List, Tuple, Union

# Third-party modules
import numpy as np

from arcann_training.common.json import convert_control_to_input

# Local imports
from arcann_training.common.utils import catch_errors_decorator


class LAMMPSPair(StrEnum):
    """
    Enumeration of supported LAMMPS pair styles.

    Attributes
    ----------
    MACE : auto
        MACE pair style.
    MLIAP : auto
        MLIAP pair style.
    SYMMETRIX : auto
        SYMMETRIX pair style.
    DEEPMD : auto
        DeepMD pair style.
    """

    MACE = auto()
    MLIAP = auto()
    SYMMETRIX = auto()
    DEEPMD = auto()

    @classmethod
    def from_string(cls, value: str) -> "LAMMPSPair":
        """
        Convert a string to a LAMMPSPair enum.

        Parameters
        ----------
        value : str
            The string representation of the LAMMPS pair style.

        Returns
        -------
        LAMMPSPair
            Corresponding enum member.

        Raises
        ------
        ValueError
            If the pair style is not supported.
        """
        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"ArcaNN does not support the pair_style: {value}. Please use mace, mliap, or symmetrix/mace (Kokkos versions are allowed)."
            ) from None


class LAMMPSInputHandler:
    """
    Handler for reading, parsing, and modifying LAMMPS input files.

    Provides methods to extract metadata, inject additional commands,
    and apply variable replacements in LAMMPS input text.

    Attributes
    ----------
    cell_info_lammps : list of str
        LAMMPS commands to output cell information.
    mace_dump_0 : str
        Default dump command for MACE simulations.
    """

    cell_info_lammps = [
        "variable v_xlo equal xlo",
        "variable v_xhi equal xhi",
        "variable v_ylo equal ylo",
        "variable v_yhi equal yhi",
        "variable v_zlo equal zlo",
        "variable v_zhi equal zhi",
        'fix extra all print _R_PRINT_FREQ_ "${v_xlo} ${v_xhi} ${v_ylo} ${v_yhi} ${v_zlo} ${v_zhi}" file cell.txt',
        "",
    ]

    mace_dump_0 = (
        "dump traj all custom _R_PRINT_FREQ_ _R_LMPTRJ_OUT_ id type x y z fx fy fz\n"
    )

    def __init__(self, lmp_input: str | Path):
        """
        Initialize a LAMMPS input handler.

        Parameters
        ----------
        lmp_input : str or Path
            Path to the LAMMPS input file.

        Raises
        ------
        FileNotFoundError
            If the LAMMPS input file does not exist.
        """
        if lmp_input.exists():
            self._lmp_input = Path(lmp_input)
        else:
            raise FileNotFoundError(f"LAMMPS input not found: {lmp_input}")

        self._raw_text = self._read_input_file()
        self._parse_metadata()

        self._load_prepare_input()
        self._prepare_input_text()

    @catch_errors_decorator
    def _read_input_file(self) -> str:
        """
        Read the LAMMPS input file.

        Returns
        -------
        str
            The content of the LAMMPS input file.
        """
        return self._lmp_input.read_text()

    @catch_errors_decorator
    def _parse_metadata(self) -> None:
        """
        Extract metadata from the LAMMPS input.

        Metadata includes:
        - LAMMPS pair style
        - Domain decomposition (for MACE)
        - PLUMED usage

        Raises
        ------
        ValueError
            If no pair_style is found in the input.
        """
        # Extract pair_style
        pair_style_match = re.search(
            r"^\s*(?!#)pair_style\s+([^\s/]+)", self._raw_text, re.MULTILINE
        )
        if not pair_style_match:
            raise ValueError(f"No pair_style found in {self._lmp_input}")

        self._lmp_pair = LAMMPSPair.from_string(pair_style_match.group(1))

        # Determine domain decomposition for MACE
        self.domain_decomp = None
        if self._lmp_pair == LAMMPSPair.MACE:
            self.domain_decomp = "no_domain_decomposition" in self._raw_text

        self._plumed = (
            True
            if re.search(r"^\s*(?!#)plumed", self._raw_text, re.MULTILINE)
            else False
        )

    @catch_errors_decorator
    def _prepare_input_text(self) -> None:
        """
        Inject additional commands into the LAMMPS input.

        Typically includes cell info variables and MACE dump commands.

        Raises
        ------
        ValueError
            If no 'run _R_NUMBER_OF_STEPS_' command is found.
        """
        match = re.search(
            r"^\s*(?!#)run\s+_R_NUMBER_OF_STEPS_", self._raw_text, re.MULTILINE
        )
        if not match:
            raise ValueError(
                f"No 'run _R_NUMBER_OF_STEPS_' found in the LAMMPS input file: {self._lmp_input}"
            )

        run_index = match.start()
        self._raw_text = (
            self._raw_text[:run_index]
            + "\n".join(self.cell_info_lammps)
            + self.mace_dump_0
            + self._raw_text[run_index:]
        )

    @property
    def lines(self, keepends=False) -> list[str]:
        """
        Split the LAMMPS input into lines.

        Parameters
        ----------
        keepends : bool, optional
            If True, line endings are preserved, by default False.

        Returns
        -------
        list of str
            Lines of the LAMMPS input.
        """
        return self._raw_text.splitlines(keepends=keepends)

    @catch_errors_decorator
    def apply_variables(
        self, variables: dict[str, str | int | float], splitlines=False
    ) -> str | list[str]:
        """
        Replace placeholders in the LAMMPS input with provided variables.

        Parameters
        ----------
        variables : dict
            Mapping of placeholders (keys) to replacement values.
        splitlines : bool, optional
            If True, returns a list of lines instead of a single string, by default False.

        Returns
        -------
        str or list of str
            Formatted input text as a string or list of lines.
        """
        formatted_text = str(self._raw_text)  # copy text

        for key, value in variables.items():
            formatted_text.replace(key, value)

        return formatted_text if not splitlines else formatted_text.splitlines()

    @property
    def lmp_pair(self) -> LAMMPSPair:
        """
        Get the detected LAMMPS pair style.

        Returns
        -------
        LAMMPSPair
            The pair style detected in the input.
        """
        return self._lmp_pair

    @property
    def raw_text(self) -> str:
        """
        Get the preprocessed raw LAMMPS input text.

        Returns
        -------
        str
            The LAMMPS input text.
        """
        return self._raw_text

    def has_plumed(self) -> bool:
        """
        Check if the LAMMPS input includes PLUMED commands.

        Returns
        -------
        bool
            True if PLUMED is detected, False otherwise.
        """
        return self._plumed

    @catch_errors_decorator
    def _parse(self) -> None:
        raw_text = self.lmp_input.read_text()
        pair_style = re.search(
            r"^\s*(?!#)pair_style\s+([^\s/]+)", raw_text
        )  # get the pair_style being used ignoring commented lines, and anything that modifies the pair like /kk

        if not pair_style:
            raise ValueError(f"No pair_style found in {self.lmp_input}")

        self.lmp_pair = LAMMPSPair.from_string(pair_style.group(1))

        if self.lmp_pair == LAMMPSPair.MACE:
            if "no_domain_decomposition" in raw_text:
                self.domain_decomp = True
            else:
                self.domain_decomp = False
        else:
            self.domain_decomp = None


# TODO: Add tests for this function
@catch_errors_decorator
def generate_input_exploration_json(
    user_input_json: Dict,
    previous_json: Dict,
    default_input_json: Dict,
    merged_input_json: Dict,
    main_json: Dict,
) -> Dict:
    """
    Update and complete input JSON by incorporating values from the user input JSON, the previous JSON, and the default JSON.

    Parameters
    ----------
    user_input_json : dict
        The input JSON provided by the user, containing user-defined parameters.
    previous_json : dict
        The JSON from the previous iteration.
    default_input_json : dict
        The default input JSON containing default parameters.
    merged_input_json : dict
        The merged input JSON.
    main_json : dict
        The main JSON.

    Returns
    -------
    dict
        The updated merged input JSON.

    Raises
    ------
    ValueError
        If the exploration type is neither "lammps" nor "i-PI".
    KeyError
        If a key is not found in any of the JSON dictionaries.
    TypeError
        If the value type is not int/float or bool (for disturbed_start).
    ValueError
        If the length of the value list is not equal to the system count.
    """
    system_count = len(main_json.get("systems_auto", []))

    previous_input_json = convert_control_to_input(previous_json, main_json)

    for key in [
        "exploration_type",
        "traj_count",
        "timestep_ps",
        "temperature_K",
        "exp_time_ps",
        "max_exp_time_ps",
        "job_walltime_h",
        "print_interval_mult",
        "previous_start",
        "disturbed_start",
    ]:
        # Get the value
        default_used = False
        if key in user_input_json:
            if (
                user_input_json[key] == "default" or user_input_json[key] is None
            ) and key in default_input_json:
                value = default_input_json[key]
                default_used = True
            else:
                value = user_input_json[key]
        elif key in previous_input_json:
            value = previous_input_json[key]
        elif key in default_input_json:
            value = default_input_json[key]
            default_used = True
        else:
            error_msg = f"'{key}' not found in any of the JSON dictionaries"
            raise KeyError(error_msg)

        if key == "exploration_type":
            merged_input_json[key] = []
            if default_used:
                merged_input_json[key] = [value[0]] * system_count
                exploration_dep = 0
            elif value == "lammps":
                merged_input_json[key] = [value] * system_count
                exploration_dep = 0
            elif value == "i-PI":
                merged_input_json[key] = [value] * system_count
                exploration_dep = 1
            elif value == "sander_emle":
                merged_input_json[key] = [value] * system_count
                exploration_dep = 0
            elif isinstance(value, List):
                exploration_dep = []
                if len(value) == system_count:
                    for it_value in value:
                        if (
                            it_value != "lammps"
                            and it_value != "i-PI"
                            and it_value != "sander_emle"
                        ):
                            error_msg = f"Exploration type {key} is not know: use ethier 'lammps' or 'i-PI' or 'sander_emle' or in a list"
                            raise ValueError(error_msg)
                        else:
                            if it_value == "lammps":
                                exploration_dep.append(0)
                            elif it_value == "i-PI":
                                exploration_dep.append(1)
                            elif it_value == "sander_emle":
                                exploration_dep.append(0)
                            merged_input_json[key].append(it_value)
                else:
                    error_msg = f"{key}: Size mismatch: The length of the list should be '{system_count}' corresponding to the number of systems."
                    raise ValueError(error_msg)
            else:
                error_msg = f"Exploration type {key} is not know: use ethier 'lammps' or 'i-PI' or 'sander_emle' or in a list"
                raise ValueError(error_msg)
        else:
            # Everything is system dependent so a list
            merged_input_json[key] = []

            # Default is used for the key
            if default_used:
                if isinstance(value[0], List):
                    if isinstance(exploration_dep, List):
                        merged_input_json[key] = [value[0][_] for _ in exploration_dep]
                    else:
                        merged_input_json[key] = [
                            value[0][exploration_dep]
                        ] * system_count
                else:
                    merged_input_json[key] = [value[0]] * system_count
            else:
                # Check if previous or user provided a list
                if isinstance(value, List):
                    if len(value) == system_count:
                        for it_value in value:
                            if (
                                isinstance(it_value, (int, float))
                                and (
                                    key != "disturbed_start" and key != "previous_start"
                                )
                            ) or (
                                isinstance(it_value, (bool))
                                and (
                                    key == "disturbed_start" or key == "previous_start"
                                )
                            ):
                                merged_input_json[key].append(it_value)
                            else:
                                error_msg = f"{key}: Type mismatch: the type is '{type(it_value)}', but it should be '{type(1)}' or '{type(1.0)}' or '{type(True)}' (for previous_start or disturbed_start)"
                                raise TypeError(error_msg)
                    else:
                        error_msg = f"{key}: Size mismatch: The length of the list should be '{system_count}' corresponding to the number of systems."
                        raise ValueError(error_msg)
                # If it is not a List
                elif (
                    isinstance(value, (int, float))
                    and (key != "disturbed_start" and key != "previous_start")
                ) or (
                    isinstance(value, (bool))
                    and (key == "disturbed_start" or key == "previous_start")
                ):
                    merged_input_json[key] = [value] * system_count
                else:
                    error_msg = f"{key}: Type mismatch: the type is '{type(it_value)}', but it should be '{type(1)}' or '{type(1.0)}' or '{type(True)}' (for previous_start or disturbed_start)"
                    raise TypeError(error_msg)

    return merged_input_json


# TODO: Add tests for this function
@catch_errors_decorator
def get_system_exploration(
    merged_input_json: Dict, system_auto_index: int
) -> Tuple[
    str,
    Union[float, int],
    Union[float, int],
    Union[float, int],
    Union[float, int],
    Union[float, int],
    Union[float, int],
    Union[float, int],
    bool,
    bool,
]:
    """
    Return a tuple of system exploration parameters based on the input JSON and system index.

    Parameters
    ----------
    merged_input_json : Dict[str, Any]
        A dictionary containing input parameters.
    system_auto_index : int
        An integer representing the system index.

    Returns
    -------
    Tuple[str, float, float, float, float, float, float, float, bool]
        A tuple containing system exploration parameters:
        - exploration_typ : str
            Type of exploration
        - traj_count : float
            Number of trajectories per system/NNP
        - timestep_ps : float
            Simulation timestep in picoseconds.
        - temperature_K : float
            Simulation temperature in Kelvin.
        - exp_time_ps : float
            Total exploration time in picoseconds.
        - max_exp_time_ps : float
            Maximum allowed exploration time in picoseconds.
        - job_walltime_h : float
            Maximum job walltime in hours.
        - print_interval_mult : float
            Print interval multiplier.
        - previous_start : bool
            Whether to start exploration from a previous minimum.
        - disturbed_start : bool
            Whether to start exploration from a disturbed minimum.
    """
    system_values = [
        merged_input_json[key][system_auto_index]
        for key in (
            "exploration_type",
            "traj_count",
            "timestep_ps",
            "temperature_K",
            "exp_time_ps",
            "max_exp_time_ps",
            "job_walltime_h",
            "print_interval_mult",
            "previous_start",
            "disturbed_start",
        )
    ]

    return tuple(system_values)


# TODO: Add tests for this function
@catch_errors_decorator
def generate_input_exploration_deviation_json(
    user_input_json: Dict,
    previous_json: Dict,
    default_input_json: Dict,
    merged_input_json: Dict,
    main_json: Dict,
) -> Dict:
    """
    Update and complete input JSON by incorporating values from the user input JSON, the previous JSON, and the default JSON.

    Parameters
    ----------
    user_input_json : dict
        The input JSON provided by the user, containing user-defined parameters.
    previous_json : dict
        The JSON from the previous iteration.
    default_input_json : dict
        The default input JSON containing default parameters.
    merged_input_json : dict
        The merged input JSON.
    main_json : dict
        The main JSON.

    Returns
    -------
    dict
        The updated merged input JSON.

    Raises
    ------
    ValueError
        If the exploration type is neither "lammps" nor "i-PI".
    KeyError
        If a key is not found in any of the JSON dictionaries.
    TypeError
        If the value type is not int/float or bool (for disturbed_start).
    ValueError
        If the length of the value list is not equal to the system count.
    """
    system_count = len(main_json.get("systems_auto", []))

    previous_input_json = convert_control_to_input(previous_json, main_json)

    for key in [
        "max_candidates",
        "sigma_low",
        "sigma_high",
        "sigma_high_limit",
        "ignore_first_x_ps",
    ]:
        # Get the value
        default_used = False
        if key in user_input_json:
            if (
                user_input_json[key] == "default" or user_input_json[key] is None
            ) and key in default_input_json:
                value = default_input_json[key]
                default_used = True
            else:
                value = user_input_json[key]
        elif key in previous_input_json:
            value = previous_input_json[key]
        elif key in default_input_json:
            value = default_input_json[key]
            default_used = True
        else:
            error_msg = f"'{key}' not found in any of the JSON dictionaries"
            raise KeyError(error_msg)

        # Everything is system dependent so a list
        merged_input_json[key] = []

        # Default is used for the key
        if default_used:
            merged_input_json[key] = [value[0]] * system_count
        else:
            # Check if previous or user provided a list
            if isinstance(value, List):
                if len(value) == system_count:
                    for it_value in value:
                        if isinstance(it_value, (int, float)):
                            merged_input_json[key].append(it_value)
                        else:
                            error_msg = f"Type mismatch: the type is '{type(it_value)}', but it should be '{type(1)}' or '{type(1.0)}'"
                            raise TypeError(error_msg)
                else:
                    error_msg = f"Size mismatch: The length of the list should be '{system_count}' corresponding to the number of systems."
                    raise ValueError(error_msg)

            # If it is not a List
            elif isinstance(value, (int, float)):
                merged_input_json[key] = [value] * system_count
            else:
                error_msg = f"Type mismatch: the type is '{type(value)}', but it should be '{type(1)}' or '{type(1.0)}'"
                raise TypeError(error_msg)
    return merged_input_json


# TODO: Add tests for this function
@catch_errors_decorator
def get_system_deviation(
    merged_input_json: Dict, system_auto_index: int
) -> Tuple[
    Union[float, int],
    Union[float, int],
    Union[float, int],
    Union[float, int],
    Union[float, int],
]:
    """
    Return a tuple of system exploration parameters based on the input JSON and system index.

    Parameters
    ----------
    merged_input_json : Dict[str, Any]
        A dictionary containing input parameters.
    system_auto_index : int
        An integer representing the system index.

    Returns
    -------
    Tuple[float, float, float, float, float, float, float, float, bool]
        A tuple containing system deviation parameters:
        - Max candidates : float
        - Sigma low : float
        - Sigma high : float
        - Sigma high limit : float
        - Ignore first x ps : float
    """
    system_values = [
        merged_input_json[key][system_auto_index]
        for key in (
            "max_candidates",
            "sigma_low",
            "sigma_high",
            "sigma_high_limit",
            "ignore_first_x_ps",
        )
    ]

    return tuple(system_values)


# TODO: Add tests for this function
@catch_errors_decorator
def generate_input_exploration_disturbed_json(
    user_input_json: Dict,
    previous_json: Dict,
    default_input_json: Dict,
    merged_input_json: Dict,
    main_json: Dict,
) -> Dict:
    """
    Update and complete input JSON by incorporating values from the user input JSON, the previous JSON, and the default JSON.

    Parameters
    ----------
    user_input_json : dict
        The input JSON provided by the user, containing user-defined parameters.
    previous_json : dict
        The JSON from the previous iteration.
    default_input_json : dict
        The default input JSON containing default parameters.
    merged_input_json : dict
        The merged input JSON.
    main_json : dict
        The main JSON.

    Returns
    -------
    dict
        The updated merged input JSON.

    Raises
    ------
    ValueError
        If the exploration type is neither "lammps" nor "i-PI".
    KeyError
        If a key is not found in any of the JSON dictionaries.
    TypeError
        If the value type is not int/float or bool (for disturbed_start).
    ValueError
        If the length of the value list is not equal to the system count.
    """
    system_count = len(main_json.get("systems_auto", []))

    previous_input_json = convert_control_to_input(previous_json, main_json)

    for key in [
        "disturbed_start_value",
        "disturbed_start_indexes",
        "disturbed_candidate_value",
        "disturbed_candidate_indexes",
    ]:
        # Get the value
        default_used = False
        if key in user_input_json:
            if (
                user_input_json[key] == "default" or user_input_json[key] is None
            ) and key in default_input_json:
                value = default_input_json[key]
                default_used = True
            else:
                value = user_input_json[key]
        elif key in previous_input_json:
            value = previous_input_json[key]
        elif key in default_input_json:
            value = default_input_json[key]
            default_used = True
        else:
            error_msg = f"'{key}' not found in any of the JSON dictionaries"
            raise KeyError(error_msg)

        # Everything is system dependent so a list
        merged_input_json[key] = []

        # Default is used for the key
        # Default is used for the key
        if default_used:
            merged_input_json[key] = [value[0]] * system_count
        else:
            if key == "disturbed_start_indexes" or key == "disturbed_candidate_indexes":
                is_list_of_list = False
                if isinstance(value, List):
                    # Search if it is a list of list
                    for it_value in value:
                        if isinstance(it_value, List):
                            is_list_of_list = True
                            break
                    if not is_list_of_list:
                        # If it is not a list of list, and empty, keep the empty list everywhere
                        if not value:
                            merged_input_json[key] = [value] * system_count
                        else:
                            for it_value in value:
                                if not isinstance(it_value, int):
                                    error_msg = f"Type mismatch: the type is '{type(it_value)}', but it should be '{type(1)}'."
                                    raise TypeError(error_msg)
                            merged_input_json[key] = [value] * system_count
                    else:
                        if len(value) != system_count:
                            error_msg = f"Size mismatch: The length of the list should be '{system_count}' corresponding to the number of systems."
                            raise ValueError(error_msg)
                        for it_value in value:
                            if not isinstance(it_value, List):
                                error_msg = f"Type mismatch: the type is '{type(it_value)}', but it should be '{type([])}'."
                                raise TypeError(error_msg)
                            if not it_value:
                                merged_input_json[key].append(it_value)
                            else:
                                for it_it_value in it_value:
                                    if not isinstance(it_it_value, (int)):
                                        error_msg = f"Type mismatch: the type is '{type(it_it_value)}', but it should be '{type(1)}'."
                                        raise TypeError(error_msg)
                                merged_input_json[key].append(it_value)
            elif isinstance(value, List):
                if len(value) == system_count:
                    for it_value in value:
                        if isinstance(it_value, (int, float)):
                            merged_input_json[key].append(it_value)
                        else:
                            error_msg = f"Type mismatch: the type is '{type(it_value)}', but it should be '{type(1)}' or '{type(1.0)}'"
                            raise TypeError(error_msg)
                else:
                    error_msg = f"Size mismatch: The length of the list should be '{system_count}' corresponding to the number of systems."
                    raise ValueError(error_msg)

            # If it is not a List
            elif isinstance(value, (int, float)):
                merged_input_json[key] = [value] * system_count
            else:
                error_msg = f"Type mismatch: the type is '{type(it_value)}', but it should be '{type(1)}' or '{type(1.0)}'"
                raise TypeError(error_msg)
    return merged_input_json


# TODO: Add tests for this function
@catch_errors_decorator
def get_system_disturb(
    merged_input_json: Dict, system_auto_index: int
) -> Tuple[
    Union[float, int],
    Union[float, int],
    List[int],
]:
    """
    Return a tuple of system exploration parameters based on the input JSON and system index.

    Parameters
    ----------
    merged_input_json : Dict[str, Any]
        A dictionary containing input parameters.
    system_auto_index : int
        An integer representing the system index.

    Returns
    -------
    Tuple[float, float, List]
        A tuple containing system deviation parameters:
        - disturbed_start_value : float
        - disturbed_candidate_value : float
        - disturbed_candidate_indexes : float
    """
    system_values = [
        merged_input_json[key][system_auto_index]
        for key in (
            "disturbed_start_value",
            "disturbed_start_indexes",
            "disturbed_candidate_value",
            "disturbed_candidate_indexes",
        )
    ]

    return tuple(system_values)


# TODO: Add tests for this function
@catch_errors_decorator
def generate_starting_points(
    exploration_type: str,
    system_auto: str,
    training_path: str,
    padded_prev_iter: str,
    previous_json: Dict,
    input_present: bool,
    previous_start: bool,
    disturbed_start: bool,
) -> Tuple[List[str], List[str], bool]:
    """
    Generates a list of starting point file names.

    Parameters
    ----------
    exploration_type : str
        An str representing the exploration type ("lammps" or "i-PI").
    system_auto : str
        The name of the system.
    training_path : str
        The path to the training directory.
    padded_prev_iter : str
        A zero-padded string representing the previous iteration number.
    previous_json : Dict
        A dictionary containing information about the previous exploration.
    input_present : bool
        A boolean indicating whether an input file is present.
    previous_start : bool
        A boolean indicating whether to start from a previous minimum.
    disturbed_start : bool
        A boolean indicating whether to start from a disturbed minimum.

    Returns
    -------
    Tuple[List[str], List[str], bool]
        A tuple containing the starting point file names, the backup starting point file names,
        and a boolean indicating wehter to start from a previous minimum and a boolean indicating whether to start from a disturbed minimum.
    """
    arcann_logger = logging.getLogger("ArcaNN")

    # Determine file extension based on exploration type
    if exploration_type == "lammps":
        file_extension = "lmp"
    elif exploration_type == "i-PI":
        file_extension = "xyz"
    # TODO: Implement the starting points for SANDER-EMLE
    elif exploration_type == "sander_emle":
        return None, None, False, False

    # Get list of starting point file names for system and iteration
    starting_points_path = list(
        Path(training_path / "starting_structures").glob(
            f"{padded_prev_iter}_{system_auto}_*.{file_extension}"
        )
    )
    orginal_starting_points_path = list(
        Path(training_path / "user_files").glob(f"{system_auto}.{file_extension}")
    )
    starting_points_all = [str(zzz).split("/")[-1] for zzz in starting_points_path]
    starting_points = [zzz for zzz in starting_points_all if "disturbed" not in zzz]
    starting_points_disturbed = [
        zzz for zzz in starting_points_all if zzz not in starting_points
    ]
    starting_points_original = [
        str(_).split("/")[-1] for _ in orginal_starting_points_path
    ]
    starting_points_bckp = deepcopy(starting_points)
    starting_points_disturbed_bckp = deepcopy(starting_points_disturbed)
    starting_points_original_bckp = deepcopy(starting_points_original)

    # Special case if no valid starting structures are found (either normal or disturbed)
    # Replace starting points with original starting points and disturbed starting points with original starting points too but do not changes (but just temporary)
    if not starting_points:
        starting_points = deepcopy(starting_points_original)
        starting_points_bckp = deepcopy(starting_points_original_bckp)
        arcann_logger.warning(
            f"No valid starting structures found for {system_auto} in iteration {padded_prev_iter}. Using original starting structures."
        )
    if not starting_points_disturbed and (
        disturbed_start
        or (
            previous_json["systems_auto"][system_auto]["disturbed_start"]
            and previous_json["systems_auto"][system_auto]["disturbed_start_value"]
        )
    ):
        starting_points_disturbed = deepcopy(starting_points_original)
        starting_points_disturbed_bckp = deepcopy(starting_points_original_bckp)
        arcann_logger.warning(
            f"No valid disturbed starting structures found for {system_auto} in iteration {padded_prev_iter}. Using original starting structures."
        )

    # Check if system should start from a disturbed minimum
    if input_present and not previous_start:
        starting_points = deepcopy(starting_points_original)
        starting_points_bckp = deepcopy(starting_points_original_bckp)
        return starting_points, starting_points_bckp, False, False
    elif input_present and disturbed_start:
        # If input file is present and disturbed start is requested, use disturbed starting points
        starting_points = deepcopy(starting_points_disturbed)
        starting_points_bckp = deepcopy(starting_points_disturbed_bckp)
        return starting_points, starting_points_bckp, True, True
    elif not input_present:
        if not (previous_json["systems_auto"][system_auto]["previous_start"]):
            starting_points = deepcopy(starting_points_original)
            starting_points_bckp = deepcopy(starting_points_original_bckp)
            return starting_points, starting_points_bckp, False, False
        # If input file is not present, check if system started from disturbed minimum in previous iteration
        elif (
            previous_json["systems_auto"][system_auto]["disturbed_start"]
            and previous_json["systems_auto"][system_auto]["disturbed_start_value"]
        ):
            # If system started from disturbed minimum in previous iteration, use disturbed starting points
            starting_points = deepcopy(starting_points_disturbed)
            starting_points_bckp = deepcopy(starting_points_disturbed_bckp)
            return starting_points, starting_points_bckp, True, True
        else:
            # Otherwise, use regular starting points
            return starting_points, starting_points_bckp, True, False
    else:
        # If input file is present but disturbed start is not requested, use regular starting points
        return starting_points, starting_points_bckp, True, False


# Unittested
@catch_errors_decorator
def create_models_list(
    main_json: Dict,
    previous_json: Dict,
    it_nnp: int,
    padded_prev_iter: str,
    training_path: Path,
    local_path: Path,
) -> Tuple[List[str], str]:
    """
    Generate a list of model file names and create symbolic links to the corresponding model files.

    Parameters
    ----------
    main_json : Dict
        The main JSON.
    previous_json : Dict
        The JSON from the previous iteration.
    it_nnp : int
        An integer representing the index of the NNP model to start from.
    padded_prev_iter : str
        A string representing the zero-padded iteration number of the previous iteration.
    training_path : Path
        The path to the training directory.
    local_path : Path
        The path to the local directory.

    Returns
    -------
    Tuple[List[str], str]
        A tuple containing the list of model file names, and a string of space-separated model file names.
    """
    # Generate list of NNP model indices and reorder based on current model to propagate
    list_nnp = [zzz for zzz in range(1, main_json["nnp_count"] + 1)]
    reorder_nnp_list = (
        list_nnp[list_nnp.index(it_nnp) :] + list_nnp[: list_nnp.index(it_nnp)]
    )

    # Determine whether to use compressed models
    compress_str = "_compressed" if previous_json["is_compressed"] else ""

    # Generate list of model file names
    models_list = [
        "graph_" + str(f) + "_" + padded_prev_iter + compress_str + ".pb"
        for f in reorder_nnp_list
    ]

    # Create symbolic links to the model files in the local directory
    for it_sub_nnp in range(1, main_json["nnp_count"] + 1):
        nnp_apath = (
            training_path
            / "NNP"
            / (
                "graph_"
                + str(it_sub_nnp)
                + "_"
                + padded_prev_iter
                + compress_str
                + ".pb"
            )
        ).resolve()
        subprocess.call(["ln", "-nsf", str(nnp_apath), str(local_path)])

    # Join the model file names into a single string for ease of use
    models_string = " ".join(models_list)

    return models_list, models_string


# Unittested
@catch_errors_decorator
def get_last_frame_number(
    model_deviation: np.ndarray, sigma_high_limit: float, disturbed_start: bool
) -> int:
    """
    Returns the index of the last frame to be processed based on the given parameters.

    Parameters
    ----------
    model_deviation : np.ndarray
        The model deviation data, represented as a NumPy array.
    sigma_high_limit : float
        The threshold value for the deviation data. Frames with deviation values above this threshold will be ignored.
    disturbed_start : bool
        Indicates whether the first frame should be ignored because it is considered "disturbed".

    Returns
    -------
    int
        The index of the last frame to be processed, based on the input parameters.
    """
    # Ignore the first frame if it's considered "disturbed"
    if disturbed_start:
        start_frame = 1
    else:
        start_frame = 0
    if model_deviation.shape[1] == 2:
        if np.any(model_deviation[start_frame:1] >= sigma_high_limit):
            last_frame = np.argmax(model_deviation[start_frame:, 1] >= sigma_high_limit)
        else:
            last_frame = -1
    else:
        if np.any(model_deviation[start_frame:, 4] >= sigma_high_limit):
            last_frame = np.argmax(model_deviation[start_frame:, 4] >= sigma_high_limit)
        else:
            last_frame = -1

    return last_frame


# TODO: Sould be renamed because it is not returning a factor or a number of steps but a time of simulation
# Unittested
@catch_errors_decorator
def update_system_nb_steps_factor(previous_json: Dict, system_auto_index: int) -> int:
    """
    Calculates a ratio based on information from a dictionary and returns the multiplied system_nb_steps.

    Parameters
    ----------
    previous_json : Dict
        A dictionary containing information about the previous iteration.
    system_auto_index : int
        An integer representing the system index.

    Returns
    -------
    int
        An integer representing the multiplying factor for system_nb_steps.

    Notes
    -----
    This function calculates a ratio of ill-described candidates to the total number of candidates in the previous iteration.
    Based on this ratio, it returns a multiplying factor for system_nb_steps.

    The multiplying factor is determined as follows:
    - If the ill-described ratio is less than 0.10, the factor is 4 times the previous system_nb_steps.
    - If the ill-described ratio is between 0.10 and 0.20, the factor is 2 times the previous system_nb_steps.
    - Otherwise, the factor is equal to the previous system_nb_steps.

    """
    # Calculate the ratio of ill-described candidates to the total number of candidates
    ill_described_ratio = (
        previous_json["systems_auto"][system_auto_index]["candidates_count"]
        + previous_json["systems_auto"][system_auto_index]["rejected_count"]
    ) / previous_json["systems_auto"][system_auto_index]["total_count"]

    # Return a multiplying factor for system_nb_steps based on the ratio of ill-described candidates
    if ill_described_ratio < 0.10:
        return (
            4
            * previous_json["systems_auto"][system_auto_index]["nb_steps"]
            * previous_json["systems_auto"][system_auto_index]["timestep_ps"]
        )
    elif ill_described_ratio < 0.20:
        return (
            2
            * previous_json["systems_auto"][system_auto_index]["nb_steps"]
            * previous_json["systems_auto"][system_auto_index]["timestep_ps"]
        )
    else:
        return (
            1
            * previous_json["systems_auto"][system_auto_index]["nb_steps"]
            * previous_json["systems_auto"][system_auto_index]["timestep_ps"]
        )

"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2026 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
"""

import builtins
import re
import sys
import warnings
from collections.abc import Iterable
from contextlib import contextmanager
from enum import auto

try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum  # type: ignore

from pathlib import Path
from typing import Literal

from arcann_training.common.utils import catch_errors_decorator


@contextmanager
def patched_cli(argv, input_value=""):
    """
    Context manager that temporarily replaces sys.argv and builtins.input for the duration of a with-block.

    This is intended for handling command-line interfaces and functions that call input(). The original
    sys.argv and builtins.input are saved on entry and restored on exit, even if an exception is raised
    within the with-block.

    Parameters
    ----------
    argv : Sequence[str]
        Sequence of strings to assign to sys.argv. It will be converted to a list before assignment.
    input_value : str, optional
        Value to return from builtins.input calls while the context manager is active. Default is the
        empty string. The patched input callable accepts a single prompt argument which is ignored and
        always returns this value.

    Yields
    ------
    None
        The context manager does not yield a value; it simply provides a temporary environment in which
        sys.argv and builtins.input are patched.

    Notes
    -----
    - The patch is global within the Python process and is not thread-safe.
    - The original values are restored in a finally block to ensure cleanup even if the with-block
    raises an exception.

    Examples
    --------
    >>> from arcann_training.exploration.lammps import patched_cli
    >>> with patched_cli(["prog", "--flag"], input_value="yes"):
    ...     # inside this block, sys.argv == ['prog', '--flag'] and input() returns 'yes'
    ...     pass
    """
    old_argv = sys.argv
    old_input = builtins.input

    try:
        sys.argv = list(argv)
        builtins.input = lambda _: input_value
        yield
    finally:
        sys.argv = old_argv
        builtins.input = old_input


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
        "dump traj_xyz all custom _R_PRINT_FREQ_ _RI_NAME__mace_run_model1.lammpstrj id type x y z",
        "dump traj_frc all custom _R_PRINT_FREQ_ _RI_NAME__mace_forces_model1.lammpstrj id type x y z fx fy fz",
        "dump_modify traj_xyz sort id",
        "dump_modify traj_frc sort id",
    )

    def __init__(self, lmp_input: str | Path, elements: list[str]):
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
        lmp_input = Path(lmp_input)
        if lmp_input.exists():
            self._lmp_input = Path(lmp_input)
        else:
            raise FileNotFoundError(f"LAMMPS input not found: {lmp_input}")

        self._raw_text = self._read_input_file()

        self._el = elements
        self._models = None
        self._parse_metadata()
        self._prepare_input_text()

        self._raw_text = self.apply_variables(
            {"_R_ATOM_LABELS_": " ".join(elements), "_RI_NAME_": self._lmp_input.stem}
        )

    def __str__(self):
        return str(self._raw_text)

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
            if re.search(
                r"^\s*(?!#)\s*fix\s+\S+\s+\S+\s+plumed\b", self._raw_text, re.MULTILINE
            )
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
        match_run = re.search(
            r"^\s*(?!#)run\s+_R_NUMBER_OF_STEPS_", self._raw_text, re.MULTILINE
        )
        if not match_run:
            raise ValueError(
                f"No 'run _R_NUMBER_OF_STEPS_' found in the LAMMPS input file: {self._lmp_input}"
            )

        match_rest = re.search(
            r"^\s*(?!#)write_restart\s+_R_RESTART_OUT_", self._raw_text, re.MULTILINE
        )

        if not match_rest:
            raise ValueError(
                f"'write_restart' not found in LAMMPS input, please add it using: 'write_restart _R_RESTART_OUT_' : {self._lmp_input}"
            )

        run_index = match_run.start()
        self._raw_text = (
            self._raw_text[:run_index]
            + "\n".join(self.cell_info_lammps)
            + "\n".join(self.mace_dump_0)
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
            formatted_text = formatted_text.replace(key, value)

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

    @property
    def input_file(self) -> Path:
        """Get the lammps input Path object.

        Returns
        -------
        Path
            The lammps input Path object.
        """
        return self._lmp_input

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
    def set_models(self, models: list[str]) -> None:
        self._models = models
        if self.lmp_pair == LAMMPSPair.DEEPMD:
            self._raw_text = self.apply_variables({"_R_MODEL_FILES_": " ".join(models)})
        else:
            self._raw_text = self.apply_variables({"_R_MODEL_FILES_": models[0]})
            self._prepare_mace_rerun()

    @catch_errors_decorator
    def _prepare_mace_rerun(self) -> None:
        for i, model in enumerate(self._models[1:], start=2):
            self._raw_text += f"""\n# ------------------
# Rerun with model {i}
# ------------------
clear

atom_style atomic
atom_modify map yes

read_restart _R_RESTART_OUT_

{self._get_pair_lmp_string(model)}

thermo_style custom step pe
thermo 1

dump traj{i} all custom _R_PRINT_FREQ_ {self._lmp_input.stem}_mace_forces_model{i}.lammpstrj id type x y z fx fy fz
dump_modify traj{i} sort id

rerun {self._lmp_input.stem}_mace_run_model1.lammpstrj dump x y z
"""

    def _get_pair_lmp_string(self, model: str) -> str:
        elements = " ".join(self._el)
        match self.lmp_pair:
            case LAMMPSPair.MACE:
                return f"pair_style mace {'no_domain_decomposition' if self.domain_decomp else ''}\npair_coeff * * {model} {elements}"
            case LAMMPSPair.MLIAP:
                return f"pair_style mliap unified {model} 0\npair_coeff * * {elements}"
            case LAMMPSPair.SYMMETRIX:
                return f"pair_style symmetrix/mace\npair_coeff * * {model} {elements}"


@catch_errors_decorator
def mace_model_converter(
    model: Path,
    to: Literal["mace", "mliap", "symmetrix"] | LAMMPSPair,
    cmd_args: dict = None,
):
    if cmd_args is None:
        cmd_args = []
    else:
        if "model" in cmd_args:
            cmd_args.pop("model")

        match to:
            case LAMMPSPair.MACE:
                cmd_args["format"] = "libtorch"
            case LAMMPSPair.MLIAP:
                cmd_args["format"] = "mliap"

        cmd_args = {f"--{k}": v for k, v in cmd_args.items()}
        cmd_args_list = []

        for k, v in cmd_args.items():
            if isinstance(v, Iterable) and not isinstance(v, str):
                cmd_args_list.extend((k, *v))
            else:
                cmd_args_list.extend((k, v))
    try:
        from mace.cli.convert_device import main as mace_device_convert
        from mace.cli.create_lammps_model import main as mace_lammps_model
    except ImportError as e:
        raise ImportError(
            "ArcaNN needs MACE installed in the same python environment to use LAMMPS."
        ) from e

    # patched_cli is needed because the conversion is configured with ArgParser
    # and there is no method to do it directly

    argv = [
        "convert_lammps",
        str(model.resolve()),
        *cmd_args_list,
    ]  # argv[0] does not matter, can be anything

    warnings.filterwarnings("ignore", category=UserWarning)

    if to != "symmetrix":
        with patched_cli(argv):
            mace_lammps_model()
    else:
        try:
            from symmetrix.cli.extract_mace import main as mace_symmetrix_model
        except ImportError as e:
            raise ImportError(
                "ArcaNN needs symmetrix installed in the same python environment to use LAMMPS with symmetrix."
            ) from e

        argv.insert(1, "-m")  # symmetrix convert needs flag -m

        # SYMMETRIX needs the model to be for CPU
        with patched_cli(["convert_device", str(model.resolve()), "-t", "cpu"]):
            mace_device_convert()

        argv[2] = str(model.with_name(model.name + ".cpu").resolve())

        with patched_cli(argv):
            mace_symmetrix_model()

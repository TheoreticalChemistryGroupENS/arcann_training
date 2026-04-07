"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2025/12/19
Last modified: 2025/12/19
"""

import logging
from pathlib import Path
from typing import Dict, Literal, Union

import ase
import numpy as np
from ase.io import read
from ase.io.extxyz import write_extxyz

from arcann_training.common.filesystem import check_directory, check_file_existence
from arcann_training.common.json import load_json_file, write_json_file
from arcann_training.common.list import string_list_to_textfile, textfile_to_string_list
from arcann_training.initialization.utils import check_typeraw_properties

arcann_logger = logging.getLogger("ArcaNN")


class DataEnsemble:
    """
    Class defining an ensemble of data (understand a set of initial, or relabeled data)

    Attributes:
    ----------
        path (Path): Path to the data ensemble
        step (Literal["initial", "extra", "system_auto", "system_adhoc", "system_disturbed"]):
            Type of the data ensemble in the step of the arcann workflow
        training_type (Literal["training", "validation"]): Whether the data ensemble is used for training or validation
        system_name (str | None): Name of the system from which the data ensemble originates. None if not applicable
        iteration (int | None): Iteration (of arcann) number from which the data ensemble originates. None if not applicable
        data_format (Literal["extxyz", "set.000"]): Type of data ensemble
        properties (Dict[int, Dict[str, str | float]]): Properties associated with the data ensemble

    Methods:
    --------
        to_dict(): Convert the data ensemble to a dictionary
        check_format(): Check the format of the data ensemble
        get_size(): Get the size of the data ensemble (ie. nb of confs)
        load_data(): Load the data of data ensemble
        get_extxyz(): Get the data under the extxyz format
        get_set000(): Get the data under the set.000 format
    """

    def __init__(
        self,
        path: str | Path,
        step: Literal[
            "initial", "extra", "system_auto", "system_adhoc", "system_disturbed"
        ],
        training_type: Literal["training", "validation", "test"],
        system_name: str | None,
        iteration: int | None,
        data_format: Literal["extxyz", "set.000"],
        properties: Dict[int, Dict[str, str | float]],
    ):
        check_directory(
            Path(path),
            abort_on_error=True,
            error_msg="The provided data path does not exist.",
        )
        self.path = Path(path)
        self.step = step
        self.training_type = training_type
        self.system_name = system_name
        self.iteration = iteration

        self.data_format = data_format
        self.properties = {int(k): v for k, v in properties.items()}

        self.size = None

        # proper data
        self.type = None
        self.box = None
        self.coord = None
        self.energy = None
        self.force = None
        self.virial = None
        self.wannier = None
        self.is_periodic = None
        self.wannier_not_cvg = None

    def to_dict(self):
        """Convert the data ensemble to a dictionary for saving in the control file"""
        return {
            "path": str(self.path),
            "step": self.step,
            "training_type": self.training_type,
            "system_name": self.system_name,
            "iteration": self.iteration,
            "data_format": self.data_format,
            "properties": self.properties,
        }

    def check_format(self):
        raise NotImplementedError

    def get_size(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def load_from_raw_arrays(
        self,
        type: np.ndarray,
        energy: np.ndarray,
        coord: np.ndarray,
        box: np.ndarray,
        force: np.ndarray,
        virial: np.ndarray | None,
        wannier: np.ndarray | None,
        wannier_not_cvg: list,
        is_periodic: bool,
    ):
        """From the raw files (type.raw, energy.raw, coord.raw, box.raw, force.raw), load the ensemble attributes"""
        self.type = type
        self.box = box
        self.coord = coord
        self.energy = energy
        self.force = force
        self.virial = virial
        self.wannier = wannier
        self.is_periodic = is_periodic
        self.wannier_not_cvg = wannier_not_cvg
        self.size = self.box.shape[0]

    def write(self):
        raise NotImplementedError


class Set000Ensemble(DataEnsemble):
    """Define a data ensemble in the set.000 format"""

    def __init__(
        self, path, step, training_type, system_name, iteration, data_format, properties
    ):
        super().__init__(
            path, step, training_type, system_name, iteration, data_format, properties
        )
        if data_format != "set.000":
            raise ValueError("Data type must be 'set.000' for Set000Ensemble")
        self.size = self.get_size()

    def check_format(self) -> None:
        """
        Check the format of the data ensemble by checking the existence of all the files and the consistency
        of the type.raw file with the properties
        """
        check_file_existence(self.path / "type.raw")
        check_typeraw_properties(self.path / "type.raw", self.properties)
        set_path = self.path / "set.000"
        for data_type in ["box", "coord", "energy", "force"]:
            check_file_existence(set_path / (data_type + ".npy"))

    def get_size(self) -> int | None:
        if (self.path / "set.000" / "box.npy").is_file():
            return np.load(self.path / "set.000" / "box.npy").shape[0]
        return None

    def load(self) -> None:
        """Load the data of the ensemble in the attributes"""
        self.check_format()
        self.type = np.loadtxt(self.path / "type.raw", dtype=int)
        self.box = np.load(self.path / "set.000" / "box.npy")
        self.coord = np.load(self.path / "set.000" / "coord.npy")
        self.energy = np.load(self.path / "set.000" / "energy.npy")
        self.force = np.load(self.path / "set.000" / "force.npy")
        self.virial = (
            np.load(self.path / "set.000" / "virial.npy")
            if (self.path / "set.000" / "virial.npy").is_file()
            else None
        )
        self.wannier = (
            np.load(self.path / "set.000" / "wannier.npy")
            if (self.path / "set.000" / "wannier.npy").is_file()
            else None
        )
        self.is_periodic = not (self.path / "nopbc").is_file()
        self.wannier_not_cvg = (
            textfile_to_string_list(self.path / "set.000" / "wannier-not-converged.txt")
            if (self.path / "set.000" / "wannier-not-converged.txt").is_file()
            else []
        )
        self.size = self.get_size()

    def write(self):
        """Write the data of the ensemble in the set.000 format from the attributes"""
        np.savetxt(self.path / "type.raw", self.type, fmt="%s")
        (self.path / "set.000").mkdir(parents=True, exist_ok=True)
        np.save(self.path / "set.000" / "box.npy", self.box)
        np.save(self.path / "set.000" / "coord.npy", self.coord)
        np.save(self.path / "set.000" / "energy.npy", self.energy)
        np.save(self.path / "set.000" / "force.npy", self.force)
        if self.virial:
            np.save(self.path / "set.000" / "virial.npy", self.virial)
        if self.wannier:
            np.save(self.path / "set.000" / "wannier.npy", self.wannier)
        if self.is_periodic is not None and not self.is_periodic:
            np.savetxt(self.path / "nopbc", np.array([True]), fmt="%s")
        if len(self.wannier_not_cvg) > 1:
            string_list_to_textfile(
                self.path / "set.000" / "wannier-not-converged.txt",
                self.wannier_not_cvg,
            )

    def to_extxyz(self, save: bool = False) -> list[ase.Atoms]:
        """
        Convert the attributes data into a list of ASE Atoms objects and optionally
        save this trajectory in the extxyz format.
        """
        self.load()
        elements = [self.properties[int(tp) + 1]["symbol"] for tp in self.type]
        frames = []
        for i in range(self.size):
            frame = ase.Atoms(
                symbols=elements,
                positions=self.coord[i].reshape(-1, 3),
                cell=self.box[i].reshape(3, 3),
                pbc=self.is_periodic,
            )
            frame.info["REF_energy"] = np.array([float(self.energy[i])])
            frame.arrays["REF_forces"] = self.force[i].reshape(-1, 3)
            if self.virial is not None:
                frame.info["REF_virials"] = self.virial[i].reshape(3, 3)
            frames.append(frame)

        if save:
            with (self.path / f"{self.path.name}.extxyz").open("w+") as f:
                write_extxyz(f, frames)
            if self.wannier is not None:
                np.save(self.path / "wannier.npy", self.wannier)
        return frames


class ExtXYZEnsemble(DataEnsemble):
    """Define a data ensemble in the extxyz format"""

    def __init__(
        self, path, step, training_type, system_name, iteration, data_format, properties
    ):
        super().__init__(
            path, step, training_type, system_name, iteration, data_format, properties
        )
        if data_format != "extxyz":
            raise ValueError("Data type must be 'extxyz' for ExtXYZEnsemble")
        self.size = self.get_size()

    def check_format(
        self, forces_keyword="REF_forces", energy_keyword="REF_energy"
    ) -> None:
        """Check the format of the data ensemble by checking the existence of the extxyz file and
        the presence of forces and energy in each frame of the trajectory.
        """
        check_file_existence(self.path / f"{self.path.name}.extxyz")
        trajectory = read(self.path / f"{self.path.name}.extxyz", index=":")
        if forces_keyword not in trajectory[0].arrays:
            raise ValueError(
                f"The extxyz file {self.path / f'{self.path.name}.extxyz'} does not contain forces in the arrays."
            )
        if energy_keyword not in trajectory[0].info:
            raise ValueError(
                f"The extxyz file {self.path / f'{self.path.name}.extxyz'} does not contain energy."
            )

    def get_size(self) -> int | None:
        if (self.path / f"{self.path.name}.extxyz").is_file():
            trajectory = read(self.path / f"{self.path.name}.extxyz", index=":")
            return len(list(trajectory))
        return None

    def load(self) -> None:
        """Load the data of the ensemble in the attributes"""
        self.check_format()
        trajectory = read(self.path / f"{self.path.name}.extxyz", index=":")
        self.box = np.array([atoms.get_cell().reshape(-1) for atoms in trajectory])
        self.coord = np.array(
            [atoms.get_positions().reshape(-1) for atoms in trajectory]
        )
        self.force = np.array(
            [atoms.arrays["REF_forces"].reshape(-1) for atoms in trajectory]
        )
        self.energy = np.array([atoms.info["REF_energy"] for atoms in trajectory])

        self.virial = (
            np.array([atoms.info["REF_virials"].reshape(-1) for atoms in trajectory])
            if "virial" in trajectory[0].info
            else None
        )
        self.wannier = (
            np.load(self.path / "wannier.npy")
            if (self.path / "wannier.npy").is_file()
            else None
        )

        self.is_periodic = trajectory[0].get_pbc()
        self.type = np.array(
            [
                int(k) - 1
                for elm in trajectory[0].get_chemical_symbols()
                for k, v in self.properties.items()
                if v["symbol"] == elm
            ]
        )
        self.wannier_not_cvg = (
            textfile_to_string_list(self.path / "wannier-not-converged.txt")
            if (self.path / "wannier-not-converged.txt").is_file()
            else []
        )
        self.size = self.get_size()
        return trajectory

    def write(self) -> None:
        """
        Write the data of the ensemble in the extxyz format from the attributes by converting
        them into a list of ASE Atoms objects saved as a extxyz trajectory.
        """
        elements = [self.properties[int(tp) + 1]["symbol"] for tp in self.type]
        frames = []
        for i in range(self.size):
            frame = ase.Atoms(
                symbols=elements,
                positions=self.coord[i].reshape(-1, 3),
                cell=self.box[i].reshape(3, 3),
                pbc=self.is_periodic,
            )
            frame.info["REF_energy"] = np.array([float(self.energy[i])])
            frame.arrays["REF_forces"] = self.force[i].reshape(-1, 3)
            if self.virial is not None:
                frame.info["REF_virials"] = self.virial[i].reshape(3, 3)
            frames.append(frame)

        if self.is_periodic is not None and not self.is_periodic:
            np.savetxt(self.path / "nopbc", np.array([True]), fmt="%s")
        if self.wannier:
            np.save(self.path / "wannier.npy", self.wannier)
        if len(self.wannier_not_cvg) > 1:
            string_list_to_textfile(
                self.path / "wannier-not-converged.txt", self.wannier_not_cvg
            )

        with (self.path / f"{self.path.name}.extxyz").open("w+") as f:
            write_extxyz(f, frames)

    def to_extxyz(self, save: bool = False) -> list[ase.Atoms]:
        """Return the data of the ensemble in the extxyz format as a list of ASE Atoms objects.
        If save is True, save the trajectory in the extxyz format."""
        if save:
            self.write()
        return self.load()

    def to_set000(self):
        """Convert the data of the ensemble in the set.000 format by writing the corresponding files from the attributes."""
        self.load()

        np.savetxt(self.path / "type.raw", self.type, fmt="%s")
        (self.path / "set.000").mkdir(parents=True, exist_ok=True)
        np.save(self.path / "set.000" / "box.npy", self.box)
        np.save(self.path / "set.000" / "coord.npy", self.coord)
        np.save(self.path / "set.000" / "energy.npy", self.energy)
        np.save(self.path / "set.000" / "force.npy", self.force)
        if self.virial:
            np.save(self.path / "set.000" / "virial.npy", self.virial)
        if self.wannier:
            np.save(self.path / "set.000" / "wannier.npy", self.wannier)
        if self.is_periodic is not None and not self.is_periodic:
            np.savetxt(self.path / "nopbc", np.array([True]), fmt="%s")
        if len(self.wannier_not_cvg) > 1:
            string_list_to_textfile(
                self.path / "set.000" / "wannier-not-converged.txt",
                self.wannier_not_cvg,
            )


class Dataset:
    """
    Class defining the dataset used for training and validation, as a list of DataEnsemble objects.
    Each DataEnsemble corresponds to a set of data (initial or relabeled from a specific system md
    simulation etc..) stored in a specific format (extxyz or set.000).

    Attributes:
    -----------
        training_dataset (Dict[str, DataEnsemble]): Dictionary of training data ensembles
        validation_dataset (Dict[str, DataEnsemble]): Dictionary of validation data ensembles
        training_paths (List[str]): List of paths to training data ensembles
        validation_paths (List[str]): List of paths to validation data ensembles
        config_file (Dict): Config file (usually config.json) containing information about the configuration
        dataset_dir (Path): Path to the dataset directory
        data_format (Literal["extxyz", "set.000"]): Type of data ensemble
        data_ensemble (DataEnsemble): Class of the data ensembles

    Methods:
    --------
        get_training_dataset(): Get the training data ensembles
        get_validation_dataset(): Get the validation data ensembles
        init_data_format(): Initialize the data type and data ensemble class
        init_dataset(main_json): Initialize the dataset from the dataset directory
        update_dataset(): Update the data ensembles available for the dataset
        init_config_file(): Initialize the control file
        update_config_file(): Update the control file
    """

    def __init__(
        self,
        training_dir: str | Path,
        config_file: Dict,
    ):
        # Attributes
        self.training_dataset: Dict[str, DataEnsemble] = {}
        self.validation_dataset: Dict[str, DataEnsemble] = {}
        self.training_paths = []
        self.validation_paths = []

        self.control_file = load_json_file(
            training_dir / "control" / "dataset.json", abort_on_error=False
        )
        self.control_file.setdefault("used_datasets", {})
        self.config_file = config_file
        self.split = self.config_file["validation/training_split"]
        self.dataset_dir = Path(training_dir) / "data"
        check_directory(
            self.dataset_dir,
            abort_on_error=True,
            error_msg="The provided dataset directory does not exist.",
        )
        self.init_data_format()  # get data type and data ensemble class

    def __str__(self):
        pass

    def get_training_dataset(self):
        raise NotImplementedError

    def get_validation_dataset(self):
        raise NotImplementedError

    def init_data_format(self) -> None:
        """From the directory of the dataset, identify the extxyz or set.000 format"""
        provided_format = self.config_file["data_format"]
        if provided_format == "set.000":
            self.data_format = "set.000"
            self.data_ensemble = Set000Ensemble
        elif provided_format == "extxyz":
            self.data_format = "extxyz"
            self.data_ensemble = ExtXYZEnsemble
        else:
            raise ValueError(
                f"The provided data format {provided_format} is not recognized. Please use 'set.000' or 'extxyz'."
            )

        # datasets = os.listdir(self.dataset_dir)
        # for dataset in datasets:
        #     dataset_path = self.dataset_dir / dataset
        #     if dataset_path.is_dir():
        #         if (dataset_path / "set.000").is_dir() and self.data_format != "set.000":
        #             self.convert
        #         elif any(file.suffix == ".extxyz" for file in dataset_path.glob("*.extxyz")):

    def read_dataset(self) -> Union[int, int]:
        """Read the datasets from the already processed datasets in the control file"""

        self.training_dataset = {
            key: self.data_ensemble(**kwargs)
            for key, kwargs in self.control_file["used_datasets"]["training"].items()
        }
        self.training_paths = list(self.training_dataset.keys())
        self.validation_dataset = {
            key: self.data_ensemble(**kwargs)
            for key, kwargs in self.control_file["used_datasets"]["validation"].items()
        }
        self.validation_paths = list(self.validation_dataset.keys())

    def load_dataset(
        self,
        extra_dataset: bool = True,
        init_dataset: bool = True,
        only_init: bool = False,
    ) -> Union[int, int, int]:
        """Load the new dataensembles from the dataset directory, depending on the control file information"""
        # TODO this function has to go!!!
        dataset_names = (
            self.training_paths + self.validation_paths
        )  # already existing/treated data ensembles
        common_kwargs = {
            "data_format": self.data_format,
            "properties": self.config_file["properties"],
        }  # attributes common to all data ensembles

        system_count, system_val_count, adhoc_count, adhoc_val_count = 0, 0, 0, 0

        # Read the NEW data ensembles
        for datadir in self.dataset_dir.iterdir():
            if datadir.is_dir() and datadir.name not in dataset_names:
                step, system_name, iteration = None, None, None
                if datadir.name.startswith("extra_"):
                    if extra_dataset and not only_init:
                        # case of extra datasets
                        step = "extra"
                    else:
                        continue

                elif datadir.name.startswith("init_"):
                    if init_dataset or only_init:
                        # case of initial datasets
                        step = "initial"
                    else:
                        continue

                elif not only_init:
                    # case of system datasets
                    system_name, iteration = datadir.name.rsplit("_", 1)
                    system_name = system_name.removesuffix("_valid")
                    arcann_logger.debug(
                        f"Loading dataset from system {system_name} at iteration {iteration} from {datadir.name}"
                    )
                    iteration = int(iteration)
                    if system_name in self.config_file["systems_auto"]:
                        step = "system_auto"
                        if "valid" in datadir.name:
                            system_val_count += 1
                        else:
                            system_count += 1
                    elif (
                        "-disturbed" in system_name
                        and system_name.removesuffix("-disturbed")
                        in self.config_file["systems_adhoc"]
                    ):
                        step = "system_disturbed"
                        if "valid" in datadir.name:
                            system_val_count += 1
                        else:
                            system_count += 1
                    else:
                        step = "system_adhoc"
                        if "valid" in datadir.name:
                            adhoc_val_count += 1
                        else:
                            adhoc_count += 1

                if step:
                    if "valid" in datadir.name:
                        self.validation_dataset[datadir.name] = self.data_ensemble(
                            path=datadir,
                            step=step,
                            training_type="validation",
                            system_name=system_name,
                            iteration=iteration,
                            **common_kwargs,
                        )
                    else:
                        self.training_dataset[datadir.name] = self.data_ensemble(
                            path=datadir,
                            step=step,
                            training_type="training",
                            system_name=system_name,
                            iteration=iteration,
                            **common_kwargs,
                        )
        self.training_paths = list(self.training_dataset.keys())
        self.validation_paths = list(self.validation_dataset.keys())
        return system_count, system_val_count, adhoc_count, adhoc_val_count

    def check_dataset(self) -> None:
        """Check the format of all data ensembles in the dataset"""
        for dataset in self.training_dataset.values():
            dataset.check_format()
        for dataset in self.validation_dataset.values():
            dataset.check_format()

        if len(self.training_dataset) != len(self.training_paths):
            raise ValueError("Mismatch between training dataset and training paths")
        if len(self.validation_dataset) != len(self.validation_paths):
            raise ValueError("Mismatch between validation dataset and validation paths")

    def remove_datasets(self, init_dataset: bool = True):
        if init_dataset:
            # remove the initial datasets
            self.training_dataset = {
                key: dataset
                for key, dataset in self.training_dataset.items()
                if dataset.step != "initial"
            }
            self.validation_dataset = {
                key: dataset
                for key, dataset in self.validation_dataset.items()
                if dataset.step != "initial"
            }
            self.training_paths = list(self.training_dataset.keys())
            self.validation_paths = list(self.validation_dataset.keys())

    def add_system_dataset(
        self,
        step: str,
        system_name: str,
        iteration: str,
        type: np.ndarray,
        energy: np.ndarray,
        coord: np.ndarray,
        box: np.ndarray,
        force: np.ndarray,
        virial: np.ndarray | None,
        wannier: np.ndarray | None,
        wannier_not_cvg: list | None,
        is_periodic: bool = True,
    ) -> None:
        """Add a new system dataset to the dataset.
        Create two data ensembles (training and validation) from the provided raw arrays."""
        if step == "system_disturbed":
            system_name += "-disturbed"
        training_dir = self.dataset_dir / f"{system_name}_{iteration}"
        training_dir.mkdir(exist_ok=True)
        validation_dir = self.dataset_dir / f"{system_name}_valid_{iteration}"
        validation_dir.mkdir(exist_ok=True)

        common_kwargs = {
            "step": step,
            "system_name": system_name,
            "iteration": int(iteration),
            "data_format": self.data_format,
            "properties": self.config_file["properties"],
        }
        training_data_ensemble = self.data_ensemble(
            path=training_dir, training_type="training", **common_kwargs
        )
        validation_data_ensemble = self.data_ensemble(
            path=validation_dir, training_type="validation", **common_kwargs
        )

        nb_of_frames = coord.shape[0]
        indices = np.random.permutation(nb_of_frames)
        split_idx = int(nb_of_frames * (1 - self.split))
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        arrays = {
            "energy": energy,
            "coord": coord,
            "box": box,
            "force": force,
            "virial": virial,
            "wannier": wannier,
        }
        train_arrays = {
            name: arr[train_idx] if arr is not None else None
            for name, arr in arrays.items()
        }
        val_arrays = {
            name: arr[val_idx] if arr is not None else None
            for name, arr in arrays.items()
        }
        # TODO probably these wannier not cvg should be splitted too but i dunno how
        training_data_ensemble.load_from_raw_arrays(
            type=type,
            **train_arrays,
            wannier_not_cvg=wannier_not_cvg,
            is_periodic=is_periodic,
        )
        training_data_ensemble.write()
        validation_data_ensemble.load_from_raw_arrays(
            type=type,
            **val_arrays,
            wannier_not_cvg=wannier_not_cvg,
            is_periodic=is_periodic,
        )
        validation_data_ensemble.write()

        self.control_file.setdefault("intermediate_datasets", {})
        self.control_file["intermediate_datasets"] |= {
            training_dir.name: training_data_ensemble.size,
            validation_dir.name: validation_data_ensemble.size,
        }

        # TODO at some point, we should add this new data ensemble that we creating to the dict so that they are
        # saved to the control file as system dataset and read at the next iteration without having to check if
        # they are new of not

    def update_control_file(self):
        # Write the LOADED new data ensembles to the control file and save it
        # Be careful to not use that without having loaded the dataset first!!
        self.control_file["initial_datasets"] = {
            **{
                key: dataset.size
                for key, dataset in self.training_dataset.items()
                if dataset.step == "initial"
            },
            **{
                key: dataset.size
                for key, dataset in self.validation_dataset.items()
                if dataset.step == "initial"
            },
        }
        self.control_file["system_datasets"] = {
            **{
                key: dataset.size
                for key, dataset in self.training_dataset.items()
                if dataset.step in ["system_auto", "system_disturbed"]
            },
            **{
                key: dataset.size
                for key, dataset in self.validation_dataset.items()
                if dataset.step in ["system_auto", "system_disturbed"]
            },
        }
        self.control_file["extra_datasets"] = {
            **{
                key: dataset.size
                for key, dataset in self.training_dataset.items()
                if dataset.step == "extra"
            },
            **{
                key: dataset.size
                for key, dataset in self.validation_dataset.items()
                if dataset.step == "extra"
            },
        }
        self.control_file["adhoc_datasets"] = {
            **{
                key: dataset.size
                for key, dataset in self.training_dataset.items()
                if dataset.step == "system_adhoc"
            },
            **{
                key: dataset.size
                for key, dataset in self.validation_dataset.items()
                if dataset.step == "system_adhoc"
            },
        }
        self.control_file["used_datasets"]["training"] = {
            key: dataset.to_dict() for key, dataset in self.training_dataset.items()
        }
        self.control_file["used_datasets"]["validation"] = {
            key: dataset.to_dict() for key, dataset in self.validation_dataset.items()
        }

        self.save_control_file()

    def save_control_file(self):
        self.control_file = {
            key: self.control_file[key] for key in sorted(self.control_file.keys())
        }
        self.control_file["system_datasets"] = {
            key: self.control_file["system_datasets"][key]
            for key in sorted(self.control_file["system_datasets"].keys())
        }
        write_json_file(
            json_dict=self.control_file,
            file_path=self.dataset_dir.parent / "control" / "dataset.json",
        )

    def convert_dataset(self, to_extxyz: bool = False, to_set000: bool = False):
        """Convert the data ensembles of the dataset from extxyz to set.000 format or vice versa"""
        if to_extxyz and self.data_format == "set.000":
            for dataset in self.training_dataset.values():
                dataset.to_extxyz(save=True)
            for dataset in self.validation_dataset.values():
                dataset.to_extxyz(save=True)
        elif to_set000 and self.data_format == "extxyz":
            for dataset in self.training_dataset.values():
                dataset.to_set000()
            for dataset in self.validation_dataset.values():
                dataset.to_set000()
        else:
            raise ValueError(
                "The provided conversion is not valid. Please use to_extxyz=True or to_set000=True."
            )

    def prepare_for_mace_train(self, data_path: Path):
        """Data should be converted into the extxyz format and put together in a single file"""
        combined_frames = []
        for dataset in self.training_dataset.values():
            dataset_frames = dataset.to_extxyz(save=False)
            arcann_logger.debug(
                f"Loading dataset {dataset.path} for MACE training preparation: {len(dataset_frames)} frames"
            )
            combined_frames.extend(dataset_frames)
        arcann_logger.debug(
            f"Number of frames in the combined training dataset: {len(combined_frames)}"
        )
        with (data_path / "training_dataset.extxyz").open("w+") as f:
            write_extxyz(f, combined_frames)

        valid_frames, test_frames = [], []
        for dataset in self.validation_dataset.values():
            dataset_frames = dataset.to_extxyz(save=False)
            arcann_logger.debug(
                f"Loading dataset {dataset.path} for MACE validation/test preparation: {len(dataset_frames)} frames"
            )
            valid_frames.extend(dataset_frames[: len(dataset_frames) // 2])
            test_frames.extend(dataset_frames[len(dataset_frames) // 2 :])
            arcann_logger.debug(
                f"Number of frames in the combined validation dataset: {len(valid_frames)}"
            )
            arcann_logger.debug(
                f"Number of frames in the combined test dataset: {len(test_frames)}"
            )
        with (data_path / "validation_dataset.extxyz").open("w+") as f:
            write_extxyz(f, valid_frames)
        with (data_path / "test_dataset.extxyz").open("w+") as f:
            write_extxyz(f, test_frames)

    def prepare_for_deepmd_train(self):
        pass

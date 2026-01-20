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
import numpy as np
import os
from pathlib import Path
from typing import Dict, Literal, Union

from arcann_training.common.filesystem import check_directory, check_file_existence
from arcann_training.common.json import load_json_file
from arcann_training.initialization.utils import check_typeraw_properties

arcann_logger = logging.getLogger("ArcaNN")

class DataEnsemble():
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
        data_type (Literal["extxyz", "set.000"]): Type of data ensemble
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
        step: Literal["initial", "extra", "system_auto", "system_adhoc", "system_disturbed"],
        training_type: Literal["training", "validation"],
        system_name: str | None,
        iteration: int | None,
        data_type: Literal["extxyz", "set.000"],
        properties: Dict[int, Dict[str, str | float]],
    ):
        check_directory(Path(path), abort_on_error=True, error_msg="The provided data path does not exist.")
        self.path = Path(path)
        self.step = step
        self.training_type = training_type
        self.system_name = system_name
        self.iteration = iteration

        self.data_type = data_type
        self.properties = properties

    def to_dict(self):
        return {
            "path": str(self.path),
            "step": self.step,
            "training_type": self.training_type,
            "system_name": self.system_name,
            "iteration": self.iteration,
            "data_type": self.data_type,
            "properties": self.properties,
        }

    def check_format(self):
        raise NotImplementedError
    
    def get_size(self):
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError
    
    def get_extxyz(self):
        raise NotImplementedError
    
    def get_set000(self):
        raise NotImplementedError


class Set000Ensemble(DataEnsemble):
    """Define a data ensemble in the set.000 format"""
    def __init__(self, path, step, training_type, system_name, iteration, data_type, properties):
        super().__init__(path, step, training_type, system_name, iteration, data_type, properties)
        assert data_type == "set.000", "Data type must be 'set.000' for Set000Ensemble"

        self.check_format()
        self.size = self.get_size()

    def check_format(self):
        check_file_existence(self.path / "type.raw")
        # Check the type.raw file against the properties
        check_typeraw_properties(
            self.path / "type.raw", self.properties
        )
        set_path = self.path / "set.000"
        for data_type in ["box", "coord", "energy", "force"]:
            check_file_existence(set_path / (data_type + ".npy"))

    def get_size(self):
        return np.load(self.path / "set.000" / "box.npy").shape[0]


class ExtXYZEnsemble(DataEnsemble):
    """Define a data ensemble in the extxyz format"""
    pass


class Dataset():
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
        data_type (Literal["extxyz", "set.000"]): Type of data ensemble
        data_ensemble (DataEnsemble): Class of data ensemble
    
    Methods:
    --------
        get_training_dataset(): Get the training data ensembles
        get_validation_dataset(): Get the validation data ensembles
        init_data_type(): Initialize the data type and data ensemble class
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

        self.control_file = load_json_file(training_dir / "control" / "dataset.json", abort_on_error=False)
        self.config_file = config_file
        self.dataset_dir = Path(training_dir) / "data"
        check_directory(self.dataset_dir, abort_on_error=True, error_msg="The provided dataset directory does not exist.")
        self.init_data_type() #get data type and data ensemble class

    def __str__(self):
        pass

    def get_training_dataset(self):
        raise NotImplementedError

    def get_validation_dataset(self):
        raise NotImplementedError

    def init_data_type(self) -> None:
        """From the directory of the dataset, identify the extxyz or set.000 format"""        
        datasets = os.listdir(self.dataset_dir)
        for dataset in datasets:
            dataset_path = self.dataset_dir / dataset
            if dataset_path.is_dir():
                if (dataset_path / "set.000").is_dir():
                    self.data_type = "set.000"
                    self.data_ensemble = Set000Ensemble
                elif any(file.suffix == ".extxyz" for file in dataset_path.glob("*.extxyz")):
                    self.data_type = "extxyz"
                    self.data_ensemble = ExtXYZEnsemble
            

    def read_dataset(self) -> Union[int, int]:
        """Read the datasets from the already processed datasets in the control file"""

        self.training_dataset = {
            key: self.data_ensemble(**kwargs) for key, kwargs in self.config_file["used_datasets"]["training"].items() 
        }
        self.training_paths = list(self.training_dataset.keys())
        self.validation_dataset = {
            key: self.data_ensemble(**kwargs) for key, kwargs in self.config_file["used_datasets"]["validation"].items()
        }
        self.validation_paths = list(self.validation_dataset.keys())

    def load_dataset(self, extra_dataset: bool=True, init_dataset: bool=True, only_init: bool=False) -> Union[int, int, int]:
        """Load the new dataensembles from the dataset directory, depending on the control file information"""

        dataset_names = self.training_paths + self.validation_paths  #already existing/treated data ensembles
        common_kwargs = {
            "data_type": self.data_type,
            "properties": self.config_file["properties"],
        } #attributes common to all data ensembles

        extra_count, init_count, system_count = 0, 0, 0

        #Read the NEW data ensembles
        for datadir in self.dataset_dir.iterdir():
            if datadir.is_dir() and datadir.name not in dataset_names:
                step, system_name, iteration = None, None, None
                if datadir.name.startswith("extra_") and extra_dataset and not only_init:
                    #case of extra datasets
                    step = "extra"
                    extra_count += 1
                
                elif datadir.name.startswith("init_") and (init_dataset or only_init):
                    #case of initial datasets
                    step = "initial"
                    init_count += 1

                elif not only_init:
                    #case of system datasets
                    system_name, iteration = datadir.name.rsplit("_", 1)
                    iteration = int(iteration)
                    if system_name in self.config_file["systems_auto"]:
                        step = "system_auto"
                    elif "-disturbed" in system_name and system_name.removesuffix("-disturbed") in self.config_file["systems_adhoc"]:
                        step = "system_disturbed"
                    else:
                        step = "system_adhoc"
                    system_count += 1    

                if step:
                    if "valid" in datadir.name:
                        self.validation_dataset[datadir.name] = self.data_ensemble(
                            path=datadir, step=step, training_type="validation", system_name=system_name, iteration=iteration, **common_kwargs
                        )
                    else:
                        self.training_dataset[datadir.name] = self.data_ensemble(
                            path=datadir, step=step, training_type="training", system_name=system_name, iteration=iteration, **common_kwargs
                        )

        #Write the new data ensembles to the control file
        self.control_file["extra_datasets"] = {
            **{key: dataset.size for key, dataset in self.training_dataset.items() if dataset.step == "extra"},
            **{key: dataset.size for key, dataset in self.validation_dataset.items() if dataset.step == "extra"},
        }
        self.control_file["adhoc_datasets"] = {
            **{key: dataset.size for key, dataset in self.training_dataset.items() if dataset.step == "system_adhoc"},
            **{key: dataset.size for key, dataset in self.validation_dataset.items() if dataset.step == "system_adhoc"},
        }
        #TODO that won't work!! we want to differenciate the new ones to the old ones
        self.control_file["training"] |= {key: dataset.to_dict() for key, dataset in self.training_dataset.items()}
        self.control_file["validation"] |= {key: dataset.to_dict() for key, dataset in self.validation_dataset.items()}

        return extra_count, system_count

    def update_datasets(self):
        raise NotImplementedError
        
    def update_config_file(self):
        raise NotImplementedError

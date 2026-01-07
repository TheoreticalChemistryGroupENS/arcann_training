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
from typing import Dict, Literal

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
        data_type (Literal["extxyz", "set.000"]): Type of data ensemble
        properties (Dict[int, Dict[str, str | float]]): Properties associated with the data ensemble
    
    Methods:
    --------
        check_format(): Check the format of the data ensemble
        get_size(): Get the size of the data ensemble (ie. nb of confs)
        load_data(): Load the data of data ensemble
        get_extxyz(): Get the data under the extxyz format
        get_set000(): Get the data under the set.000 format
    """
    def __init__(
        self,
        path: str | Path,
        data_type: Literal["extxyz", "set.000"],
        properties: Dict[int, Dict[str, str | float]],
    ):
        check_directory(Path(path), abort_on_error=True, error_msg="The provided data path does not exist.")
        self.path = Path(path)
        self.data_type = data_type
        self.properties = properties

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
        control_file (Dict): Control file containing information about the data ensembles
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
        init_control_file(): Initialize the control file
        update_control_file(): Update the control file
    """
    def __init__(
        self,
        dataset_dir,
        control_file_path,
        main_json,
    ):
        # Attributes
        self.training_dataset: Dict[str, DataEnsemble] = {}
        self.validation_dataset: Dict[str, DataEnsemble] = {}
        self.training_paths = {}
        self.validation_paths = {}

        self.control_file = load_json_file(control_file_path)
        check_directory(Path(dataset_dir), abort_on_error=True, error_msg="The provided dataset directory does not exist.")
        self.dataset_dir = Path(dataset_dir)
        
        # Populate data type and data ensemble class
        self.init_data_type()
        self.init_datasets(main_json)
        self.init_control_file()

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
            

    def init_datasets(self, main_json) -> None:
        initial_datasets_paths = [_ for _ in (self.dataset_dir).glob("init_*")]
        if len(initial_datasets_paths) == 0:
            raise FileNotFoundError(f"No initial datasets found in the provided dataset directory: {self.dataset_dir}")
        
        training_dataset_paths = [path for path in initial_datasets_paths if "init_valid" not in path.name]
        self.training_dataset = {
            path.name: self.data_ensemble(path, self.data_type, main_json["properties"]) for path in training_dataset_paths
        }
        self.training_paths = list(self.training_dataset.keys())
        valid_dataset_paths = [path for path in initial_datasets_paths if "init_valid" in path.name]
        self.validation_dataset = {
            path.name: self.data_ensemble(path, self.data_type, main_json["properties"]) for path in valid_dataset_paths
        }
        self.validation_paths = list(self.validation_dataset.keys())

    def update_datasets(self):
        raise NotImplementedError
    
    def init_control_file(self):
        self.control_file = {
            **{key: dataset.size for key, dataset in self.training_dataset.items()},
            **{key: dataset.size for key, dataset in self.validation_dataset.items()},
        }
        return self.control_file
    
    def update_control_file(self):
        raise NotImplementedError


class Set000Ensemble(DataEnsemble):
    """Define a data ensemble in the set.000 format"""
    def __init__(self, path, data_type, properties):
        super().__init__(path, data_type, properties)
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
"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2026/08/31
Last modified: 2026/08/31

Test cases for the (labeling) utils module, focused on the PolarMACE-specific
'systems_charge' / 'systems_total_spin' per-system fields.

Classes
-------
TestGenerateInputLabelingJsonPolarMace():
    Test case for the 'generate_input_labeling_json' function's handling of
    'systems_charge' and 'systems_total_spin'.
TestGetSystemLabelingPolarMace():
    Test case for the 'get_system_labeling' function's handling of
    'systems_charge' and 'systems_total_spin'.
"""

# Standard library modules
import unittest

# Local imports
from arcann_training.labeling.utils import (
    generate_input_labeling_json,
    get_system_labeling,
)

DEFAULT_INPUT_JSON = {
    "labeling_program": "cp2k",
    "walltime_first_job_h": [0.5],
    "walltime_second_job_h": [1.0],
    "nb_nodes": [1],
    "nb_mpi_per_node": [10],
    "nb_threads_per_mpi": [1],
    "systems_charge": [0],
    "systems_total_spin": [1],
}


class TestGenerateInputLabelingJsonPolarMace(unittest.TestCase):
    """
    Test case for the 'generate_input_labeling_json' function's handling of
    'systems_charge' and 'systems_total_spin'.

    Methods
    -------
    test_default_broadcast_to_all_systems():
        Tests that the default charge/total_spin values are broadcast to every system.
    test_user_provided_per_system_values():
        Tests that user-provided per-system charge/total_spin lists are used as-is.
    test_size_mismatch_raises():
        Tests that a list of the wrong length raises a ValueError.
    """

    def setUp(self):
        self.main_json = {"systems_auto": {"system1": {}, "system2": {}}}

    def test_default_broadcast_to_all_systems(self):
        merged_input_json = generate_input_labeling_json(
            user_input_json={},
            previous_json={},
            default_input_json=DEFAULT_INPUT_JSON,
            merged_input_json={},
            main_json=self.main_json,
        )
        self.assertEqual(merged_input_json["systems_charge"], [0, 0])
        self.assertEqual(merged_input_json["systems_total_spin"], [1, 1])

    def test_user_provided_per_system_values(self):
        merged_input_json = generate_input_labeling_json(
            user_input_json={"systems_charge": [-2, 1], "systems_total_spin": [1, 3]},
            previous_json={},
            default_input_json=DEFAULT_INPUT_JSON,
            merged_input_json={},
            main_json=self.main_json,
        )
        self.assertEqual(merged_input_json["systems_charge"], [-2, 1])
        self.assertEqual(merged_input_json["systems_total_spin"], [1, 3])

    def test_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
            generate_input_labeling_json(
                user_input_json={"systems_charge": [-2]},
                previous_json={},
                default_input_json=DEFAULT_INPUT_JSON,
                merged_input_json={},
                main_json=self.main_json,
            )


class TestGetSystemLabelingPolarMace(unittest.TestCase):
    """
    Test case for the 'get_system_labeling' function's handling of
    'systems_charge' and 'systems_total_spin'.

    Methods
    -------
    test_charge_and_total_spin_are_returned_per_system():
        Tests that the correct per-system charge/total_spin are returned by index.
    """

    def test_charge_and_total_spin_are_returned_per_system(self):
        merged_input_json = {
            "labeling_program": "cp2k",
            "walltime_first_job_h": [0.5, 0.5],
            "walltime_second_job_h": [1.0, 1.0],
            "nb_nodes": [1, 2],
            "nb_mpi_per_node": [10, 20],
            "nb_threads_per_mpi": [1, 1],
            "systems_charge": [-2, 1],
            "systems_total_spin": [1, 3],
        }
        (
            _labeling_program,
            _walltime_first_job_h,
            _walltime_second_job_h,
            _nb_nodes,
            _nb_mpi_per_node,
            _nb_threads_per_mpi,
            charge,
            total_spin,
        ) = get_system_labeling(merged_input_json, system_auto_index=0)
        self.assertEqual(charge, -2)
        self.assertEqual(total_spin, 1)

        (*_rest, charge, total_spin) = get_system_labeling(
            merged_input_json, system_auto_index=1
        )
        self.assertEqual(charge, 1)
        self.assertEqual(total_spin, 3)


if __name__ == "__main__":
    unittest.main()

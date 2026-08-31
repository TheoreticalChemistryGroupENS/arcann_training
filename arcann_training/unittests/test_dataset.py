"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2026/08/31
Last modified: 2026/08/31

Test cases for the PolarMACE-specific fields (charge, total_spin, external_field,
elec_temp, REF_charges) of the (common) dataset module.

Class
-----
TestSet000EnsemblePolarMace():
    Test case for the Set000Ensemble PolarMACE fields round trip.
TestExtXYZEnsemblePolarMace():
    Test case for the ExtXYZEnsemble PolarMACE fields round trip.
TestCheckPolarMaceFormat():
    Test case for the 'check_polar_mace_format' method.
TestAddSystemDatasetPolarMace():
    Test case for the 'add_system_dataset' PolarMACE fields splitting.
"""

# Standard library modules
import tempfile
import unittest
from pathlib import Path

# Third-party modules
import numpy as np

# Local imports
from arcann_training.common.dataset import Dataset, ExtXYZEnsemble, Set000Ensemble

PROPERTIES = {1: {"symbol": "H", "mass": 1.008}, 2: {"symbol": "O", "mass": 15.999}}


def _make_frames(nframes: int = 3, natoms: int = 2):
    type_ = np.array([0, 1] * (natoms // 2) if natoms % 2 == 0 else [0] * natoms)
    energy = np.linspace(-10.0, -9.0, nframes)
    coord = np.random.rand(nframes, natoms * 3)
    box = np.tile(np.eye(3).reshape(-1), (nframes, 1)) * 10.0
    force = np.random.rand(nframes, natoms * 3)
    charge = np.resize(np.array([0, 1, -1]), nframes)
    total_spin = np.resize(np.array([1, 2, 1]), nframes)
    external_field = np.random.rand(nframes, 3)
    elec_temp = np.resize(np.array([300.0, 400.0, 500.0]), nframes)
    ref_charges = np.random.rand(nframes, natoms)
    return {
        "type": type_,
        "energy": energy,
        "coord": coord,
        "box": box,
        "force": force,
        "charge": charge,
        "total_spin": total_spin,
        "external_field": external_field,
        "elec_temp": elec_temp,
        "ref_charges": ref_charges,
    }


class TestSet000EnsemblePolarMace(unittest.TestCase):
    """
    Test case for the Set000Ensemble PolarMACE fields round trip.

    Methods
    -------
    test_write_load_roundtrip():
        Tests that PolarMACE fields survive a write()/load() round trip.
    test_to_extxyz_contains_polar_mace_fields():
        Tests that to_extxyz() sets the mandatory and optional PolarMACE fields.
    test_optional_fields_absent():
        Tests that ensembles without PolarMACE data load with None fields.
    """

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp_dir.name)
        self.frames = _make_frames()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _new_ensemble(self):
        return Set000Ensemble(
            path=self.path,
            step="initial",
            training_type="training",
            system_name=None,
            iteration=None,
            data_format="set.000",
            properties=PROPERTIES,
        )

    def test_write_load_roundtrip(self):
        ensemble = self._new_ensemble()
        ensemble.load_from_raw_arrays(
            type=self.frames["type"],
            energy=self.frames["energy"],
            coord=self.frames["coord"],
            box=self.frames["box"],
            force=self.frames["force"],
            virial=None,
            wannier=None,
            wannier_not_cvg=[],
            is_periodic=True,
            charge=self.frames["charge"],
            total_spin=self.frames["total_spin"],
            external_field=self.frames["external_field"],
            elec_temp=self.frames["elec_temp"],
            ref_charges=self.frames["ref_charges"],
        )
        ensemble.write()

        reloaded = self._new_ensemble()
        reloaded.load()
        np.testing.assert_array_equal(reloaded.charge, self.frames["charge"])
        np.testing.assert_array_equal(reloaded.total_spin, self.frames["total_spin"])
        np.testing.assert_allclose(
            reloaded.external_field, self.frames["external_field"]
        )
        np.testing.assert_allclose(reloaded.elec_temp, self.frames["elec_temp"])
        np.testing.assert_allclose(reloaded.ref_charges, self.frames["ref_charges"])

    def test_to_extxyz_contains_polar_mace_fields(self):
        ensemble = self._new_ensemble()
        ensemble.load_from_raw_arrays(
            type=self.frames["type"],
            energy=self.frames["energy"],
            coord=self.frames["coord"],
            box=self.frames["box"],
            force=self.frames["force"],
            virial=None,
            wannier=None,
            wannier_not_cvg=[],
            is_periodic=True,
            charge=self.frames["charge"],
            total_spin=self.frames["total_spin"],
            external_field=self.frames["external_field"],
            elec_temp=self.frames["elec_temp"],
            ref_charges=self.frames["ref_charges"],
        )
        ensemble.write()

        frames = self._new_ensemble().to_extxyz(save=False)
        for i, frame in enumerate(frames):
            self.assertEqual(frame.info["charge"], int(self.frames["charge"][i]))
            self.assertEqual(
                frame.info["total_spin"], int(self.frames["total_spin"][i])
            )
            np.testing.assert_allclose(
                frame.info["external_field"], self.frames["external_field"][i]
            )
            self.assertAlmostEqual(frame.info["elec_temp"], self.frames["elec_temp"][i])
            np.testing.assert_allclose(
                frame.arrays["REF_charges"], self.frames["ref_charges"][i]
            )

    def test_optional_fields_absent(self):
        ensemble = self._new_ensemble()
        ensemble.load_from_raw_arrays(
            type=self.frames["type"],
            energy=self.frames["energy"],
            coord=self.frames["coord"],
            box=self.frames["box"],
            force=self.frames["force"],
            virial=None,
            wannier=None,
            wannier_not_cvg=[],
            is_periodic=True,
        )
        ensemble.write()

        reloaded = self._new_ensemble()
        reloaded.load()
        self.assertIsNone(reloaded.charge)
        self.assertIsNone(reloaded.total_spin)
        self.assertIsNone(reloaded.external_field)
        self.assertIsNone(reloaded.elec_temp)
        self.assertIsNone(reloaded.ref_charges)


class TestExtXYZEnsemblePolarMace(unittest.TestCase):
    """
    Test case for the ExtXYZEnsemble PolarMACE fields round trip.

    Methods
    -------
    test_write_load_roundtrip():
        Tests that PolarMACE fields survive a write()/load() round trip via extxyz.
    """

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp_dir.name)
        self.frames = _make_frames()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _new_ensemble(self):
        return ExtXYZEnsemble(
            path=self.path,
            step="initial",
            training_type="training",
            system_name=None,
            iteration=None,
            data_format="extxyz",
            properties=PROPERTIES,
        )

    def test_write_load_roundtrip(self):
        ensemble = self._new_ensemble()
        ensemble.load_from_raw_arrays(
            type=self.frames["type"],
            energy=self.frames["energy"],
            coord=self.frames["coord"],
            box=self.frames["box"],
            force=self.frames["force"],
            virial=None,
            wannier=None,
            wannier_not_cvg=[],
            is_periodic=True,
            charge=self.frames["charge"],
            total_spin=self.frames["total_spin"],
            external_field=self.frames["external_field"],
            elec_temp=self.frames["elec_temp"],
            ref_charges=self.frames["ref_charges"],
        )
        ensemble.write()

        reloaded = self._new_ensemble()
        reloaded.load()
        np.testing.assert_array_equal(reloaded.charge, self.frames["charge"])
        np.testing.assert_array_equal(reloaded.total_spin, self.frames["total_spin"])
        # extxyz round-trips floats through limited-precision text, so allow for
        # that (rather than exact equality) instead of the tight default tolerance
        np.testing.assert_allclose(
            reloaded.external_field, self.frames["external_field"], atol=1e-6
        )
        np.testing.assert_allclose(
            reloaded.elec_temp, self.frames["elec_temp"], atol=1e-6
        )
        np.testing.assert_allclose(
            reloaded.ref_charges, self.frames["ref_charges"], atol=1e-6
        )


class TestCheckPolarMaceFormat(unittest.TestCase):
    """
    Test case for the 'check_polar_mace_format' method.

    Methods
    -------
    test_raises_when_missing():
        Tests that a ValueError is raised when a mandatory field is missing.
    test_passes_when_present():
        Tests that no error is raised when all mandatory fields are present.
    """

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp_dir.name)
        self.frames = _make_frames()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _new_ensemble(self):
        return Set000Ensemble(
            path=self.path,
            step="initial",
            training_type="training",
            system_name=None,
            iteration=None,
            data_format="set.000",
            properties=PROPERTIES,
        )

    def test_raises_when_missing(self):
        ensemble = self._new_ensemble()
        ensemble.load_from_raw_arrays(
            type=self.frames["type"],
            energy=self.frames["energy"],
            coord=self.frames["coord"],
            box=self.frames["box"],
            force=self.frames["force"],
            virial=None,
            wannier=None,
            wannier_not_cvg=[],
            is_periodic=True,
            charge=self.frames["charge"],
            total_spin=self.frames["total_spin"],
            # external_field intentionally omitted
        )
        with self.assertRaises(ValueError) as cm:
            ensemble.check_polar_mace_format()
        self.assertIn("external_field", str(cm.exception))

    def test_passes_when_present(self):
        ensemble = self._new_ensemble()
        ensemble.load_from_raw_arrays(
            type=self.frames["type"],
            energy=self.frames["energy"],
            coord=self.frames["coord"],
            box=self.frames["box"],
            force=self.frames["force"],
            virial=None,
            wannier=None,
            wannier_not_cvg=[],
            is_periodic=True,
            charge=self.frames["charge"],
            total_spin=self.frames["total_spin"],
            external_field=self.frames["external_field"],
        )
        ensemble.check_polar_mace_format()  # should not raise


class TestAddSystemDatasetPolarMace(unittest.TestCase):
    """
    Test case for the 'add_system_dataset' PolarMACE fields splitting.

    Methods
    -------
    test_polar_mace_fields_are_split():
        Tests that charge/total_spin/external_field/elec_temp/REF_charges are
        split between the training and validation ensembles.
    """

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.training_dir = Path(self.tmp_dir.name)
        (self.training_dir / "control").mkdir()
        (self.training_dir / "data").mkdir()
        self.config_file = {
            "validation/training_split": 0.5,
            "data_format": "set.000",
            "properties": PROPERTIES,
        }

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_polar_mace_fields_are_split(self):
        dataset = Dataset(training_dir=self.training_dir, config_file=self.config_file)
        frames = _make_frames(nframes=4)
        dataset.add_system_dataset(
            step="system_auto",
            system_name="test_system",
            iteration="0",
            type=frames["type"],
            energy=frames["energy"],
            coord=frames["coord"],
            box=frames["box"],
            force=frames["force"],
            virial=None,
            wannier=None,
            wannier_not_cvg=[],
            is_periodic=True,
            charge=frames["charge"],
            total_spin=frames["total_spin"],
            external_field=frames["external_field"],
            elec_temp=frames["elec_temp"],
            ref_charges=frames["ref_charges"],
        )

        train_dir = self.training_dir / "data" / "test_system_0"
        valid_dir = self.training_dir / "data" / "test_system_valid_0"
        train_ensemble = Set000Ensemble(
            path=train_dir,
            step="system_auto",
            training_type="training",
            system_name="test_system",
            iteration=0,
            data_format="set.000",
            properties=PROPERTIES,
        )
        valid_ensemble = Set000Ensemble(
            path=valid_dir,
            step="system_auto",
            training_type="validation",
            system_name="test_system",
            iteration=0,
            data_format="set.000",
            properties=PROPERTIES,
        )
        train_ensemble.load()
        valid_ensemble.load()

        self.assertEqual(
            train_ensemble.charge.shape[0] + valid_ensemble.charge.shape[0], 4
        )
        self.assertEqual(
            train_ensemble.external_field.shape[0]
            + valid_ensemble.external_field.shape[0],
            4,
        )
        combined_charges = np.concatenate(
            [train_ensemble.charge, valid_ensemble.charge]
        )
        np.testing.assert_array_equal(
            np.sort(combined_charges), np.sort(frames["charge"])
        )


if __name__ == "__main__":
    unittest.main()

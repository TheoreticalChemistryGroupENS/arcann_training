# Initialization

Now that you have decided the subsystems that you want to train your NNP on and prepared all the required files you can initialize the ArcaNN procedure by running (from the $WORK_DIR folder):

```bash
python -m arcann_training initialization start
```

Now it should have generated your first `000-training` directory. In `$WORK_DIR` you will also find a `default_input.json` file that looks like this:

```JSON
{
    "step_name": "initialization",
    "systems_auto": ["SYSNAME1", "SYSNAME2", "SYSNAME3"],
    "nnp_count": 3,
    "nnp_program": "deepmd",
    "validation/training_split": 0.2,
    "data_format": "set.000"
}
```

The keywords have the following meaning:

- `"systems_auto"` contains the names of all the systems that were found in your `$WORK_DIR/user_files/` directory (i.e. all `LMP` files). If you list systems explicitly in your own `input.json` instead of relying on auto-detection, each name must have a matching `NAME.lmp` file in `user_files/`, or initialization aborts.
- `"nnp_count"` is the number of neural network potentials used in the committee (3 by default).
- `"nnp_program"` selects the neural network potential architecture for the **whole** procedure. It can be `"deepmd"` (the default, using DeePMD-kit) or `"mace"` (using MACE). This choice is made **once**, here at initialization, and every later step reads it automatically — you do not repeat it in the training, exploration, labeling or test inputs.
- `"data_format"` sets the format used to store your training data. Use `"set.000"` (the binary `.npy` format read by DeePMD-kit) when `nnp_program` is `"deepmd"`, and `"extxyz"` (extended XYZ text files, read by MACE) when `nnp_program` is `"mace"`. If you leave the default, make sure it matches your chosen program.
- `"validation/training_split"` is the fraction of your labeled data set aside for validation rather than training (0.2 means 20% validation, 80% training). This is used to monitor the quality of the model during training.

**Important — choosing MACE:** if you set `"nnp_program"` to `"mace"`, you must provide a MACE configuration file named `mace_MACEVERSION.yml` (or `.yaml`) in your `user_files/` folder — for example `mace_0.3.14.yml` — in the same way that DeePMD-kit requires a `dptrain_VERSION.json` file. If this file is missing, initialization will stop and tell you so. Note that the parity is only partial: for DeePMD-kit, ArcaNN also cross-checks the `type_map` order in `dptrain_VERSION.json` against `properties.txt`, but it does **not** perform an equivalent check for the MACE yml — a wrong element order there will not be caught at initialization. See [Iterative procedure prerequisites](iter_prerequisites.md) for how to prepare these files.

**Also required:** a `properties.txt` file in `user_files/` (see [Iterative procedure prerequisites](iter_prerequisites.md)) — initialization aborts if it is missing. It is also cross-checked against every `LMP` file: an `LMP` file with an atom type absent from `properties.txt`, or with a mass that disagrees with it by more than `0.01`, will abort initialization (a placeholder mass of exactly `0.1` in the `LMP` file is treated as "unset" and skipped from this check).

The initialization will create several folders. The most important one is the `control/` folder, in which essential data files will be stored throughout the iterative procedure. These files will be written in `.json` format and should NOT be modified. Right after initialization, `control/` contains `config.json` (your initialization choices, such as subsystem names and options) and `dataset.json` (bookkeeping for your training datasets). Finally the `000-training` empty folder should also have been created by the execution of the python script, where you will perform the first iteration of [training](training.md).

If at this point you want to modify the datasets used for the first training you simply have to create an `input.json` from the `default_input.json` file and remove or add the system names to the list. You could also change the number of NNP if you wish. Then you only have to execute the command of the initialization phase again and your `000-training` directory will be updated.

## Switching architectures: the `transition` phase

The `initialization` step has a second, optional phase called `transition`. Its purpose is to **convert your existing datasets between the file formats used by the two architectures**, so you can switch `nnp_program` without regenerating your data.

The two programs store data differently: DeePMD-kit uses the `set.000` (`.npy`) format, while MACE uses the `extxyz` format. If your `data/` folder already contains data sets in one format (for example from a previous DeePMD-kit campaign) and you now want to train with the other program, run:

```bash
python -m arcann_training initialization transition
```

ArcaNN will detect the format currently present in your `data/` folder and convert every data set to the format required by the `nnp_program` you have selected. Your data folder must contain only one format at a time; if both are present, the phase will stop and ask you to keep only one. You would typically run this phase right after changing `nnp_program` (and `data_format`) in your initialization input.

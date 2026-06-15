#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# Created: 2026/02/25
# Last modified: 2026/02/25
#----------------------------------------------
# You must keep the _R_VARIABLES_ in the file.
# You must keep the name file as job_deepmd_train_ARCHTYPE_myHPCkeyword.sh.
#----------------------------------------------
# Project/Account
#SBATCH --account=_R_PROJECT_@_R_ALLOC_
# QoS/Partition/SubPartition
#SBATCH --qos=_R_QOS_
#SBATCH --partition=_R_PARTITION_
#SBATCH -C _R_SUBPARTITION_
# Number of Nodes/MPIperNodes/OpenMPperMPI/GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 10
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
# Walltime
#SBATCH -t _R_WALLTIME_
# Merge Output/Error
#SBATCH -o MACE_Train.%j
#SBATCH -e MACE_Train.%j
# Name of job
#SBATCH -J MACE_Train
# Email
#SBATCH --mail-type FAIL,BEGIN,END,ALL
#SBATCH --mail-user _R_EMAIL_
#

#----------------------------------------------
# Files / Variables - They should not be changed
#----------------------------------------------

MACE_MODEL_VERSION="_R_MACE_VERSION_"
MACE_FONDATION_FILE="_R_MACE_FONDATION_FILE_" 
MACE_IN_FILE="_R_MACE_INPUT_FILE_"
MACE_LOG_FILE="_R_MACE_LOG_FILE_"
MACE_OUT_FILE="_R_MACE_OUTPUT_FILE_"
MACE_DATA_DIR="../data"

MACE_CONDA_INSTALL="" #If you don't want to use a specific version of MACE, but rather your own you installed on a conda env, specify the path of this env here

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }

# Check
[ -f "${MACE_IN_FILE}" ] || { echo "${MACE_IN_FILE} does not exist. Aborting..."; exit 1; }

# This part copies the data from the MACE_DATA_DIR to the job folder (because they are one up and they should be in the same folder)
[ -d ${MACE_DATA_DIR} ] || { echo "${MACE_DATA_DIR} does not exist. Aborting..."; exit 1; }
mkdir -p "${SLURM_SUBMIT_DIR}"/data || { echo "Could not create ${SLURM_SUBMIT_DIR}/data. Aborting..."; exit 1; }
{ cp -r ${MACE_DATA_DIR}/* "${SLURM_SUBMIT_DIR}"/data && echo "${MACE_DATA_DIR} copied successfully"; } || { echo "Could not copy ${MACE_DATA_DIR}. Aborting..."; exit 1; }

# This part copies the MACE_FONDATION_FILE to the job folder if it exists
if [ -f ${MACE_FONDATION_FILE} ]; then
    { ln -s "$(realpath "${MACE_FONDATION_FILE}")" "${SLURM_SUBMIT_DIR}" && echo "${MACE_FONDATION_FILE} linked successfully"; } || { echo "Could not link ${MACE_FONDATION_FILE}. Aborting..."; exit 1; }
else
    echo "${MACE_FONDATION_FILE} does not exist. Skipping copy."
fi

# Example to use the DeepMD_MODEL_VERSION variable
if [ ${MACE_MODEL_VERSION} == "0.3.14" ]; then
    # Load the MACE module
    module load mace
elif [ -n "$MACE_CONDA_INSTALL" ]; then
    # Activate the conda environment
    module load conda
    source ${CONDA_PREFIX}/bin/activate
    conda activate ${MACE_CONDA_INSTALL}
else
    echo "MACE version ${MACE_MODEL_VERSION} is not available. Aborting..."
    exit 1
fi

# Run the MACE train
echo "# [$(date)] Running MACE train..."
mace_run_train --config=${MACE_IN_FILE} 1> ${MACE_LOG_FILE} 2> ${MACE_OUT_FILE} 
echo "# [$(date)] MACE train finished."

# This are useless files, so we remove them
if [ -f out.json ]; then rm out.json; fi
if [ -f input_v2_compat.json ]; then rm input_v2_compat.json; fi

sleep 2
exit
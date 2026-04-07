#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2026 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# Created: 2026/03/27
# Last modified: 2026/03/27
# Project/Account
#SBATCH --account=_R_PROJECT_@_R_ALLOC_
# QoS/Partition/SubPartition
#SBATCH --qos=_R_QOS_
#SBATCH --partition=_R_PARTITION_
#SBATCH -C _R_SUBPARTITION_
# Number of Nodes/MPIperNodes/OpenMPperMPI/GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 12
#SBATCH --hint=nomultithread
# Walltime
#SBATCH -t _R_WALLTIME_
# Merge Output/Error
#SBATCH -o MACE_COMPRESS.%A_%a
#SBATCH -e MACE_COMPRESS.%A_%a
# Name of job
#SBATCH -J MACE_COMPRESS
# Email
#SBATCH --mail-type FAIL,BEGIN,END,ALL
#SBATCH --mail-user _R_EMAIL_


#----------------------------------------------
# Files / Variables - They should not be changed
#----------------------------------------------

MACE_MODEL_VERSION="_R_MACE_VERSION_"
MACE_MODEL_FILE="_R_MACE_MODEL_FILE_"
MACE_MODEL_STYLE="_R_MACE_MODEL_STYLE_"

#----------------------------------------------
# Adapt the following lines to your HPC system
#----------------------------------------------

cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }

# Check
[ -f ${MACE_MODEL_FILE} ] || { echo "${MACE_MODEL_FILE} does not exist. Aborting..."; exit 1; }

# Example to use the MACE_MODEL_VERSION variable
if [ "${MACE_MODEL_VERSION}" == "0.3.14" ]; then
    module purge
    module load mace
elif [ "${MACE_MODEL_VERSION}" == "0.3.15" ] ; then
    module purge
    module load mace/${MACE_MODEL_VERSION}
else
    echo "MACE ${MACE_MODEL_VERSION} is not installed on ${SLURM_JOB_QOS}. Aborting..."; exit 1
fi

if [ "${MACE_MODEL_STYLE}" == "symmetrix" ]; then
    module load symmetrix
    echo "# [$(date)] Converting model to CPU for symmetrix"
    mace_convert_device "${MACE_MODEL_FILE}" --target_device cpu --output_file "${MACE_MODEL_FILE}.cpu"
    echo "# [$(date)] Converting mace model to symmetrix"
    symmetrix_extract_mace --model "${MACE_MODEL_FILE}.cpu" --atomic-numbers _R_ATOMIC_NUMBERS_ --output "${MACE_MODEL_FILE}.json"
elif [ "${MACE_MODEL_STYLE}" == "mace" ]; then
    echo "# [$(date)] Converting mace model to LAMMPS libtorch"
    mace_create_lammps_model "${MACE_MODEL_FILE}" --format libtorch
elif [ "${MACE_MODEL_STYLE}" == "mliap" ]; then
    # cuequivariance is required to convert to mliap

    echo "# [$(date)] Converting mace model to LAMMPS MLIAP"
    mace_create_lammps_model "${MACE_MODEL_FILE}" --format mliap
fi

sleep 2
exit

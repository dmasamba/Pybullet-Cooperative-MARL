#!/bin/bash
### Sets the job's name.
#SBATCH --job-name=15dgr_cylinder

### Sets the job's output file and path.
#SBATCH --output=Training/clusterResults/clusterOutputs/successOutputs/output.out.%j

### Sets the job's error output file and path.
#SBTACH --error=Training/clusterResults/clusterOutputs/errorOutputs/errors.err.%j

### Requested number of nodes for this job. Can be a single number or a range.
#SBATCH -N 1

### Requested partition (group of nodes, i.e. compute, bigmem, gpu, etc.) for the resource allocation.
#SBATCH -p sxmq

### Requested number of gpus
#SBATCH --gres=gpu:2

### Limit on the total run time of the job allocation.
#SBATCH --time=100:00:00

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo "Activating TensorFlow-2.6.2 environment"
source /home/tmorgan01/anaconda3/etc/profile.d/conda.sh
conda activate dan

echo "Running clusterTrain.py"
cd /home/tmorgan01/SeniorProjectOfficial/Pybullet-Cooperative-MARL/pushBox/
python3 /home/tmorgan01/SeniorProjectOfficial/Pybullet-Cooperative-MARL/pushBox/clusterTrain.py

echo "Deactivating TensorFlow-2.6.2 environment"
deactivate

echo "Done."

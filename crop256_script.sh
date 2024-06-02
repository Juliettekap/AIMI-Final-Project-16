#!/bin/sh

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1

# Load modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Copy data to scratch storage
echo "Copying data to $TMPDIR"
rm -r $TMPDIR/input
mkdir $TMPDIR/input
cp -rv $HOME/input/images $TMPDIR/input/images
cp -rv $HOME/input/picai_labels/ $TMPDIR/input/picai_labels
# cp -r $HOME/input/crop256/preprocessed/ $TMPDIR/input/
# cp -r $HOME/input/crop256/overviews/ $TMPDIR/input/

# Fix scratch storage paths
# python3 $HOME/commands/replace_path_in_overviews.py $TMPDIR/input/overviews /scratch-local/tfrederick.6456163 $TMPDIR

# Set up environment
python3 -m pip install --user --upgrade pip
python3 -m pip install --user scikit-build
python3 -m pip install --user -r $HOME/repos/picai_baseline/src/picai_baseline/unet/requirements.txt
python3 -m pip install --user picai_baseline==0.8.5

python3 $HOME/repos/picai_baseline/src/picai_baseline/prepare_data_semi_supervised.py \
       --imagesdir=$TMPDIR/input/images \
       --workdir=$TMPDIR/input/crop256/preprocessed \
       --labelsdir=$TMPDIR/input/picai_labels \
       --spacing 3.0 0.5 0.5 \
       --matrix_size 20 256 256

# Create overviews
python3 $HOME/repos/picai_baseline/src/picai_baseline/unet/plan_overview.py \
       --task=Task2203_picai_baseline \
       --workdir=$TMPDIR/input \
       --preprocessed_data_path=$TMPDIR/input/crop256/preprocessed/nnUNet_raw_data/{task} \
       --overviews_path=$TMPDIR/input/crop256/overviews/


mkdir $HOME/input/crop256
mkdir $HOME/input/crop256/preprocessed
mkdir $HOME/input/crop256/overviews

cp -rv $TMPDIR/input/crop256/preprocessed $HOME/input/crop256/preprocessed/
cp -rv $TMPDIR/input/crop256/overviews $HOME/input/crop256/overviews/

# Run training script
python3 $HOME/repos/picai_baseline/src/picai_baseline/unet/train.py \
        --weights_dir=$HOME/models/picai/unet/baseline \
        --overviews_dir=$TMPDIR/input/crop256/overviews/ \
        --folds 0 1 2 3 4 \
        --num_epochs 20 \
        --validate_n_epochs 1 \
        --validate_min_epoch 0 \
        --focal_loss_gamma 0 \
        --image_shape 20 256 256

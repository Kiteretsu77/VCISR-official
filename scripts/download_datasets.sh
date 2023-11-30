#!/bin/bash

# Download Needed datasets
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip DIV2K_train_HR.zip

# Clean
rm DIV2K_train_HR.zip

# Move
mkdir datasets/
mv DIV2K_train_HR/ datasets/

# Crop images with overlap
python scripts/crop_images.py -i datasets/DIV2K_train_HR --crop_size 360 -o datasets/train_hr --output_usm datasets/train_hr_usm

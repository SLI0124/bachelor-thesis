#!/bin/bash

# check if wget is installed
if ! [ -x "$(command -v wget)" ]; then
  echo 'Error: wget is not installed. Please install wget.' >&2
  exit 1
fi

# check if unzip is installed
if ! [ -x "$(command -v unzip)" ]; then
  echo 'Error: unzip is not installed. Please install unzip.' >&2
  exit 1
fi

# check if tar is installed
if ! [ -x "$(command -v tar)" ]; then
  echo 'Error: tar is not installed. Please install tar.' >&2
  exit 1
fi

# make data/datasets directory, if it doesn't exist
if [ ! -d "data/datasets" ]; 
then
  mkdir -p data/datasets
  echo "data/datasets directory created"
else
  echo "data/datasets directory already exists"
fi

# make data/splits directory, if it doesn't exist
if [ ! -d "data/splits" ];
then
  mkdir -p data/splits
  echo "data/splits directory created"
else
  echo "data/splits directory already exists"
fi

# make data/full_images directory, if it doesn't exist
if [ ! -d "data/full_images" ];
then
  mkdir -p data/full_images
  echo "data/full_images directory created"
else
  echo "data/full_images directory already exists"
fi

# download CNRPark-EXT full images, unzip it and delete the zip file and clean the directory
# if you have zip files already downloaded, put it in the data/full_images directory and it will not download it again
if [ ! -f "data/full_images/CNR-EXT_FULL_IMAGE_1000x750.tar" ];
then
  wget -P data/full_images https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNR-EXT_FULL_IMAGE_1000x750.tar
else
  echo "data/full_images/CNR-EXT_FULL_IMAGE_1000x750.tar already exists"
fi

echo "Extracting CNRPark-EXT full images..."
mkdir -p data/full_images/cnr
tar -xf data/full_images/CNR-EXT_FULL_IMAGE_1000x750.tar -C data/full_images/cnr
rm data/full_images/CNR-EXT_FULL_IMAGE_1000x750.tar

# make data/datasets/cnr directory, if it doesn't exist and proceed to download the datasets
if [ ! -d "data/datasets/cnr" ];
then
  mkdir -p data/datasets/cnr
  echo "data/datasets/cnr directory created"

  # download the CNR-EXT dataset, unzip it and delete the zip file and clean the directory
  # if you have zip files already downloaded, put it in the data/datasets directory and it will not download it again
  # but it will delete the zip file and clean the directory
  if [ ! -f "data/datasets/CNR-EXT-Patches-150x150.zip" ];
  then
    wget -P data/datasets https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNR-EXT-Patches-150x150.zip
  else
    echo "data/datasets/CNR-EXT-Patches-150x150.zip already exists"
  fi

  echo "Cleaning CNR-EXT dataset..."
  unzip -q data/datasets/CNR-EXT-Patches-150x150.zip -d data/datasets/cnr/CNR-EXT-Patches-150x150
  rm data/datasets/CNR-EXT-Patches-150x150.zip

  # download the CNR dataset, unzip it and delete the zip file and clean the directory
  # if you have zip files already downloaded, put it in the data/datasets directory and it will not download it again
  # but it will delete the zip file and clean the directory
  if [ ! -f "data/datasets/CNRPark-Patches-150x150.zip" ];
  then
    wget -P data/datasets https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNRPark-Patches-150x150.zip
  else
    echo "data/datasets/CNRPark-Patches-150x150.zip already exists"
  fi

  echo "Cleaning CNR dataset..."
  unzip -q data/datasets/CNRPark-Patches-150x150.zip -d data/datasets/cnr/CNRPark-Patches-150x150
  rm data/datasets/CNRPark-Patches-150x150.zip
else
  echo "data/datasets/cnr directory already exists"
fi

# make data/datasets/pklot directory, if it doesn't exist and proceed to download the dataset
if [ ! -d "data/datasets/pklot" ];
then
  mkdir -p data/datasets/pklot
  echo "data/datasets/pklot directory created"

  # download the PKLot dataset, unzip it, delete the zip file and clean the directory
  # if you have zip files already downloaded, put it in the data/datasets directory and it will not download it again
  # but it will delete the zip file and clean the directory
  if [ ! -f "data/datasets/PKLot.tar.gz" ];
  then
    wget -P data/datasets http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz
  else
    echo "data/datasets/PKLot.tar.gz already exists"
  fi

  echo "Extracting PKLot dataset..."
  tar -xzf data/datasets/PKLot.tar.gz -C data/datasets/pklot

  echo "Cleaning PKLot dataset..."
  mkdir -p data/full_images/pklot
  mv data/datasets/pklot/PKLot/PKLot/* data/full_images/pklot
  mv data/datasets/pklot/PKLot/PKLotSegmented/ data/datasets/pklot/
  rm -rf data/datasets/pklot/PKLot
  rm data/datasets/PKLot.tar.gz
else
  echo "data/datasets/pklot directory already exists"
fi

# make data/datasets/spkl directory, if it doesn't exist and proceed to download the dataset
if [ ! -d "data/datasets/spkl" ];
then
  mkdir -p data/datasets/spkl
  echo "data/datasets/spkl directory created"

  # download the SPKLv2 dataset, unzip it and delete the zip file and clean the directory
  # if you have zip files already downloaded, put it in the data/datasets directory and it will not download it again
  # but it will delete the zip file and clean the directory
  if [ ! -f "data/datasets/SPKLv2_test.zip" ];
  then
    wget -P data/datasets https://sc.link/1KZu
  else
    echo "data/datasets/SPKLv2_test.zip already exists"
  fi

  echo "Extracting SPKLv2 dataset..."
  unzip -q data/datasets/SPKLv2_test.zip -d data/datasets/spkl

  echo "Cleaning SPKLv2 dataset..."
  mkdr data/full_images/spkl
  mv data/datasets/spkl/SPKLv2/SPKLv2/patch_markup/ data/datasets/spkl/
  mv data/datasets/spkl/SPKLv2/SPKLv2/images/ data/full_images/spkl
  mv data/datasets/spkl/SPKLv2/SPKLv2/int_markup/ data/full_images/spkl
  rm -rf data/datasets/spkl/SPKLv2
  rm data/datasets/spkl/patch_markup/*.json
  rm data/datasets/SPKLv2_test.zip
else
  echo "data/datasets/spkl directory already exists"
fi

# make data/datasets/acpds directory, if it doesn't exist and proceed to download the dataset
if [ ! -d "data/datasets/acpds" ];
then
  mkdir -p data/datasets/acpds
  echo "data/datasets/acpds directory created"

  # download the ACPDS dataset, unzip it and delete the zip file and clean the directory
  # if you have zip files already downloaded, put it in the data/datasets directory and it will not download it again
  # but it will delete the zip file and clean the directory
  if [ ! -f "data/datasets/ACPDS.zip" ];
  then
    wget -P data/datasets https://sc.link/1KZq
  else
    echo "data/datasets/ACPDS.zip already exists"
  fi

  echo "Extracting ACPDS dataset..."
  unzip -q data/datasets/ACPDS.zip -d data/datasets/acpds

  echo "Cleaning ACPDS dataset..."
  mv data/datasets/acpds/ACPDS/ACPDS/patch_markup/ data/datasets/acpds/
  mkdir -p data/full_images/acpds
  mv data/datasets/acpds/ACPDS/ACPDS/images/ data/full_images/acpds
  mv data/datasets/acpds/ACPDS/ACPDS/int_markup/ data/full_images/acpds
  rm -rf data/datasets/acpds/ACPDS/
  rm data/datasets/acpds/patch_markup/*.json
  rm data/datasets/ACPDS.zip
else
  echo "data/datasets/acpds directory already exists"
fi

echo "All datasets have been downloaded and cleaned successfully!"

# echo "Now I will create text file with their splits for training and testing. Basic splits are 80% for training
#  and 20% for testing and 50% for training and 50% for testing. If you'd like to change the splits, please do so in the
#   create_splits.py file."
cd creating_split_files/ || exit

python3 create_split_files.py
python3 create_training_splits.py --split_ratio 80
python3 create_training_splits.py --split_ratio 50

echo "All splits have been created successfully!"

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
  rm -rf data/datasets/pklot/PKLot/PKLot
  mv data/datasets/pklot/PKLot/PKLotSegmented/ data/datasets/pklot/
  rm -rf data/datasets/pklot/PKLot
  rm data/datasets/PKLot.tar.gz
else
  echo "data/datasets/pklot directory already exists"
fi

# make data/datasets/acmps directory, if it doesn't exist and proceed to download the dataset
if [ ! -d "data/datasets/acmps" ];
then
  mkdir -p data/datasets/acmps
  echo "data/datasets/acmps directory created"

  # download the ACMPS dataset, unzip it and delete the zip file and clean the directory
  # if you have zip files already downloaded, put it in the data/datasets directory and it will not download it again
  # but it will delete the zip file and clean the directory
  if [ ! -f "data/datasets/ACMPS.zip" ];
  then
    wget -P data/datasets https://sc.link/1KZv
  else
    echo "data/datasets/ACMPS.zip already exists"
  fi

  echo "Extracting ACMPS dataset..."
  unzip -q data/datasets/ACMPS.zip -d data/datasets/acmps

  echo "Cleaning ACMPS dataset..."
  mv data/datasets/acmps/ACMPS/ACMPS/patch_markup/ data/datasets/acmps/
  rm -rf data/datasets/acmps/ACMPS/
  rm data/datasets/ACMPS.zip
else
  echo "data/datasets/acmps directory already exists"
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
  mv data/datasets/spkl/SPKLv2/SPKLv2/patch_markup/ data/datasets/spkl/
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

# python3 utils/create_split_files.py

#!/bin/bash

# delete existing data/datasets directory for fresh download
if [ -d "data/datasets" ]; then
  rm -r data/datasets
  echo "data/datasets directory deleted"
fi

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

# make data/datasets directory, if it doesn't exist
if [ ! -d "data/datasets" ]; then
  mkdir -p data/datasets
  echo "data/datasets directory created"
fi

# make data/datasets/pklot directory, if it doesn't exist
if [ ! -d "data/datasets/pklot" ]; then
  mkdir -p data/datasets/pklot
  echo "data/datasets/pklot directory created"
fi

# make data/datasets/cnr directory, if it doesn't exist
if [ ! -d "data/datasets/cnr" ]; then
  mkdir -p data/datasets/cnr
  echo "data/datasets/cnr directory created"
fi

# download the PKLot dataset, unzip it, delete the zip file and move directory out of subdirectory
wget -P data/datasets http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz
tar -xvzf data/datasets/PKLot.tar.gz -C data/datasets/pklot
rm data/datasets/PKLot.tar.gz
mv data/datasets/pklot/PKLot/* data/datasets/pklot

# download the CNR-EXT dataset, unzip it and delete the zip file
wget -P data/datasets https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNR-EXT-Patches-150x150.zip
unzip data/datasets/CNR-EXT-Patches-150x150.zip -d data/datasets/cnr/CNR-EXT-Patches-150x150
rm data/datasets/CNR-EXT-Patches-150x150.zip


# download the CNR dataset, unzip it and delete the zip file
wget -P data/datasets https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNRPark-Patches-150x150.zip
unzip data/datasets/CNRPark-Patches-150x150.zip -d data/datasets/cnr/CNRPark-Patches-150x150
rm data/datasets/CNRPark-Patches-150x150.zip

# download the CNR dataset splits, unzip them and delete the zip files
wget -P data/datasets https://github.com/fabiocarrara/deep-parking/releases/download/archive/splits.zip
unzip data/datasets/splits.zip -d data/datasets/cnr/splits
rm data/datasets/splits.zip

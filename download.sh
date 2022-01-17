#!/bin/bash

DATASET_URL="https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/download"

if [ ! -d "$PWD/data" ]; then
	mkdir data
fi

pushd data

echo "Downloading dataset ..."
wget $DATASET_URL

echo "Extracting archive ..."
unzip archive.zip

echo "done ..."

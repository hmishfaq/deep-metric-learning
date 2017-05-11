#!/usr/bin/env bash

mkdir -p datasets

cd datasets

echo "Downloading cars dataset ..."
wget -O cars_train.tgz http://imagenet.stanford.edu/internal/car196/cars_train.tgz
tar -xzf cars_train.tgz
wget -O cars_test.tgz http://imagenet.stanford.edu/internal/car196/cars_test.tgz
tar -xzf cars_test.tgz

echo "Downloading birds dataset ..."
wget -O birds.tgz http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -xzf birds.tgz

cd ..

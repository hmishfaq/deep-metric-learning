#!/usr/bin/env bash

mkdir -p datasets

cd datasets

echo "Downloading cars dataset ..."
# train
if [ ! -f cars_train.tgz ]
then
	wget -O cars_train.tgz http://imagenet.stanford.edu/internal/car196/cars_train.tgz
fi
mkdir -p cars_train
tar -xzf cars_train.tgz -C cars_train --strip-components=1
# test
if [ ! -f cars_test.tgz ]
then
	wget -O cars_test.tgz http://imagenet.stanford.edu/internal/car196/cars_test.tgz
fi
mkdir -p cars_test
tar -xzf cars_test.tgz -C cars_test --strip-components=1
# labels
if [ ! -f cars_labels.tgz ]
then
	wget -O cars_labels.tgz http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
fi
mkdir -p cars_labels
tar -xzf cars_labels.tgz -C cars_labels --strip-components=1

echo "Downloading birds dataset ..."
if [ ! -f birds.tgz ]
then
	wget -O birds.tgz http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
fi
mkdir -p birds
tar -xzf birds.tgz -C birds --strip-components=1

cd ..

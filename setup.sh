#!/bin/bash
echo "Downloading Recipe1M+ dataset locally"
mkdir dataset/recipe
cd dataset/recipe
wget http://wednesday.csail.mit.edu/temporal/release/recipes_with_nutritional_info.json
wget http://wednesday.csail.mit.edu/temporal/release/det_ingrs.json
wget http://wednesday.csail.mit.edu/temporal/release/recipe1M_layers.tar.gz
tar -xvzf recipe1M_layers.tar.gz
rm layer2.json
rm recipe1M_layers.tar.gz
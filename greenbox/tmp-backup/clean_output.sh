#!/bin/bash

rm -rf clean_output/
cp -r output/ clean_output/
for file in clean_output/*;
do
    sed -e "s/Total NR Energy//g"  -i $file
    sed -e "s/R Energy//g"  -i $file
    sed -e "s/Total//g"  -i $file
    sed -e "s/Carbon//g"  -i $file
    sed -e "s/footprint//g"  -i $file
    sed -e "s/\[//g"  -i $file
    sed -e "s/]//g"  -i $file
    sed -e "s/nr carbon, r carbon, total carbon//g"  -i $file
    sed -e "s/://g"  -i $file
    sed -e "s/://g"  -i $file
    sed -e "s/^.\{1\}//g" -i $file
    sed -e "s/ //g" -i $file
done
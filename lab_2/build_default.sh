#!/bin/bash
dir="build_default"
# phi_file_name="phi_default.dat"
# rho_file_name="rho_default.dat"

# x=500
# y=500

rm -Rf $dir
mkdir $dir
cd $dir

gcc ../mainDefault.c -lm -O0
time ./a.out

# gnuplot -c ../script.gpi $phi_file_name $phi_file_name $x $y
# gnuplot -c ../script.gpi $rho_file_name $rho_file_name $x $y
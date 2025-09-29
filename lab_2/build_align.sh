#!/bin/bash
dir="build_align"
phi_file_name="phi_align.dat"
rho_file_name="rho_align.dat"

x=500
y=500

rm -Rf $dir
mkdir $dir
cd $dir

gcc ../main128align.c -lm -march=native
time ./a.out

gnuplot -c ../script.gpi $phi_file_name $phi_file_name $x $y
gnuplot -c ../script.gpi $rho_file_name $rho_file_name $x $y
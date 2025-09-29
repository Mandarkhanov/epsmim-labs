#!/bin/bash
dir="build_omp_simd"

rm -Rf $dir
mkdir $dir
cd $dir

gcc ../mainOmpSimd.c -lm -fopenmp-simd -Ofast
time ./a.out
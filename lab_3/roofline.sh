#!/bin/bash
dir_name="build_1"
out_name="1line.out"

rm -rf $dir_name
mkdir $dir_name
cp $out_name $dir_name

advixe-cl --collect survey --project-dir ./$dir_name --search-dir src:r= ./$dir_name/$out_name
# advixe-cl --collect roofline --project-dir ./$dir_name --search-dir src:r=./ -- ./$dir_name/$out_name
advixe-cl --report roofline --project-dir ./$dir_name --report-output ./$dir_name/report.html --format xml

cd ..
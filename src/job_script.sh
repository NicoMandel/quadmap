#!/bin/bash -l

# These are the limits for the hpc script
memory_limit="5000mb"
time_limit="18:00:00"
header_1="#!/bin/bash -l"
header_2="#PBS -l walltime=${time_limit}"
header_3="#PBS -l mem=${memory_limit}"
header_4="#PBS -l ncpus=1"

# Body for the hpc script
# conda and directory
body_1="conda activate robostackenv"
body_2="cd ~/quadmap/src"

# General stuff for the script
# declare -a StringArray=("sim_hyb-10" "sim_hyb-20")
search_dir=~/rosbag/pcl
for m in "$search_dir"/*.bag 
do
	fname=$(basename "$m" .bag)
	this_script_file="run_sim_${fname}.sh"
	# HPC required header
	echo ${header_1} >> ${this_script_file}
	echo "#PBS -N ${fname}" >> ${this_script_file}
	echo ${header_2} >> ${this_script_file}
	echo ${header_3} >> ${this_script_file}
	echo ${header_4} >> ${this_script_file}
	
	# body - loading conda and directory
	echo ${body_1} >> ${this_script_file}
	echo ${body_2} >> ${this_script_file}
	
	# Actual command to execute
	echo "python bag-test.py --file ${fname} --output ../output --input ~/rosbag/pcl " >> ${this_script_file}
	qsub ${this_script_file}
	
	#cleanup
	rm ${this_script_file}
done


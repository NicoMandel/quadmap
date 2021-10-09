#!/bin/bash -l

# These are the limits for the hpc script
memory_limit="6000mb"
time_limit="18:00:00"
header_1="#!/bin/bash -l"
header_2="#PBS -l walltime=${time_limit}"
header_3="#PBS -l mem=${memory_limit}"
header_4="#PBS -l ncpus=1"

# Body for the hpc script
# conda and directory
body_1="conda activate robostackenv"
body_2="cd ~/quadmap/src"
py_file="load_multi_tree.py"

# General stuff for the script
declare -a StringArray=("sim_hyb-10" "sim_hyb-20" "sim_tgt1-ascend" "sim_tgt1-descend" "sim_tgt2-ascend" "sim_tgt2-descend" "sim_mission-10m" "sim_mission-20m" \
"exp_tgt1-ascend" "exp_tgt1-descend" "exp_tgt2-ascend" "exp_tgt2-descend" "exp_hyb-freq-20m" "exp_mission-20m")
numlist=(14 12 10 8 6 4 1)

for e in "${StringArray[@]}" 
do
	for d in ${numlist[@]}
	do
		this_script_file="plot_${e}_${d}.sh"
		# HPC required header
		echo ${header_1} >> ${this_script_file}
		echo "#PBS -N plot_${e}_${d}" >> ${this_script_file}
		echo ${header_2} >> ${this_script_file}
		echo ${header_3} >> ${this_script_file}
		echo ${header_4} >> ${this_script_file}
		
		# body - loading conda and directory
		echo ${body_1} >> ${this_script_file}
		echo ${body_2} >> ${this_script_file}
		
		# Actual command to execute
		commandstring="python ${py_file} --input ../output/skips --file ${e} -d ${d} -s"

		echo ${commandstring} >> ${this_script_file}
		echo "running: ${commandstring}"
		qsub ${this_script_file}
		
		#cleanup
		rm ${this_script_file}
	done
done


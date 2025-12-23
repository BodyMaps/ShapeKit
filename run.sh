export inputs="/path/to/your/input/folder"
export outputs="/path/to/your/output/folder"
export CPU_NUM=16

python -W ignore main.py --input_folder $inputs --output_folder $outputs --cpu_count $CPU_NUM --continue_prediction
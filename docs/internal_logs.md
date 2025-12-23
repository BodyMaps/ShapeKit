```bash
rm -rf ShapeKit/
git clone https://github.com/BodyMaps/ShapeKit.git
cd ShapeKit

export INPUT="/mnt/bodymaps/release/PanTSMini_1220/mask_only/PanTSMini/PanTSMini/"
export OUTPUT="/mnt/bodymaps/temp_data_to_delete_very_soon/zzhou82/PanTSMini_2025_1220/"
export CPU_NUM=62
export LOG="logs/PanTSMini_2025_1220"

python -W ignore main.py --input_folder $INPUT --output_folder $OUTPUT --cpu_count $CPU_NUM --log_folder $LOG --continue_prediction
```
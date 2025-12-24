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

```bash
export INPUT="/mnt/bodymaps/mask_only/AbdomenAtlas3.0Mini/AbdomenAtlas3.0Mini/"
export OUTPUT="/mnt/bodymaps/temp_data_to_delete_very_soon/zzhou82/AbdomenAtlas3.0Mini_2025_1224/"
export CPU_NUM=62
export LOG="logs/AbdomenAtlas3.0Mini_2025_1224"

python -W ignore main.py --input_folder $INPUT --output_folder $OUTPUT --cpu_count $CPU_NUM --log_folder $LOG --continue_prediction
```

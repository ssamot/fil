import seaborn as sns

ALPHA=0.3
FIGX=6
FIGY=4.5
CONFIDENCE=0.95
LINEWIDTH=1.0

base_pallete = sns.color_palette()

COLOR_PYTORCH=base_pallete[0]
COLOR_LGBM=base_pallete[1]
COLOR_SPLINES=base_pallete[2]
COLOR_DT=base_pallete[3]
COLOR_RANDOM=base_pallete[5]

TEXT_PYTORCH="DeepCFR"
TEXT_LGBM="LUGL-DeepCFR-LightGBM"
TEXT_SPLINES="LUGL-DeepCFR-Multi-S"
TEXT_DT="LUGL-DeepCFR-Multi-D"
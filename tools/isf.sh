# INPUTFILE="/home/debian/water/TIP4P/2005/shear/246_equil/dump_H2O_246.lammpstrj"
INPUTFILE="/home/debian/water/TIP4P/2005/traj_2.5e-5_246.lammpstrj"
OUTSTR="246"
DT_FS="500"
KTARGET="2.0"
DK="0.15"

echo "Begining analyzing..."
echo "First begin with isf"

TMAX_PS="1000"
STRIDE="0.5"

# python3 isf_analysis.py --dumps $INPUTFILE --dt_fs $DT_FS --k_target $KTARGET --tmax_ps $TMAX_PS --origin_stride_ps $STRIDE --logtime --no_com_removal --unwrapped
python3 /home/debian/water/tools/isf.py \
  --dumps $INPUTFILE \
  --dt_fs $DT_FS \
  --k_target $KTARGET \
  --dk $DK \
  --tmax_ps $TMAX_PS \
  --origin_stride_ps $STRIDE \
  --logtime \
  --mode self \
  --no_com_removal \
  --unwrapped \
  --shear_rate 2.5e-5 \
#   --max_origins 500 \
#   --chunk_size 200 \
#   --memory_limit_gb 16.0 \
#   --progress_interval 500
# python3 isf.py \
#   --dumps $INPUTFILE \
#   --dt_fs $DT_FS \
#   --k_target $KTARGET \
#   --dk $DK \
#   --tmax_ps $TMAX_PS \
#   --origin_stride_ps $STRIDE \
#   --logtime \
#   --n_log 150 \
#   --mode self \
#   --out_prefix water_220K \
#   --no_com_removal \
#   --unwrapped \
#   --max_origins 500 \
#   --chunk_size 200 \
#   --memory_limit_gb 16.0 \
#   --progress_interval 500
python3 /home/debian/water/tools/isf_visualize.py
TRAJ="/home/debian/water/TIP5P/Tanaka_2018/rst/mini_dump.xyz"
A="1.0"
STRIDE="5"
DT="0.1"
START_FRAME="5000"
END_FRAME="6000"
OUT="/home/debian/water/TIP5P/Tanaka_2018/rst/analyze/chi4_short"

python3 new_chi.py --traj $TRAJ --a $A --stride $STRIDE --dt $DT --start_frame $START_FRAME --end_frame $END_FRAME --out $OUT
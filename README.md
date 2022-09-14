export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/anthony/.mujoco/mujoco210/bin

# test
# remember to startx first to have a display

DISPLAY=:3 python3 train.py --dataset=/home/anthony/robomimic/datasets/lift/ph/low_dim.hdf5
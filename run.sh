# roboturk

# convert to robomimic dataset format
# do it for each of the datasets
python robomimic/scripts/conversion/convert_roboturk_pilot.py --folder /home/anthony/RoboTurkPilot/bins-Cereal
python robomimic/scripts/conversion/convert_roboturk_pilot.py --folder /home/anthony/RoboTurkPilot/bins-Can
python robomimic/scripts/conversion/convert_roboturk_pilot.py --folder /home/anthony/RoboTurkPilot/bins-Milk
python robomimic/scripts/conversion/convert_roboturk_pilot.py --folder /home/anthony/RoboTurkPilot/bins-Bread

# get the obs key from the dataset states
python robomimic/scripts/dataset_states_to_obs.py --dataset /home/anthony/RoboTurkPilot/bins-Bread/demo.hdf5 --output_name low_dim.hdf5 --done_mode 2

# playback
python robomimic/scripts/playback_dataset.py \
    --dataset /home/anthony/RoboTurkPilot/bins-Bread/demo.hdf5 \
    --render_image_names agentview eye_in_hand \
    --video_path rollouts/obs_bins_Bread.mp4 \
    --n 1

# training 
CUDA_VISIBLE_DEVICES=0 DISPLAY=:0 python3 train.py \
    --dataset=/home/anthony/RoboTurkPilot/bins-Can/low_dim.hdf5 \
    --algo=dt \
    --name=bins-Can-BC

# rollouts
python robomimic/scripts/run_trained_agent.py --agent /home/anthony/robomimic/dt_trained_models/bins-Can-BC/20220918133848/models/model_epoch_5_SawyerPickPlaceCanTeleop_success_0.0.pth \
    --n_rollouts 5 \
    --horizon 400 \
    --seed 0 \
    --video_path rollouts/bins-Can-trained.mp4 \
    --camera_names agentview eye_in_hand 
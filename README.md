# Randomized Oscillators Network (RON)

This repo is a fork of the repo https://github.com/tk-rusch/coRNN, where the coRNN model was first introduced. We extend coRNN to the reservoir computing paradigm and we dub our model Randomized Oscillators Network (RON).

Each main file runs RON and all other models on a given dataset. 

``--no_friction`` (to use the hcoRNN model)

``--esn --no_friction`` (to use the RON model)

Example: to train and test on the Adiac classification tasks, for 5 consecutive times, just type:

CUDA_VISIBLE_DEVICES=0 python test_Adiac_task.py --n_hid 100 --epsilon 5 --gamma 3 --dt 0.01 --no_friction --esn --inp_scaling 10 --rho 9 --epsilon_range 1 --gamma_range 2 --use_test --test_trials 5 --batch 30

wait a handful of seconds and read the mean and std results from the corresponding .txt log file (Adiac_log_coESN.txt).

#!/usr/bin/env bash

echo "Running PPO on CartPole-v1"
nohup python experiments.py --env CartPole-v1 --device 0 --expert PPO > out_CartPole_PPO_CnnPolicy.txt 2>&1 &
sleep 10

echo "Running PPO on MountainCar-v0"
nohup python experiments.py --env MountainCar-v0 --device 1 --expert PPO > out_MaintainCar_PPO_CnnPolicy.txt 2>&1 &
sleep 10

echo "Running PPO on Pendulum-v0"
nohup python experiments.py --env Pendulum-v0 --device 0 --expert PPO > out_Pendulum_PPO_CnnPolicy.txt 2>&1 &
sleep 10

echo "Running PPO on LunarLander-v2"
nohup python experiments.py --env LunarLander-v2 --device 1 --expert PPO > out_LunarLander_PPO_CnnPolicy.txt 2>&1 &
sleep 10

echo "Finish all"
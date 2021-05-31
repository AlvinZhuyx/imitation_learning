import gym
import numpy as np
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)

from stable_baselines import PPO2, GAIL
from stable_baselines.gail import ExpertDataset, generate_expert_traj
from stable_baselines.common import make_vec_env

def parse_args():
  parser = argparse.ArgumentParser('Imitation Learning for ECE239')
  parser.add_argument('--env', type=str, default='LunarLander-v2', help='Name of environment')
  parser.add_argument('--expert', type=str, default='PPO', help='Expert algorithm, choose from {SAC or PPO}.')
  parser.add_argument('--policy_type', type=str, default='MlpPolicy')
  parser.add_argument('--device', type=str, default='3', help='Device to use')
  parser.add_argument('--train_log_dir', type=str, default='./log', help='Location for saving model, dataset and log')
  
  # training parameters for expert
  parser.add_argument('--times_expert', type=int, default=3, help='Number of times to train expert')
  parser.add_argument('--expert_training_step', type=int, default=int(1e6), help='Num of steps for training expert algorithm.')
  parser.add_argument('--n_env', type=int, default=1, help='Num of parallel environments to run')
  parser.add_argument('--n_steps', type=int, default=1024, help='The number of steps to run for each environment per update')
  parser.add_argument('--nminibatches', type=int, default=32, help='Number of training minibatches per update.')
  parser.add_argument('--noptepochs', type=int, default=4)
  parser.add_argument('--lam', type=float, default=0.98)
  parser.add_argument('--gamma', type=float, default=0.999)
  parser.add_argument('--ent_coef', type=float, default=0.01)
  parser.add_argument('--learning_rate', type=float, default=2.5e-4)
  parser.add_argument('--cliprange', type=float, default=0.2)

  # training parameters for student
  parser.add_argument('--expert_episodes', type=int, default=int(1000), help='Num of training data generated from expert')
  parser.add_argument('--student_training_step', type=int, default=int(1e6), help='Num of steps for training imitation algorithm(student)')
  
  args = parser.parse_args()
  return args



def evaluate(model, env, num_steps=1000):
  episode_rewards = [0.0]
  obs = env.reset()
  for i in range(num_steps):
      action, _states = model.predict(obs)
      obs, reward, done, info = env.step(action)
      episode_rewards[-1] += reward
      if done:
          obs = env.reset()
          episode_rewards.append(0.0)
  mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
  print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
  
  return mean_100ep_reward

def main(args):
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = args.device

  # train expert model for multiple times and save the best model
  best_reward = -np.inf
  train_env = make_vec_env(args.env, n_envs=args.n_env)
  eval_env = gym.make(args.env)
  '''
  for i in range(args.times_expert):
    train_env.reset()
    train_log_dir = os.path.join(args.train_log_dir, args.env + '_' + args.expert)
    if args.expert == 'PPO':
        expert_model = PPO2(args.policy_type, env=train_env, n_steps=args.n_steps, nminibatches=args.nminibatches, noptepochs=args.noptepochs, ent_coef=args.ent_coef,\
                        lam=args.lam, gamma=args.gamma, cliprange=args.cliprange, learning_rate=args.learning_rate, verbose=1, tensorboard_log=train_log_dir)
    else:
        raise NotImplementedError
    expert_model.learn(total_timesteps=args.expert_training_step)
    mean_reward = evaluate(expert_model, eval_env, num_steps=10000)
    if mean_reward > best_reward:
        best_reward = mean_reward
        expert_model.save(os.path.join(args.train_log_dir, args.env + '_expert'))
    del expert_model
  
  train_env.reset()
  '''
  train_log_dir = os.path.join(args.train_log_dir, args.env + '_' + args.expert)
  expert_model = PPO2.load(os.path.join(args.train_log_dir, args.env + '_expert'), env=train_env)
  generate_expert_traj(expert_model, os.path.join(train_log_dir, 'expert_traj'), n_timesteps=-1, n_episodes=args.expert_episodes)
  train_env.close()
  
  dataset = ExpertDataset(expert_path=os.path.join(train_log_dir, 'expert_traj.npz'), traj_limitation=-1)
  gail_model = GAIL(args.policy_type, args.env, dataset, verbose=1, tensorboard_log=train_log_dir)
  gail_model.learn(args.student_training_step)
  
  evaluate(gail_model, eval_env, num_steps=10000)
  gail_model.save(os.path.join(args.train_log_dir, args.env + '_GAIL'))
  eval_env.close()

if __name__=="__main__":
    args = parse_args()
    main(args)



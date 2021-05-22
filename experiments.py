import gym
import numpy as np
import argparse
import os

from stable_baselines import DQN, SAC, PPO1, GAIL
from stable_baselines.gail import ExpertDataset, generate_expert_traj

def parse_args():
  parser = argparse.ArgumentParser('Imitation Learning for ECE239')
  parser.add_argument('--env', type=str, default='CartPole-v1', help='Name of environment')
  parser.add_argument('--expert', type=str, default='PPO', help='Expert algorithm, choose from {SAC or PPO}.')
  parser.add_argument('--expert_training_step', type=int, default=int(1e6), help='Num of steps for training expert algorithm.')
  parser.add_argument('--expert_episodes', type=int, default=int(1e4), help='Num of training data generated from expert')
  parser.add_argument('--student_training_step', type=int, default=int(1e6), help='Num of steps for training imitation algorithm(student)')
  parser.add_argument('--policy_type', type=str, default='CnnPolicy')
  parser.add_argument('--device', type=str, default='0', help='Device to use')
  parser.add_argument('--train_log_dir', type=str, default='./log', help='Location for saving model, dataset and log')
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
  env = gym.make(args.env)
  train_log_dir = os.path.join(args.train_log_dir, args.env + '_' + args.expert + '_' + args.policy_type)
  if args.expert == 'SAC':
    expert_model = SAC(args.policy_type, env, verbose=1, tensorboard_log=train_log_dir)
  elif args.expert == 'PPO':
    expert_model = PPO1(args.policy_type, env, verbose=1, tensorboard_log=train_log_dir)
  elif args.expert == 'DQN':
    expert_model = DQN(args.policy_type, env, verbose=1, tensorboard_log=train_log_dir)
  else:
    raise NotImplementedError
  expert_model.learn(total_timesteps=args.expert_training_step)
  generate_expert_traj(expert_model, os.path.join(train_log_dir, 'expert_traj'), n_timesteps=-1, n_episodes=args.expert_episodes)

  dataset = ExpertDataset(expert_path=os.path.join(train_log_dir, 'expert_traj.npz'), traj_limitation=-1)
  gail_model = GAIL(args.policy_type, env, dataset, verbose=1, tensorboard_log=train_log_dir)
  gail_model.learn(args.student_training_step)
  evaluate(gail_model, env, num_steps=10000)
  gail_model.save(train_log_dir)
  env.close()

if __name__=="__main__":
    args = parse_args()
    main(args)
import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

from gym_torcs_multi import TorcsEnv

import multiprocessing as mp
from functools import partial

def job(rank, args, device, shared_model):
    torch.manual_seed(args.seed + rank)
    time.sleep(rank*4)
    episode_rewards = deque(maxlen=10)
    envs = TorcsEnv(rank, None, throttle=True, gear_change=False)
    
    actor_critic = Policy(
        24,
        3,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    actor_critic.load_state_dict(shared_model.state_dict())
    state_in = np.zeros(24)
    action_out = np.zeros(3)
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              state_in.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    obs = envs.reset()
    obs = torch.from_numpy(obs)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    acc_r = 0
    done = [False]

    for step in range(args.num_steps):
        if done[0]:
            episode_rewards.append(acc_r)
            obs = envs.reset()
            obs = torch.from_numpy(obs)
            rollouts.obs[0].copy_(obs)
            rollouts.to(device)
            acc_r = 0
            
        # Sample actions
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                rollouts.masks[step])

        # Obser reward and next obs
        target_action = action.numpy()[0]
        obs, reward, done, infos = envs.step(target_action)
        
        acc_r += reward
        obs = torch.from_numpy(obs).float().to(device)
        reward = torch.from_numpy(np.array([reward])).unsqueeze(dim=1).float()
        done = [done]

        masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor(
            [[1.0] for done_ in done])
        rollouts.insert(obs, recurrent_hidden_states, action,
                        action_log_prob, value, reward, masks, bad_masks)
    # print(rank, np.mean(episode_rewards),np.median(episode_rewards), 
                        # np.min(episode_rewards), np.max(episode_rewards))
    s = "{},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(rank, np.mean(episode_rewards),np.median(episode_rewards), 
                        np.min(episode_rewards), np.max(episode_rewards))
    print(s)
    with open("logs/{}_mp.csv".format(args.env_name), 'a') as fl:
        fl.write(s)
    return rollouts

def main():
    args = get_args()

    args.env_name = "Torcs-v1"

    args.algo = 'ppo'
    args.use_gae = True
    args.log_interval = 1 
    args.num_steps = 2048
    args.num_processes = 1
    args.lr = 3e-4
    args.entropy_coef = 0
    args.value_loss_coef = 0.5
    args.ppo_epoch = 10
    args.num_mini_batch = 32
    args.gamma = 0.99
    args.gae_lambda = 0.95
    args.num_env_steps = 1000000
    args.use_linear_lr_decay = True
    args.use_proper_time_limits = True
    args.save_dir = "saved"
    args.seed = 0
    args.cuda = False

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
    #                      args.gamma, args.log_dir, device, False)
    # envs = gym.make(args.env_name)
    # envs.seed(args.seed)

    actor_critic = Policy(
        24,
        3,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    # actor_critic = torch.load("saved/ppo/Torcs-v0_new_mp.pt")
    # print(actor_critic)

    
    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    # rollouts = RolloutStorage(args.num_steps, args.num_processes,
    #                           envs.observation_space.shape, envs.action_space,
    #                           actor_critic.recurrent_hidden_state_size)

    # obs = envs.reset()
    # obs = torch.from_numpy(obs)
    # rollouts.obs[0].copy_(obs)
    # rollouts.to(device)
    acc_r = 0

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    done = [False]
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        print(j,"update")
        os.system("pkill torcs")
        p_job = partial(job, args=args, device=device, shared_model=actor_critic)
        pool = mp.Pool()
        res = pool.map(p_job, range(12))
        pool.close()
        pool.join()

        for rollouts in res:
            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                    args.gae_lambda, args.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            # print(getattr(utils.get_vec_normalize(envs), 'ob_rms', None))
            torch.save([
                actor_critic
                #,getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + "_new_mp.pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            s = "{},{:.2f},{:.2f}\n".format(j, np.mean(episode_rewards), np.median(episode_rewards))
            with open("logs/{}_new_mp.csv".format(args.env_name), 'a') as fl:
                fl.write(s)
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

if __name__ == "__main__":
    main()

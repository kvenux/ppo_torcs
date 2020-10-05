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

def main():
    os.system("pkill torcs")
    args = get_args()

    args.env_name = "Torcs-v0"

    args.algo = 'ppo'
    args.use_gae = True
    args.log_interval = 1 
    args.num_steps = 1000
    # args.num_steps = 2048
    args.num_processes = 1
    args.lr = 3e-5
    # args.lr = 3e-4
    args.entropy_coef = 0
    args.value_loss_coef = 0.5
    args.ppo_epoch = 10
    args.num_mini_batch = 32
    args.gamma = 0.99
    args.gae_lambda = 0.95
    args.num_env_steps = 1000000
    args.use_linear_lr_decay = True
    args.use_proper_time_limits = True
    args.cuda = False

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = TorcsEnv(0, None, throttle=True, gear_change=False)
    # envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma, args.log_dir, device, False)

    print(envs.action_space)
    actor_critic = Policy(
        24,
        3,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    
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
    state_in = np.zeros(24)
    action_out = np.zeros(3)
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              state_in.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    obs = torch.from_numpy(obs)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    acc_r = 0
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            action_numpy = action.numpy()
            obs, reward, done, infos = envs.step(action_numpy[0])

            acc_r += reward

            obs = torch.from_numpy(obs)
            reward = torch.Tensor([[reward]])
            if done:
                episode_rewards.append(acc_r)
                envs.reset()
                acc_r = 0

            done = [done]

            # for info in infos:
            #     if 'episode' in info.keys():
            #         episode_rewards.append(info['episode']['r'])
            
            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[1.0]])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        # reset env to update policy
        envs.reset()

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

            torch.save([
                actor_critic
                # , getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            s = "{},{:.2f},{:.2f}\n".format(j, np.mean(episode_rewards), np.median(episode_rewards))
            with open("logs/{}.csv".format(args.env_name), 'a') as fl:
                fl.write(s)
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        # if (args.eval_interval is not None and len(episode_rewards) > 1
        #         and j % args.eval_interval == 0):
        #     ob_rms = utils.get_vec_normalize(envs).ob_rms
        #     evaluate(actor_critic, ob_rms, args.env_name, args.seed,
        #              args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()

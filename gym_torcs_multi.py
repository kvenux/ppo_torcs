import gym
from gym import spaces
import numpy as np
# from os import path
import snakeoil3 as snakeoil3
import numpy as np
import copy
import collections as col
import signal
import subprocess
import time
import os
from PIL import Image
import random


class TorcsEnv:
    terminal_judge_start = 100  # If after 100 timestep still no progress, terminated
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    initial_reset = True

    def __init__(self, env_number, lock, vision=False, throttle=False, gear_change=False):
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change

        self.initial_run = True
        self.cur_pic_index = 0

        ##print("launch torcs")
        # os.system('pkill torcs')
        self.port_number = 3101 + env_number
        self.env_number = env_number
        self.lock = lock
        self.pro = None
        self.cur_ep = 0
        if(self.pro != None):
            os.killpg(os.getpgid(self.pro.pid), signal.SIGTERM) 
        time.sleep(0.2)
        # if self.vision is True:
        #     os.system('torcs -nofuel -nodamage -nolaptime -vision &')
        # else:
        #     os.system('torcs -nofuel -nolaptime &')
        self.restart_window()
        # cmd = 'torcs -nofuel -nolaptime -p {} &'.format(self.port_number)
        # self.pro = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        
        # time.sleep(0.5)

        # xdo_cmd = "xdotool windowmove $(xdotool getactivewindow) 300 300"
        # os.system(xdo_cmd)
        # os.system('sh autostart.sh')
        # time.sleep(0.5)

        """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        """
        # if throttle is False:
        #     self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        # else:
        #     self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))

        if vision is False:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
            self.observation_space = spaces.Box(low=low, high=high)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
            self.observation_space = spaces.Box(low=low, high=high)

    def step(self, u):
    # def step(self, action_index):
       #print("Step")
        # convert thisAction to the actual torcs actionstr
        # u = self.map_action(action_index)
        ob = self.get_obs()
        # # target_speed = 0.50
       
        u = np.clip(u, -1.0, 1.0)
        # print(u)

        # u[1] = (u[1] + 1.0)/2.0
        target_speed = 0.80
        speed_p = 3.0
        if ob.speedX < target_speed:
            u[1] = (target_speed - ob.speedX) * speed_p
        else:
            u[1] = 0.1
        u[1] += u[1]/4.0
        if ob.speedX < 0.30:
            u[2] = 0.0            
        else:
            u[2] = (u[2] + 1.0)/8.0
        # u[2] = 0.0
        # print(u)

        client = self.client

        this_action = self.agent_to_torcs(u)


        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']
            action_torcs['brake'] = this_action['brake']

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1
            if self.throttle:
                if client.S.d['speedX'] > 50:
                    action_torcs['gear'] = 2
                if client.S.d['speedX'] > 80:
                    action_torcs['gear'] = 3
                if client.S.d['speedX'] > 110:
                    action_torcs['gear'] = 4
                if client.S.d['speedX'] > 140:
                    action_torcs['gear'] = 5
                if client.S.d['speedX'] > 170:
                    action_torcs['gear'] = 6
        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        try:
            client.respond_to_server()
            # Get the response of TORCS
            client.get_servers_input()
        except:
            ob = self.get_obs()
            s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ))
            reward = 0
            client.R.d['meta'] = True
            return s_t, reward, client.R.d['meta'], {}
            pass

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        trackPos = np.array(obs['trackPos'])
        sp = np.array(obs['speedX'])
        damage = np.array(obs['damage'])
        rpm = np.array(obs['rpm'])
        
        sin_speed_theta = 1.1
        track_theta = 1.3
        # track_theta = 0
        damage_theta = 10
        progress = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle']))*sin_speed_theta - sp * np.abs(obs['trackPos'])*track_theta
        # print("{:.5f} {:.5f} {:.5f}".format(sp*np.cos(obs['angle']) , np.abs(sp*np.sin(obs['angle'])) , sp * np.abs(obs['trackPos'])))
        # print(damage)
        reward = progress

        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            # print('collision!')
            # reward = -1
            reward -= (obs['damage'] - obs_pre['damage'])*damage_theta
            episode_terminate = True
            client.R.d['meta'] = True


        # Termination judgement #########################
        episode_terminate = False
        #if (abs(track.any()) > 1 or abs(trackPos) > 1):  # Episode is terminated if the car is out of track
        #    reward = -200
        #    episode_terminate = True
        #    client.R.d['meta'] = True

        #if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
        #    if progress < self.termination_limit_progress:
        #        print("No progress")
        #        episode_terminate = True
        #        client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            episode_terminate = True
            client.R.d['meta'] = True
        # print(obs['rpm'])
        if self.time_step > 50 and obs['rpm']/10000.0 <= 0.09426:
            # print(obs['rpm'])
            episode_terminate = True
            client.R.d['meta'] = True

        if self.time_step > 500:
            episode_terminate = True
            client.R.d['meta'] = True

        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        ob = self.get_obs()
        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ))

        return s_t, reward, client.R.d['meta'], {}

    def reset(self, relaunch=False):
        # print("Reset")
        if (self.cur_ep + 1) % 20 == 0:
            self.reset_torcs()

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        while True:
            try:
                self.client = snakeoil3.Client(self, p=self.port_number, vision=self.vision)  # Open new UDP in vtorcs
                break
            except:
                time.sleep(5)
                pass
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        ob = self.get_obs()
        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ))
        self.cur_ep += 1
        return s_t

    def end(self):
        print('pkill torcs')
        if self.pro != None:
            os.killpg(os.getpgid(self.pro.pid), signal.SIGTERM) 
        # os.system('pkill torcs')

    def get_obs(self):
        return self.observation

    def restart_window(self):
        # while os.environ["TORCS_RESTART"] == "1":
        #     print("env: {} is waiting for signal".format(self.env_number))
        #     time.sleep(10 + random.randint(1,10))
        # os.environ["TORCS_RESTART"] == "1"
        # self.lock.acquire()
        cmd = 'torcs -nofuel -nolaptime -p {} &'.format(self.port_number)
        self.pro = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        time.sleep(1.0)
        # self.env_number
        row = self.env_number % 4
        col = int(self.env_number / 4)
        x = 64 + 320 * row
        y = 30 + 280 * col
        # print(self.env_number, x, y)
        xdo_cmd = "xdotool windowmove $(xdotool getactivewindow) {} {}".format(x, y)
        os.system(xdo_cmd)
        os.system('sh autostart.sh')
        time.sleep(0.5)
        # self.lock.release()
        # os.environ["TORCS_RESTART"] == "0"

    def reset_torcs(self):
        # print("relaunch torcs")
        # os.system('pkill torcs')
        if self.pro != None:
            os.killpg(os.getpgid(self.pro.pid), signal.SIGTERM) 
        # time.sleep(0.2)
        # if self.vision is True:
        #     os.system('torcs -nofuel -nodamage -nolaptime -vision &')
        # else:
        #     os.system('torcs -nofuel -nolaptime &')
        # cmd = 'torcs -nofuel -nolaptime -p {} &'.format(self.port_number)
        # self.pro = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        # time.sleep(0.5)
        # os.system('sh autostart.sh')
        # time.sleep(0.5)
        self.restart_window()

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({'accel': u[1]})
            torcs_action.update({'brake': u[2]})

        if self.gear_change is True: # gear change action is enabled
            torcs_action.update({'gear': int(u[3])})

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        # print(len(image_vec))
        # r = image_vec[0:len(image_vec):3]
        # g = image_vec[1:len(image_vec):3]
        # b = image_vec[2:len(image_vec):3]

        # sz = (64, 64)
        # r = np.array(r).reshape(sz)
        # g = np.array(g).reshape(sz)
        # b = np.array(b).reshape(sz)
        data = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(64):
            for j in range(64):
                cur_index = (i*64 + j)*3
                data[i][j] = [image_vec[cur_index], image_vec[cur_index+1], image_vec[cur_index+2]]
        img = Image.fromarray(data, 'RGB')
        img.save("saved_pic/{}.png".format(self.cur_pic_index))
        self.cur_pic_index += 1
        # print(np.array([r, g, b], dtype=np.uint8).shape)
        print(data.shape)
        # return np.array([r, g, b], dtype=np.uint8)
        return data

    def make_observaton(self, raw_obs):
        if self.vision is False:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle', 'damage',
                     'opponents',
                     'rpm',
                     'track', 
                     'trackPos',
                     'wheelSpinVel']
            Observation = col.namedtuple('Observaion', names)
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/300.0,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/300.0,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/300.0,
                               angle=np.array(raw_obs['angle'], dtype=np.float32)/3.1416,
                               damage=np.array(raw_obs['damage'], dtype=np.float32),
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32))
        else:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle',
                     'opponents',
                     'rpm',
                     'track',
                     'trackPos',
                     'wheelSpinVel',
                     'img']
            Observation = col.namedtuple('Observaion', names)

            # Get RGB from observation
            # print(raw_obs)
            image_rgb = self.obs_vision_to_image_rgb(raw_obs['img'])
            # image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[8]])

            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               angle=np.array(raw_obs['angle'], dtype=np.float32)/3.1416,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb)

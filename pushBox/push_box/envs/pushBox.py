import os
import math
import numpy as np
from typing import ClassVar

import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding

import pybullet as p
import pybullet_data

class PushBoxEnv(gymnasium.Env):
     
    metadata: ClassVar = {
        "render_modes": ["human", "rgb_array"], 
        "render_fps": 60
    }
     
    def __init__(self):
        # 5 finite actions: nothing, right, left, up, and down
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)

        p.connect(p.DIRECT) #or p.DIRECT for non-graphical version
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8)
        # p.setRealTimeSimulation(0) 

         # Load assets
        p.loadURDF(os.path.join("plane.urdf"), [0, 0, 0], [0, 0, 0, 1])
        blockid1 = p.loadURDF(os.path.join("block.urdf"), [-7, 0, 0.25], [0, 0, 0, 1], useFixedBase = True)
        blockid2 = p.loadURDF(os.path.join("block.urdf"), [7, 0, 0.25], [0, 0, 0, 1], useFixedBase = True)
        blockid3 = p.loadURDF(os.path.join("block.urdf"), [0, -7, 0.25], [0, 0, 2, 2], useFixedBase = True)
        blockid4 = p.loadURDF(os.path.join("block.urdf"), [0, 7, 0.25], [0, 0, 2, 2], useFixedBase = True)

        # # randomize agent position
        agent_pos = np.random.uniform(low=-4, high=4, size=2)
        # randomize target position
        target_pos = np.random.uniform(low=-3, high=3, size=2)
        # Randomize box position close to the target
        box_pos_offsetx = np.random.uniform(low=1, high=3, size=1) * np.random.choice([-1, 1])
        box_pos_offsety = np.random.uniform(low=1, high=3, size=1) * np.random.choice([-1, 1])
        box_pos_offset = np.concatenate((box_pos_offsetx, box_pos_offsety))
        box_pos = target_pos + box_pos_offset

        self.agent = p.loadURDF(os.path.join("cube.urdf"), [agent_pos[0], agent_pos[1], 0.25], [0, 0, 0, 1])
        self.box = p.loadURDF(os.path.join("box.urdf"), [box_pos[0], box_pos[1], 0.5], [0, 0, 0, 1])
        self.target = p.loadURDF(os.path.join("sphere2red_nocol.urdf"), [target_pos[0], target_pos[1], 0.15], [0, 0, 0, 1], useFixedBase = True)
        self.focus_position, _ = p.getBasePositionAndOrientation(self.agent)
        p.resetDebugVisualizerCamera(cameraDistance=16.50, cameraYaw=0, cameraPitch=-50, cameraTargetPosition=self.focus_position)

        self.state = self.init_state()
        self.step_count = 0
        self.maxSteps = 10000
    
    def init_state(self):
           
        # # # randomize agent position
        # agent_pos = np.random.uniform(low=-4, high=4, size=2)
        # # randomize target position
        # target_pos = np.random.uniform(low=-3, high=3, size=2)
        # # Randomize box position close to the target
        # box_pos_offsetx = np.random.uniform(low=1, high=3, size=1) * np.random.choice([-1, 1])
        # box_pos_offsety = np.random.uniform(low=1, high=3, size=1) * np.random.choice([-1, 1])
        # box_pos_offset = np.concatenate((box_pos_offsetx, box_pos_offsety))
        # box_pos = target_pos + box_pos_offset

        # # # +/- x linear positions (distance 2)
        # # Randomize a common position for agent, target, and box along the x-axis
        # common_position = np.random.uniform(low=-5, high=6, size=1)

        # # Randomize x-offsets for target, box, and agent
        # sign_change = np.random.choice([-1, 1])
        # x_offset_target = np.array([6]) * sign_change
        # x_offset_box = np.array([3]) * sign_change
        # x_offset_agent = np.array([0]) * sign_change

        # # Calculate the x-positions for target, box, and agent
        # agent_x = x_offset_agent
        # box_x = x_offset_box
        # target_x = x_offset_target

        # # Combine x and y coordinates for agent, target, and box
        # target_pos = np.array([target_x, common_position])
        # box_pos = np.array([box_x, common_position])
        # agent_pos = np.array([agent_x, common_position])

        # # randomize agent, box, and target position on the same line (distance 2)

        # # +xy linear positions (distance 2)
        # Randomly choose the alignment axis (either 'x' or 'y')
        alignment_axis = np.random.choice(['x', 'y'])

        if alignment_axis == 'x':
            # Randomize a common position for agent, target, and box along the x-axis
            common_position = np.random.uniform(low=-5, high=6, size=1)

            # Randomize y-offsets for target, box, and agent
            sign_change = np.random.choice([-1, 1])
            y_offset_target = np.array([4]) * sign_change
            y_offset_box = np.array([2]) * sign_change
            y_offset_agent = np.array([0]) * sign_change

            # Calculate the y-positions for target, box, and agent
            agent_y = y_offset_agent
            box_y = y_offset_box
            target_y = y_offset_target
            

            # Combine x and y coordinates for agent, target, and box
            target_pos = np.array([common_position, target_y])
            box_pos = np.array([common_position, box_y])
            agent_pos = np.array([common_position, agent_y])
        else:  # alignment_axis == 'y'
            # Randomize a common position for agent, target, and box along the y-axis
            common_position = np.random.uniform(low=-5, high=6, size=1)

            # Randomize x-offsets for target, box, and agent
            sign_change = np.random.choice([-1, 1])
            x_offset_target = np.array([4]) * sign_change
            x_offset_box = np.array([2]) * sign_change
            x_offset_agent = np.array([0]) * sign_change

            # Calculate the x-positions for target, box, and agent
            agent_x = x_offset_agent
            box_x = x_offset_box
            target_x = x_offset_target

            # Combine x and y coordinates for agent, target, and box
            target_pos = np.array([target_x, common_position])
            box_pos = np.array([box_x, common_position])
            agent_pos = np.array([agent_x, common_position])

        p.resetBasePositionAndOrientation(self.agent, ([agent_pos[0], agent_pos[1], 0.25]), ([0, 0, 0, 1]))
        p.resetBasePositionAndOrientation(self.box, ([box_pos[0], box_pos[1], 0.5]), [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.target, ([target_pos[0], target_pos[1], 0.15]), ([0, 0, 0, 1]))
        # self.focus_position, _ = p.getBasePositionAndOrientation(self.agent)
        # p.resetDebugVisualizerCamera(cameraDistance=15.70, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=self.focus_position)

    def step(self, action):
        self.step_count += 1
        done = False
        #get agent, box and target positions
        agent_pos = np.array(p.getBasePositionAndOrientation(self.agent)[0][:2])
        target_pos = np.array(p.getBasePositionAndOrientation(self.target)[0][:2])
        box_pos = np.array(p.getBasePositionAndOrientation(self.box)[0][:2])
        agent_vec = box_pos - agent_pos
        box_vec = target_pos - box_pos
        #get agent velocity
        agent_vel = np.array(p.getBaseVelocity(self.agent)[0][:2])
        #get box velocity
        box_vel = np.array(p.getBaseVelocity(self.box)[0][:2])
        box_vel = box_vel if np.linalg.norm(box_vel) > 0.1 else [0.0, 0.0]

        # Calculate distance-based reward; if box is going toward target, pos reward; if it's going opposite dir, neg reward
        norm_box_vel = np.linalg.norm(box_vel[:2])
        unit_vec = box_vel / norm_box_vel if norm_box_vel != 0.0 else box_vel # to avoid division by 0
        box_reward_dist = 0.001 * np.dot(box_vec[:2]/np.linalg.norm(box_vec[:2]), unit_vec)
        if np.isnan(box_reward_dist):
            box_reward_dist = 0.0 

        # Calculate distance-based reward, if agent is going toward box pos reward if it's going opposite dir, neg reward
        # norm_agent_vel = np.linalg.norm(agent_vel[:2])
        # unit_vec = agent_vel / norm_agent_vel if norm_agent_vel != 0.0 else agent_vel # to avoid division by 0
        # agent_reward_dist = 0.00001 * np.dot(agent_vec[:2]/np.linalg.norm(agent_vec[:2]), unit_vec)
        # if np.isnan(agent_reward_dist):
        #     agent_reward_dist = 0.0

        # reward for moving the box
        # reward_moving = 0.0
        # current_vel = np.linalg.norm(box_vel)
        # if current_vel > self.previous_vel:
        #     reward_moving = 0.005
        #     self.previous_vel = current_vel
        #     print('box moved')


        # Reward the agent if box reaches target
        reward_finding = 0.0
        if np.linalg.norm(box_pos - target_pos) < 0.6:
            print('box reached target')
            reward_finding = 5.0
            self.reset()
            done = True

        # Existential reward penalty
        # existential_penalty = -0.00001

        if (self.step_count >= self.maxSteps):
            self.reset()
            done = True

        reward = box_reward_dist + reward_finding

        #move agent based on applied velocity
        self.set_actions(action)
        p.stepSimulation()

        if self.render_mode == 'human':
            self.render()

        obs = self.get_obs()
        return (
            obs,
            reward,
            done,
            False,
            dict(box_reward_dist=box_reward_dist, reward_finding=reward_finding)
        )

    def reset(self, seed=None, options=None):
        self.state = self.init_state()
        self.step_count = 0

        p.resetBaseVelocity(self.agent, [0, 0, 0])
        p.resetBaseVelocity(self.box, [0, 0, 0])

        return (self.get_obs(), {})
    
    def get_obs(self):
        # get agent's position
        agent_pos = np.array(p.getBasePositionAndOrientation(self.agent)[0][:2])
        
        # get box's position
        box_pos = np.array(p.getBasePositionAndOrientation(self.box)[0][:2])
        
        # get target's position
        target_pos = np.array(p.getBasePositionAndOrientation(self.target)[0][:2])
        
        # get agent's velocity
        agent_vel = np.array(p.getBaseVelocity(self.agent)[0][:2])
        
        # get box's velocity
        box_vel = np.array(p.getBaseVelocity(self.box)[0][:2])

        # # distance observations
        # # get agent to box distance
        # agentToBox = box_pos - agent_pos
        # # get box to target distance
        # boxToTarget = target_pos - box_pos
        # # get agent to box distance
        # agentToTarget = target_pos - agent_pos

    
        return np.concatenate((agent_pos, box_pos, target_pos, agent_vel, box_vel))
    

    def set_actions(self, action):

        force2apply = 70
        
        if action == 0: # do nothing
            force_x = 0
            force_y = 0
        elif action == 1:   # go right
            force_x = force2apply
            force_y = 0
        elif action == 2:   # go left
            force_x = -force2apply #* vecx
            force_y = 0
        elif action == 3:   # go up
            force_x = 0
            force_y = force2apply #* vecy
        elif action == 4:   # go down
            force_x = 0
            force_y = -force2apply #* vecy

        # apply velocity to joints of agents 
        p.applyExternalForce(self.agent, -1, [force_x, force_y, 0], p.getBasePositionAndOrientation(self.agent)[0], p.WORLD_FRAME)

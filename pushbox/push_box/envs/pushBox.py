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

        self.state = self.init_state()
        self.step_count = 0
        self.maxSteps = 30000
        self.previous_vel = 1.0
    
    def init_state(self):
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

        # randomize agent, box, and target position on the same line

        # Randomly choose the alignment axis (either 'x' or 'y')
        # alignment_axis = np.random.choice(['x', 'y'])

        # if alignment_axis == 'x':
        #     # Randomize a common position for agent, target, and box along the x-axis
        #     common_position = np.random.uniform(low=-2, high=2, size=1)

        #     # Randomize y-offsets for target, box, and agent
        #     y_offset_target = np.random.uniform(low=1, high=3, size=1)
        #     y_offset_box = np.random.uniform(low=1, high=3, size=1)
        #     y_offset_agent = np.random.uniform(low=0, high=1, size=1)

        #     # Calculate the y-positions for target, box, and agent
        #     agent_y = common_position + y_offset_agent
        #     box_y = agent_y + y_offset_box
        #     target_y = box_y + y_offset_target
            

        #     # Combine x and y coordinates for agent, target, and box
        #     target_pos = np.array([common_position, target_y])
        #     box_pos = np.array([common_position, box_y])
        #     agent_pos = np.array([common_position, agent_y])
        # else:  # alignment_axis == 'y'
        #     # Randomize a common position for agent, target, and box along the y-axis
        #     common_position = np.random.uniform(low=-2, high=2, size=1)

        #     # Randomize x-offsets for target, box, and agent
        #     x_offset_target = self.np_random.uniform(low=1, high=3, size=1)
        #     x_offset_box = self.np_random.uniform(low=1, high=3, size=1)
        #     x_offset_agent = self.np_random.uniform(low=0, high=1, size=1)

        #     # Calculate the x-positions for target, box, and agent
        #     agent_x = common_position + x_offset_agent
        #     box_x =  agent_x + x_offset_box
        #     target_x = box_x + x_offset_target

        #     # Combine x and y coordinates for agent, target, and box
        #     target_pos = np.array([target_x, common_position])
        #     box_pos = np.array([box_x, common_position])
        #     agent_pos = np.array([agent_x, common_position])

        self.agent = p.loadURDF(os.path.join("cube.urdf"), [agent_pos[0], agent_pos[1], 0.25], [0, 0, 0, 1])
        self.box = p.loadURDF(os.path.join("box.urdf"), [box_pos[0], box_pos[1], 0.5], [0, 0, 0, 1])
        self.target = p.loadURDF(os.path.join("sphere2red_nocol.urdf"), [target_pos[0], target_pos[1], 0.15], [0, 0, 0, 1], useFixedBase = True)
        self.focus_position, _ = p.getBasePositionAndOrientation(self.agent)
        p.resetDebugVisualizerCamera(cameraDistance=15.70, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=self.focus_position)

    def step(self, action):
        self.step_count += 1
        done = False
        #get agent, box and target positions
        agent_pos = np.array(p.getBasePositionAndOrientation(self.agent)[0])
        target_pos = np.array(p.getBasePositionAndOrientation(self.target)[0])
        box_pos = np.array(p.getBasePositionAndOrientation(self.box)[0])
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
        if np.linalg.norm(box_pos - target_pos) < 0.7:
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
        p.disconnect()
        self.state = self.init_state()
        self.step_count = 0
        self.previous_vel = 1.0

        return (self.get_obs(), {})
    
    def get_obs(self):
        # get agent's position
        agent_pos = p.getBasePositionAndOrientation(self.agent)[0]
        agent_pos = np.asarray(agent_pos)
        agent_pos = agent_pos.flat[:2]
        # get box's position
        box_pos = p.getBasePositionAndOrientation(self.box)[0]
        box_pos = np.asarray(box_pos)
        box_pos = box_pos.flat[:2]
        # get target's position
        target_pos = p.getBasePositionAndOrientation(self.target)[0]
        target_pos = np.asarray(target_pos)
        target_pos = target_pos.flat[:2]
        # get agent's velocity
        agent_vel = p.getBaseVelocity(self.agent)[0]
        agent_vel = np.asarray(agent_vel)
        agent_vel = agent_vel.flat[:2]
        # get box's velocity
        box_vel = p.getBaseVelocity(self.box)[0][:2]
        box_vel = np.asarray(box_vel)
        box_vel = box_vel.flat[:2]

        return np.concatenate((agent_pos, box_pos, target_pos, agent_vel, box_vel))
    

    def set_actions(self, action):
        agent_pos = np.array(p.getBasePositionAndOrientation(self.agent)[0])
        box_pos = np.array(p.getBasePositionAndOrientation(self.box)[0])

        force2apply = 70
        # vecx = np.linalg.norm(box_pos[0] - agent_pos[0])
        # vecy = np.linalg.norm(box_pos[1] - agent_pos[1])
        
        if action == 0: # do nothing
            force_x = 0
            force_y = 0
        elif action == 1:   # go right
            # if vecx > 2:
            #     force2apply = 30
            force_x = force2apply
            force_y = 0
        elif action == 2:   # go left
            # if vecx > 2:
            #     force2apply = 30
            force_x = -force2apply #* vecx
            force_y = 0
        elif action == 3:   # go up
            # if vecy > 2:
            #     force2apply = 30
            force_x = 0
            force_y = force2apply #* vecy
        elif action == 4:   # go down
            # if vecy > 2:
            #     force2apply = 30
            force_x = 0
            force_y = -force2apply #* vecy

        # apply velocity to joints of agents 
        p.applyExternalForce(self.agent, -1, [force_x, force_y, 0], p.getBasePositionAndOrientation(self.agent)[0], p.WORLD_FRAME)

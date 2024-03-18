import os
import math
import time
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
        self.action_space = spaces.Box(
            low=np.array([-1,-1], dtype=np.float32),
            high=np.array([1,1], dtype=np.float32))
        # self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)

        p.connect(p.GUI) #or p.DIRECT for non-graphical version
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
        box_pos = np.random.uniform(low=-3, high=3, size=2)
        # Randomize box position close to the target
        target_pos_offsetx = np.random.uniform(low=1, high=3, size=1) * np.random.choice([-1, 1])
        target_pos_offsety = np.random.uniform(low=1, high=3, size=1) * np.random.choice([-1, 1])
        target_pos_offset = np.concatenate((target_pos_offsetx, target_pos_offsety))
        target_pos = box_pos + target_pos_offset

        box_shape = 0.6
        agent_mass = 4
        box_mass = 2
        visualShapeId = -1

        agentId =  p.createCollisionShape(
            p.GEOM_CYLINDER, radius=0.4, height=0.3)
        
        boxId = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=box_shape, height=0.3)
        
        self.agent = p.createMultiBody(
            agent_mass,
            agentId,
            visualShapeId, 
            [agent_pos[0], agent_pos[1], 1])
        
        self.box = p.createMultiBody(
            box_mass,
            boxId,
            visualShapeId, 
            [box_pos[0], box_pos[1], 1])
        
        p.changeVisualShape(self.agent, -1, rgbaColor=[1, 1, 0, 1])
        p.changeVisualShape(self.box, -1, rgbaColor=[0.769, 0.537, 0.157, 1])
        
        self.target = p.loadURDF(os.path.join("sphere2red_nocol.urdf"), [target_pos[0], target_pos[1], 0.15], [0, 0, 0, 1], useFixedBase = True)

        self.focus_position, _ = p.getBasePositionAndOrientation(self.agent)
        p.resetDebugVisualizerCamera(cameraDistance=16.50, cameraYaw=0, cameraPitch=-50, cameraTargetPosition=self.focus_position)

        self.state = self.init_state()
        self.step_count = 0
        self.maxSteps = 300
    
    def init_state(self):

        # # randomize agent position
        agent_pos = np.random.uniform(low=-2, high=2, size=2)

        boxVec = np.random.uniform(low=1, high=2, size=2)
        boxVec = self.rotate_angle(boxVec, np.pi*np.random.uniform(-1, 1))
        box_pos = agent_pos + boxVec

        # angle rotation
        theta = np.random.uniform(low=-15, high=15, size=1)/2
        theta = theta*np.pi/180

        vecAB = (box_pos-agent_pos)/np.linalg.norm(box_pos-agent_pos)

        target_pos = box_pos + vecAB * 2
        target_pos = self.rotate_angle(target_pos, theta, box_pos)

        
        p.resetBasePositionAndOrientation(self.agent, ([agent_pos[0], agent_pos[1], 0.15]), ([0, 0, 0, 1]))
        p.resetBasePositionAndOrientation(self.box, ([box_pos[0], box_pos[1], 0.15]), [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.target, ([target_pos[0], target_pos[1], 0.15]), ([0, 0, 0, 1]))
        # self.focus_position, _ = p.getBasePositionAndOrientation(self.agent)
        # p.resetDebugVisualizerCamera(cameraDistance=15.70, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=self.focus_position)

    def rotate_angle(self, xy, radians, origin=(0, 0)):
        """Rotate a point around a given point.
        
        I call this the "high performance" version since we're caching some
        values that are needed >1 time. It's less readable than the previous
        function but it's faster.
        """
        x, y = xy
        offset_x, offset_y = origin
        adjusted_x = (x - offset_x)
        adjusted_y = (y - offset_y)
        cos_rad = math.cos(radians)
        sin_rad = math.sin(radians)
        qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
        qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

        return qx, qy
    
    def step(self, action):
        # time.sleep(1/60)
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
        box_reward_dist = 0.01 * np.dot(box_vec[:2]/np.linalg.norm(box_vec[:2]), unit_vec)
        if np.isnan(box_reward_dist):
            box_reward_dist = 0.0 

        # Reward the agent if box reaches target
        reward_finding = 0.0
        if np.linalg.norm(box_pos - target_pos) < 0.6:
            print('box reached target')
            reward_finding = 5.0
            self.reset()
            done = True

        # Existential reward penalty
        existential_penalty = -1/self.maxSteps

        if (self.step_count >= self.maxSteps):
            self.reset()
            done = True

        reward = box_reward_dist + reward_finding + existential_penalty

        #move agent based on applied velocity
        # self.set_actions(action)
        p.applyExternalForce(self.agent, -1, [70*action[0], 70*action[1], 0], p.getBasePositionAndOrientation(self.agent)[0], p.WORLD_FRAME)
        p.stepSimulation()

        if self.render_mode == 'human':
            self.render()

        obs = self.get_obs()
        return (
            obs,
            reward,
            done,
            False,
            dict(box_reward_dist=box_reward_dist, reward_finding=reward_finding, existential_penalty=existential_penalty)
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
        p.applyExternalForce(self.agent, -1, (force_x, force_y, 0), p.getBasePositionAndOrientation(self.agent)[0], p.WORLD_FRAME)

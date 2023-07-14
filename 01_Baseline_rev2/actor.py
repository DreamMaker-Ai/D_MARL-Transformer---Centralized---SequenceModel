import random

import pylab as p
import ray
import gym
import tensorflow as tf
import numpy as np
from collections import deque

from battlefield_strategy import BattleFieldStrategy

from models import MarlTransformerSequenceModel
from utils_gnn import get_alive_agents_ids
from utils_transformer import make_mask, make_padded_obs
from utils_sequential import shuffle_alive_agents, get_shuffled_tensor, sequential_model, \
    building_policy, make_c0_matrix, make_c1_matrix, make_c_matrix


@ray.remote
# @ray.remote(num_cpus=1, num_gpus=0)
class Actor:
    def __init__(self, pid, epsilon):
        self.pid = pid

        # Make a copy of environment
        self.env = BattleFieldStrategy()
        self.action_space_dim = self.env.action_space.n

        # Make a q_network
        self.policy = MarlTransformerSequenceModel(config=self.env.config)

        self.epsilon = epsilon
        self.gamma = self.env.config.gamma
        self.n_frames = self.env.config.n_frames
        self.obs_shape = (self.env.config.grid_size,
                          self.env.config.grid_size,
                          self.env.config.observation_channels * self.n_frames)

        # Define local buffer
        self.buffer = []
        self.episode_buffer = []

        # Initialize environment
        ### The followings are reset in 'reset_states'
        self.frames = None  # For each agent in env
        self.states = None
        self.prev_actions = None

        self.alive_agents_ids = None  # For all agents, including dummy ones
        self.padded_states = None
        self.padded_prev_actions = None
        self.mask = None

        self.episode_reward = None
        self.step = None

        ### Initialize above Nones
        observations = self.env.reset()
        self.reset_states(observations)

        # actorからはGPUを見えなくする
        tf.config.set_visible_devices([], 'GPU')

        # Build policy
        shuffled_alive_agents_ids, shuffled_order = shuffle_alive_agents(self.alive_agents_ids)
        shuffled_padded_states = \
            get_shuffled_tensor(self.padded_states,
                                shuffled_order=shuffled_order)  # (1,n,g,g,ch*n_frames)

        building_policy(config=self.env.config,
                        policy=self.policy,
                        alive_agents_ids=shuffled_alive_agents_ids,
                        padded_states=shuffled_padded_states,
                        masks=self.mask,
                        training=False)

    def reset_states(self, observations):
        # TODO prev_actions
        """
        alive_agents_ids: list of alive agent id

        # For agents in Env
             each agent stacks observations n-frames in channel-dims
             -> observations[red.id]: (grid_size,grid_size,channels)

             -> generate deque of length=n_frames
             self.frames[red.id]: deque[(grid_size,grid_size,channels),...]

             -> transform to states
             states[red.id]: (grid_size,grid_size,channels*n_frames)

             self.prev_actions[red.id]: int (TODO)
        """

        self.frames = {}
        self.states = {}
        self.prev_actions = {}

        for red in self.env.reds:
            # all reds are alive when reset

            self.frames[red.id] = deque([observations[red.id]] * self.n_frames,
                                        maxlen=self.n_frames)
            # [(grid_size,grid_size,channels),...,(grid_size,grid_size,channels)]

            self.states[red.id] = np.concatenate(self.frames[red.id], axis=2).astype(np.float32)
            # (grid_size,grid_size,channels*n_frames)

            # self.prev_actions[red.id] = 0

        # alive_agents_ids: list of alive agent id, [int,...], len=num_alive_agents
        self.alive_agents_ids = get_alive_agents_ids(env=self.env)

        # Get raw observation list of alive agents
        raw_states = []

        for a in self.alive_agents_ids:
            agent_id = 'red_' + str(a)
            raw_states.append(self.states[agent_id])  # append (g,g,ch*n_frames)

        # Get padded observations ndarray for all agents, including dead and dummy agents
        self.padded_states = \
            make_padded_obs(max_num_agents=self.env.config.max_num_red_agents,
                            obs_shape=self.obs_shape,
                            raw_obs=raw_states)  # (1,n,g,g,ch*n_frames)

        # Get mask for the padding
        self.mask = make_mask(alive_agents_ids=self.alive_agents_ids,
                              max_num_agents=self.env.config.max_num_red_agents)  # (1,n)

        # reset episode variables
        self.episode_reward = 0
        self.step = 0

    def rollout(self, current_weights):
        """
        rolloutを,self.env.config.actor_rollout_steps回（=10回）実施し、
        各経験の優先度（TD error）と経験（transitions）を格納

        :return td_errors, transitions, self.pid (process id)
                td_errors: (self.env.config.actor_rollout_steps,)=(10,)
                transitions=[transition,...], len=self.env.config.actor_rollout_steps=10
                    transition = (
                        shuffled_padded_states,  # (1,n,g,g,ch*n_frames)
                        (shuffled_)padded_actions,  # (1,n)
                        shuffled_padded_rewards,  # (1,n)
                        shuffled_next_padded_states,  # (1,n,g,g,ch*n_frames)  # 使わない
                        shuffled_padded_dones,  # (1,n), bool
                        self.mask,  # (1,n), bool
                        shuffled_padded_next_states_for_q,  # (1,n,g,g,ch*n_frames)
                        shuffled_padded_next_actions_for_q,  # (1,n)
                        c,  # order change matrix, (1,n,n)
                    )
        """
        # 重みを更新
        self.policy.set_weights(weights=current_weights)

        # 1 episode分のrolloutを行って、経験をepisode_bufferに一時保管
        dones = {"all_dones": False}

        while not dones["all_dones"]:

            shuffled_alive_agents_ids, shuffled_order = shuffle_alive_agents(
                self.alive_agents_ids)

            shuffled_padded_states = \
                get_shuffled_tensor(self.padded_states,
                                    shuffled_order=shuffled_order)  # (1,n,g,g,ch*n_frames)

            generated_qlogits, generated_actions = \
                sequential_model(config=self.env.config,
                                 policy=self.policy,
                                 shuffled_padded_states=shuffled_padded_states,
                                 mask=self.mask,
                                 training=False
                                 )
            # (1,n,action_dim), (1,n))

            # get alive_agents & all agents actions. action=0 <- do nothing
            actions = {}  # For alive agents

            acts = generated_actions[0]  # (n,)
            padded_actions = np.expand_dims(acts, axis=0)  # (1,n)

            for i, a in enumerate(shuffled_alive_agents_ids):
                agent_id = 'red_' + str(a)

                if np.random.rand() >= self.epsilon:  # epsilon-greedy
                    actions[agent_id] = acts[i]
                else:
                    actions[agent_id] = np.random.randint(low=0,
                                                          high=self.action_space_dim)

                padded_actions[0, i] = actions[agent_id]

            # One step of Lanchester simulation, for alive agents in env
            next_obserations, rewards, dones, infos = self.env.step(actions)

            # Make next_agents_states, next_agents_adjs, and next_alive_agents_ids,
            # including dummy ones
            next_alive_agents_ids = get_alive_agents_ids(env=self.env)

            ### For alive agents in env
            next_states = {}

            for idx in next_alive_agents_ids:
                agent_id = 'red_' + str(idx)

                self.frames[agent_id].append(
                    next_obserations[agent_id]
                )  # append (g,g,ch) to deque

                next_states[agent_id] = np.concatenate(
                    self.frames[agent_id], axis=2
                ).astype(np.float32)  # (g,g,ch*n_frames)

            # Get next_observation list of alive agents
            raw_next_states = []

            for a in next_alive_agents_ids:
                agent_id = 'red_' + str(a)
                raw_next_states.append(next_states[agent_id])  # append (g,g,ch*n_frames)

            # Get padded next observations ndarray of all agent
            next_padded_states = \
                make_padded_obs(
                    max_num_agents=self.env.config.max_num_red_agents,
                    obs_shape=self.obs_shape,
                    raw_obs=raw_next_states
                )  # (1,n,g,g,ch*n_frames)

            # Get next mask for the padding
            next_mask = \
                make_mask(
                    alive_agents_ids=next_alive_agents_ids,
                    max_num_agents=self.env.config.max_num_red_agents
                )  # (1,n)

            # 終了判定
            if self.step > self.env.config.max_steps:

                for idx in self.alive_agents_ids:
                    agent_id = 'red_' + str(idx)
                    dones[agent_id] = True

                dones['all_dones'] = True

            # agents_rewards and agents_dones, including dead and dummy ones
            agents_rewards = []
            agents_dones = []

            for a in self.alive_agents_ids:
                agent_id = 'red_' + str(a)
                agents_rewards.append(float(rewards[agent_id]))
                agents_dones.append(dones[agent_id])

            while len(agents_rewards) < self.env.config.max_num_red_agents:
                agents_rewards.append(0.0)
                agents_dones.append(True)

            if len(agents_rewards) != self.env.config.max_num_red_agents:
                raise ValueError()

            if len(agents_dones) != self.env.config.max_num_red_agents:
                raise ValueError()

            # Update episode rewards
            self.episode_reward += np.sum(agents_rewards)

            # list -> ndarray
            padded_rewards = np.stack(agents_rewards, axis=0)  # (n,)
            padded_rewards = np.expand_dims(padded_rewards, axis=0)  # (1,n)

            padded_dones = np.stack(agents_dones, axis=0)  # (n,), bool
            padded_dones = np.expand_dims(padded_dones, axis=0)  # (1,n)

            c0 = make_c0_matrix(shuffled_order=shuffled_order,
                                n=self.env.config.max_num_red_agents)  # (1,n,n)

            c1 = make_c1_matrix(alive_ids=self.alive_agents_ids,
                                n=self.env.config.max_num_red_agents)  # (1,n,n)

            # Enumerate to shuffled_order
            shuffled_padded_rewards = get_shuffled_tensor(padded_rewards, shuffled_order)  # (1,n)
            shuffled_padded_dones = get_shuffled_tensor(padded_dones, shuffled_order)  # (1.n), bool

            # Append to buffer
            episode_transition = (
                shuffled_padded_states,  # (1,n,g,g,ch*n_frames)
                padded_actions,  # (1,n)
                shuffled_padded_rewards,  # (1,n)
                next_padded_states,  # (1,n,g,g,ch*n_frames),  使わない
                shuffled_padded_dones,  # (1,n), bool
                self.mask,  # (1,n), bool
                c0,  # (1,n,n)
                c1,  # (1,n,n)
            )

            self.episode_buffer.append(episode_transition)

            if dones['all_dones']:
                # print(f'episode reward = {self.episode_reward}')
                observations = self.env.reset()
                self.reset_states(observations)
            else:
                self.alive_agents_ids = next_alive_agents_ids
                self.padded_states = next_padded_states
                self.mask = next_mask

                self.step += 1

        """ self.episode_buffer から pisode_transition を順に読み込んで、
            transitionを作成し、self.buffer 順に格納する """
        self.make_transitions()

        # 各transitionの初期優先度（td_error）の計算
        if self.env.config.prioritized_replay:

            states = np.vstack([transition[0] for transition in self.buffer])  # (b,15,15,15,6)
            actions = np.vstack([transition[1] for transition in self.buffer])  # (b,15)
            rewards = np.vstack([transition[2] for transition in self.buffer])  # (b, 15)
            next_states = np.vstack(
                [transition[3] for transition in self.buffer])  # (b,15,15,1,6)  使わない
            dones = np.vstack([transition[4] for transition in self.buffer])  # (b,15)
            mask = np.vstack([transition[5] for transition in self.buffer])  # (b,15)

            next_states_for_q = \
                np.vstack([transition[6] for transition in self.buffer])  # (b,15,15,15,6)
            next_actions_for_q = \
                np.vstack([transition[7] for transition in self.buffer])  # (b,15)
            cs = np.vstack([transition[8] for transition in self.buffer])  # (b,15,15)

            """ compute TD errors """
            b = states.shape[0]  # b=102
            start_symbols = [5] * b
            start_symbols = np.array(start_symbols, dtype=np.int32)  # (b,)
            start_symbols = np.expand_dims(start_symbols, axis=-1)  # (b,1)

            next_input_actions = np.concatenate(
                [start_symbols, next_actions_for_q[:, :-1]], axis=-1)  # (b,n)

            next_q_values, _ = self.policy(next_states_for_q,
                                           next_input_actions,
                                           mask, training=False)
            # (b,n,action_dim)

            next_q_values = \
                tf.einsum('bij, bjk -> bik', cs, next_q_values)  # (b,n,action_dim)

            next_actions = tf.argmax(next_q_values, axis=-1)  # (b,15)
            next_actions = tf.cast(next_actions, dtype=tf.int32)
            next_actions_one_hot = tf.one_hot(next_actions,
                                              depth=self.action_space_dim)  # (b,15,5)

            next_maxQ = tf.reduce_sum(next_q_values * next_actions_one_hot, axis=-1)  # (b,15)

            TQ = rewards + self.gamma * (1 - dones) * next_maxQ  # (b,15)

            # input actions

            input_actions = np.concatenate(
                [start_symbols, actions[:, :-1]], axis=-1
            )  # (b,n)=[(b,1)(b,n-1)]=(b,15)

            q_values, _ = self.policy(
                states,
                input_actions,
                mask,
                training=False
            )  # (b,n,action_dim)=(b,15,5)

            actions_one_hot = tf.one_hot(actions, depth=self.action_space_dim)  # (b,15,5)
            Q = tf.reduce_sum(q_values * actions_one_hot, axis=-1)  # (b,15)

            td_errors = np.abs(TQ - Q)  # (b,15)

            masked_td_errors = td_errors * mask.astype(np.float32)  # (b,15)
            masked_td_errors = \
                np.sum(masked_td_errors, axis=-1) / \
                np.sum(mask.astype(np.float32), axis=-1)  # (b,)

        else:
            masked_td_errors = np.ones((self.env.config.actor_rollout_steps,),
                                       dtype=np.float32)  # (b,)

        transitions = self.buffer
        self.buffer = []
        self.episode_buffer = []

        return masked_td_errors, transitions, self.pid

    def make_transitions(self):
        """
            self.episode_buffer から pisode_transition を順に読み込んで、
            transitionを作成し、self.buffer 順に格納する。

            episode_transition = (
                shuffled_padded_states,  # (1,n,g,g,ch*n_frames)
                padded_actions,  # (1,n)
                shuffled_padded_rewards,  # (1,n)
                shuffled_next_padded_states,  # (1,n,g,g,ch*n_frames)  ＃使わない
                shuffled_padded_dones,  # (1,n), bool
                self.mask,  # (1,n), bool
                c0  # (1,n,n)
                c1  # (1,n,n)
            )

            transition = (
                shuffled_padded_states,  # (1,n,g,g,ch*n_frames)
                padded_actions,  # (1,n)
                shuffled_padded_rewards,  # (1,n)
                shuffled_next_padded_states,  # (1,n,g,g,ch*n_frames)  # 使わない
                shuffled_padded_dones,  # (1,n), bool
                self.mask,  # (1,n), bool
                shuffled_next_padded_states,  # (1,n,g,g,ch*n_frames)
                shuffled_next_padded_actions,  # (1,n)
                c,  # (1,n,n)
            )
        """

        for i in range(1, len(self.episode_buffer)):
            current_ex = self.episode_buffer[i - 1]
            next_ex = self.episode_buffer[i]

            next_s = next_ex[0]  # (1,n,g,g,ch*n_frames)
            next_a = next_ex[1]  # (1,n)

            c0 = current_ex[6]  # (1,n,n)
            c1 = current_ex[7]  # (1,n,n)
            c2 = next_ex[7]  # (1,n,n)
            c3 = next_ex[6]  # (1,n,n)

            c = make_c_matrix(c0, c1, c2, c3)

            transition = list(current_ex[0:6])
            transition.append(next_s)
            transition.append(next_a)
            transition.append(c)

            transition = tuple(transition)
            self.buffer.append(transition)

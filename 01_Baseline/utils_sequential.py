import numpy as np
import tensorflow as tf
import random


def building_policy(config, policy, alive_agents_ids, padded_states, masks, training):
    # Make input action sequence for building policy
    input_actions = [0] * config.max_num_red_agents
    input_actions[0] = 5

    for i in range(len(alive_agents_ids) - 1):
        input_actions[i+1] = 3

    input_actions = np.array(input_actions, dtype=np.int32)  # (n,)
    input_actions = np.expand_dims(input_actions, axis=0)  # (b,n)=(1,15)

    # Build policy
    policy(padded_states, input_actions, masks, training=training)


def sequential_model(config, policy, shuffled_padded_states, mask, training):
    """
    :param shuffled_padded_states: (b,n,g,g,ch*n_frames)
    :param mask: (b,n)

    :return:
        qlogits_generated: (b,n,action_dim)
        actions_generated: (b,n)
    """

    batch_size = shuffled_padded_states.shape[0]
    num_agents = config.max_num_red_agents  # n=15

    start_actions = [5] * batch_size  # 5=start symbol
    start_actions = np.array(start_actions, dtype=np.int32)  # (b,)
    start_actions = np.expand_dims(start_actions, axis=-1)  # (b,1)

    qlogits_generated = []
    actions_generated = []

    for i in range(num_agents):
        pad_len = num_agents - (i + 1)  # n-1,...,0
        pad_0 = np.zeros((batch_size, pad_len), dtype=np.int32)  # (b,n-(i+1))

        input_actions = \
            np.concatenate([start_actions, pad_0],
                           axis=-1, dtype=np.int32)  # (b,n); (b,i+1)+(b,n-(i+1)), int32

        if input_actions.shape[1] != num_agents:
            raise ValueError()

        shuffled_qlogits, _ = \
            policy(shuffled_padded_states,  # (b,n,g,g,ch*n_frames)
                   input_actions,  # (b,n), int32
                   mask,  # (b,n), bool
                   training=training)  # (b,n,action_dim), _

        qlogit = shuffled_qlogits[:, i, :]  # (b,action_dim)
        qlogit = tf.expand_dims(qlogit, axis=1)  # (b,1,action_dim)

        action = np.argmax(qlogit, axis=-1)  # (b,1)

        qlogits_generated.append(qlogit)  # list of (b,1,action_dim)
        actions_generated.append(action)  # list of (b,1)

        start_actions = np.concatenate([start_actions, action], axis=-1)  # (b i+2)

    qlogits_generated = tf.concat(qlogits_generated, axis=1)  # (b,n,action_dim)

    actions_generated = np.concatenate(actions_generated, axis=-1)  # (b,n)

    return qlogits_generated, actions_generated  # (b,n,action_dim), (b,n)


def get_shuffled_tensor(x, shuffled_order):
    """
    :param x: (1,n,g,g,ch*n_frames) or (1,n)
    :param shuffled_order: [int,...], len=num_alive_agents=a
    :return: shuffled_x, (1,n,g,g,ch*n_frames) or (1,n)
    """

    num_alive_agents = len(shuffled_order)  # a

    u = x[:, :num_alive_agents]  # (1,a,g,g,ch*n_frames) or (1,a)
    v = x[:, num_alive_agents:]  # (1,n-a,g,g,ch*n_frames) or (1,n-a)

    u = u[:, shuffled_order]  # (1,a,g,g,ch*n_frames) or (1,a)

    shuffled_x = np.concatenate([u, v], axis=1)  # (1,n,g,g,ch*n_frames) or (1,n)

    return shuffled_x


def shuffle_alive_agents(alive_agents_ids):
    """
    :param alive_agents_ids: [int, ...], len=a
    :return: shuffled_alibve_agents_ids, [int, ...], len=a
    """
    order = [i for i in range(len(alive_agents_ids))]

    shuffled_order = random.sample(order, len(order))
    shuffled_alibve_agents_ids = []
    for i in shuffled_order:
        shuffled_alibve_agents_ids.append(alive_agents_ids[i])

    return shuffled_alibve_agents_ids, shuffled_order


def main():
    n = 5
    g = 2
    ch = 1
    n_frames = 1

    alive_agents_ids = [1, 3, 4]
    print(alive_agents_ids)

    shuffled_alive_agents_ids, shuffled_order = shuffle_alive_agents(alive_agents_ids)
    print(shuffled_order)
    print(shuffled_alive_agents_ids)

    # for states
    padded_states_1 = np.random.rand(1, len(alive_agents_ids), g, g, ch * n_frames)
    print(padded_states_1.shape)  # (1,a,g,g,ch*n_frames)

    padded_states_2 = np.zeros((1, n - len(alive_agents_ids), g, g, ch * n_frames))
    print(padded_states_2.shape)  # (1,n-a,g,g,ch*n_frames)

    padded_states = np.concatenate([padded_states_1, padded_states_2], axis=1)
    print(padded_states.shape)  # (1,n,g,g,ch*n_frames)

    shuffled_padded_states = get_shuffled_tensor(padded_states, shuffled_order)
    print(shuffled_padded_states.shape)

    print(padded_states)
    print(shuffled_padded_states)

    # for dones
    dones1 = [False] * len(alive_agents_ids)
    dones1[1] = True
    dones2 = [True] * (n - len(alive_agents_ids))
    dones = dones1 + dones2
    dones = np.array(dones)
    dones = np.expand_dims(dones, axis=0)
    print(dones)
    print(shuffled_order)

    shuffled_dones = get_shuffled_tensor(dones, shuffled_order)
    print(shuffled_dones)


if __name__ == '__main__':
    main()

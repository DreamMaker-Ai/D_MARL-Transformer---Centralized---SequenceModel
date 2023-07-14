import numpy as np
import tensorflow as tf
import random


def building_policy(config, policy, alive_agents_ids, padded_states, masks, training):
    # Make input action sequence for building policy
    input_actions = [0] * config.max_num_red_agents
    input_actions[0] = 5

    for i in range(len(alive_agents_ids) - 1):
        input_actions[i + 1] = 3

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
    :return: shuffled_alive_agents_ids, [int, ...], len=a
    """
    order = [i for i in range(len(alive_agents_ids))]

    shuffled_order = random.sample(order, len(order))
    shuffled_alive_agents_ids = []
    for i in shuffled_order:
        shuffled_alive_agents_ids.append(alive_agents_ids[i])

    return shuffled_alive_agents_ids, shuffled_order


def sequential_model_2(config, policy, shuffled_padded_states, mask, training):
    """
    policy出力から,最終シーケンスの scores を読み込んでリターンする部分を追加

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

        shuffled_qlogits, scores = \
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

    return qlogits_generated, actions_generated, scores
    # (b,n,action_dim), (b,n)
    # scores=[enc_score, dec_scores], enc_scores: (1,num_head,n,n),
    # dec_scores=[score_causal_self_att, score_causal_att], [(1,num_head,n,n),(1,num_head,n,n)]


def make_c0_matrix(shuffled_order, n):
    all_ids = [*range(n)]  # len=n

    diff = list(set(all_ids) ^ set(shuffled_order))
    diff.sort()

    agents_order = shuffled_order + diff  # len=n

    c0 = np.eye(n, dtype=np.float32)[agents_order]  # (n,n)

    c0 = np.expand_dims(c0, axis=0)  # (1,n,n)

    return c0


def make_c1_matrix(alive_ids, n):
    all_ids = [*range(n)]
    diff = list(set(all_ids) ^ set(alive_ids))
    diff.sort()

    c1 = np.eye(n, dtype=np.float32)[alive_ids + diff]
    c1 = np.expand_dims(c1, axis=0)

    return c1


def make_c_matrix(c0, c1, c2, c3):
    c2_inv = np.linalg.inv(c2)
    c3_inv = np.linalg.inv(c3)

    c = np.einsum('bij, bjk -> bik', c2_inv, c3_inv)
    c = np.einsum('bij, bjk -> bik', c1, c)
    c = np.einsum('bij, bjk -> bik', c0, c)

    return c


def test_c():
    n = 15

    x_ids = [0, 2, 3, 5, 10, 13]
    next_x_ids = [0, 2, 5, 13]

    shuffled_ids, shuffled_order = shuffle_alive_agents(x_ids)
    next_shuffled_ids, next_shuffled_order = shuffle_alive_agents(next_x_ids)

    all_ids = [*range(n)]

    diff1 = list(set(all_ids) ^ set(x_ids))
    diff2 = list(set(all_ids) ^ set(next_x_ids))

    shuffled_x = shuffled_ids + diff1
    shuffled_next_x = next_shuffled_ids + diff2

    shuffled_x = np.expand_dims(shuffled_x, axis=0)
    shuffled_next_x = np.expand_dims(shuffled_next_x, axis=0)

    c0 = make_c0_matrix(shuffled_order, n)
    c1 = make_c1_matrix(x_ids, n)
    c2 = make_c1_matrix(next_x_ids, n)
    c3 = make_c0_matrix(next_shuffled_order, n)

    c = make_c_matrix(c0, c1, c2, c3)

    print(shuffled_x)
    print(np.einsum('bij, bj -> bi', c, shuffled_next_x))


def main():
    test_c()


if __name__ == '__main__':
    main()

import os.path

import tensorflow as tf
import numpy as np

from config import Config
from utils_transformer import make_mask, make_padded_obs, make_causal_mask, make_attention_mask


class CNNBlock(tf.keras.models.Model):
    """
    :param obs_shape: (15,15,6)
    :param max_num_agents=n=15
    :return: (None,n,hidden_dim)=(None,15,256)

    Model: "cnn_block_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     time_distributed_6 (TimeDis  multiple                 448
     tributed)

     time_distributed_7 (TimeDis  multiple                 36928
     tributed)

     time_distributed_8 (TimeDis  multiple                 36928
     tributed)

     time_distributed_9 (TimeDis  multiple                 36928
     tributed)

     time_distributed_10 (TimeDi  multiple                 0
     stributed)

     time_distributed_11 (TimeDi  multiple                 147712
     stributed)

    =================================================================
    Total params: 258,944
    Trainable params: 258,944
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(CNNBlock, self).__init__(**kwargs)

        self.config = config

        self.conv0 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=1,
                    strides=1,
                    activation='relu',
                    kernel_initializer='Orthogonal'
                )
            )

        self.conv1 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    strides=2,
                    activation='relu',
                    kernel_initializer='Orthogonal'
                )
            )

        self.conv2 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    strides=1,
                    activation='relu',
                    kernel_initializer='Orthogonal'
                )
            )

        self.conv3 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    strides=1,
                    activation='relu',
                    kernel_initializer='Orthogonal'
                )
            )

        self.flatten1 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Flatten()
            )

        self.dense1 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.hidden_dim,
                    activation=None
                )
            )

    @tf.function
    def call(self, inputs):
        # inputs: (b,n,g,g,ch*n_frames)=(1,15,15,15,6)

        h = self.conv0(inputs)  # (1,15,15,15,64)
        h = self.conv1(h)  # (1,15,7,7,64)
        h = self.conv2(h)  # (1,15,5,5,64)
        h = self.conv3(h)  # (1,15,3,3,64)

        h1 = self.flatten1(h)  # (1,15,576)

        features = self.dense1(h1)  # (b,n,hidden_dim)=(1,15,256)

        return features

    def build_graph(self):
        """ For summary & plot_model """
        x = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.grid_size,
                   self.config.grid_size,
                   self.config.observation_channels * self.config.n_frames)
        )

        model = \
            tf.keras.models.Model(
                inputs=[x],
                outputs=self.call(x),
                name='cnn_model'
            )

        return model


class PositionalEncoding:
    def __init__(self, seq_len, depth):
        """
        :param seq_len: config.max_num_red_agents=n(=15, for training)
        :param depth: config.hidden_dim(=256) or config.key_dim(=64)
        """
        super(PositionalEncoding, self).__init__()

        h_dim = depth / 2

        positions = np.arange(seq_len)  # (n,)
        positions = np.expand_dims(positions, axis=-1)  # (n,1)

        depths = np.arange(h_dim) / h_dim  # (h_dim,)
        depths = np.expand_dims(depths, axis=0)  # (1,h_dim)

        angle_rates = 1 / (10000 ** depths)  # (1, h_dim)
        angle_rads = positions * angle_rates  # (n, h_dim)

        pos_encoding = \
            np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)  # (n,depth)

        # Set the relative scale of features and pos_encoding
        # pos_encoding = pos_encoding / np.sqrt(depth)

        self.pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)


class SelfAttentionBlock(tf.keras.models.Model):
    """
    Two layers of MultiHeadAttention (Self Attention with provided mask)

    :param mask: (None,n,n), bool
    :param max_num_agents=15=n
    :param hidden_dim = 256

    :return: features: (None,n,hidden_dim)=(None,15,256)
             score: (None,num_heads,n,n)=(None,2,15,15)

    Model: "self_attention_block"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     multi_head_attention (Multi  multiple                 131712
     HeadAttention)

     add (Add)                   multiple                  0

     dense_1 (Dense)             multiple                  131584

     dense_2 (Dense)             multiple                  131328

     dropout (Dropout)           multiple                  0

     add_1 (Add)                 multiple                  0

    =================================================================
    Total params: 394,624
    Trainable params: 394,624
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)

        self.config = config

        self.mha1 = \
            tf.keras.layers.MultiHeadAttention(
                num_heads=self.config.num_heads,
                key_dim=self.config.key_dim,
            )

        self.add1 = \
            tf.keras.layers.Add()

        """
        self.layernorm1 = \
            tf.keras.layers.LayerNormalization(
                axis=-1, center=True, scale=True
            )
        """

        self.dense1 = \
            tf.keras.layers.Dense(
                units=config.hidden_dim * 2,
                activation='relu',
            )

        self.dense2 = \
            tf.keras.layers.Dense(
                units=config.hidden_dim,
                activation=None,
            )

        self.dropoout1 = tf.keras.layers.Dropout(rate=self.config.dropout_rate)

        self.add2 = tf.keras.layers.Add()

        """
        self.layernorm2 = \
            tf.keras.layers.LayerNormalization(
                axis=-1, center=True, scale=True
            )
        """

    @tf.function
    def call(self, inputs, mask=None, training=True):
        # inputs: (None,n,hidden_dim)=(None,15,256)
        # mask: (None,n)=(None,15), bool

        attention_mask = make_attention_mask(mask)

        x, score = \
            self.mha1(
                query=inputs,
                key=inputs,
                value=inputs,
                attention_mask=attention_mask,
                return_attention_scores=True,
            )  # (None,n,hidden_dim),(None,num_heads,n,n)=(None,15,256),(None,2,15,15)

        x1 = self.add1([inputs, x])  # (None,n,hidden_dim)=(None,15,256)

        # x1 = self.layernorm1(x1)

        x2 = self.dense1(x1)  # (None,n,hidden_dim)=(None,15,512)

        x2 = self.dense2(x2)  # (None,n,hidden_dim)=(None,15,256)

        x2 = self.dropoout1(x2, training=training)

        features = self.add2([x1, x2])  # (None,n,hidden_dim)=(None,15,256)

        # features = self.layernorm2(features)

        return features, score

    def build_graph(self, mask, idx):
        x = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.hidden_dim)
        )

        model = tf.keras.models.Model(
            inputs=[x],
            outputs=self.call(x, mask, training=True),
            name='mha_' + str(idx),
        )

        return model


class MaskBlock(tf.keras.models.Model):
    """
    Model: "mask_block"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
    =================================================================
    Total params: 0
    Trainable params: 0
    """

    def __init__(self):
        super(MaskBlock, self).__init__()

    @tf.function
    def call(self, features, mask):
        """
        :param features: (b,n,hidden_dim) or (b,n,emb_dim)
        :param mask: (b,n)
        :return: features, (b,n,hidden_dim) or (b,n,emb_dim)
        """

        broadcast_float_mask = \
            tf.expand_dims(
                tf.cast(mask, 'float32'),
                axis=-1
            )  # (1,15,1)

        features = features * broadcast_float_mask  # (1,15,256)

        return features  # (None,15,256)


class EncoderBlock(tf.keras.models.Model):
    """
    Model: "encoder_block"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     cnn_block_1 (CNNBlock)      multiple                  258944

     dropout_1 (Dropout)         multiple                  0

     add_2 (Add)                 multiple                  0

     mask_block (MaskBlock)      multiple                  0

     self_attention_block_1 (Sel  multiple                 394624
     fAttentionBlock)

     mask_block_1 (MaskBlock)    multiple                  0

    =================================================================
    Total params: 653,568
    Trainable params: 653,568
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config):
        super(EncoderBlock, self).__init__()

        self.config = config

        self.pos_encoding = PositionalEncoding(
            seq_len=self.config.max_num_red_agents,
            depth=self.config.hidden_dim
        )

        self.cnn = CNNBlock(config=self.config)

        self.dropout = tf.keras.layers.Dropout(rate=self.config.dropout_rate)

        self.add1 = tf.keras.layers.Add()

        self.mask_blk1 = MaskBlock()

        self.self_mha = SelfAttentionBlock(config=self.config)

        self.mask_blk2 = MaskBlock()

    @tf.function
    def call(self, x, mask, training=True):
        # x: (b,n,g,g,ch*n_frames)
        # mask: (b,n)

        # position encoding block
        positions = \
            tf.expand_dims(self.pos_encoding.pos_encoding, axis=0)  # (1,n,hidden_dim)=(1,15,256)

        # cnn block
        features = self.cnn(x)  # (b,n,hidden_dim)=(b,15,256)

        features = self.dropout(features)  # (b,15,256)

        # add (with relative scale of features and pos_encoding) & mask process
        features = self.add1([features * tf.math.sqrt(tf.cast(self.config.hidden_dim, tf.float32)),
                              positions])  # (b,15,256)

        features = self.mask_blk1(features, mask)  # (b,15,256)

        # self-attention block, output query
        query, score1 = self.self_mha(features,
                                      mask,
                                      training=training)
        # (b,n,hidden_dim)=(b,15,256), (b,2,n,n)=(b,2,15,15)

        query = self.mask_blk2(query, mask)  # (b,15,256)

        return query, score1


class AttentionBlock(tf.keras.models.Model):
    """
    Attention block of the decoder

    Model: "attention_block"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     multi_head_attention_2 (Mul  multiple                 33216
     tiHeadAttention)

     add_5 (Add)                 multiple                  0

     mask_block_3 (MaskBlock)    multiple                  0

     multi_head_attention_3 (Mul  multiple                 82560
     tiHeadAttention)

     add_6 (Add)                 multiple                  0

     mask_block_4 (MaskBlock)    multiple                  0

     dense_6 (Dense)             multiple                  65792

     add_7 (Add)                 multiple                  0

    =================================================================
    Total params: 181,568
    Trainable params: 181,568
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config):
        super(AttentionBlock, self).__init__()

        self.config = config

        self.causal_self_attenion = tf.keras.layers.MultiHeadAttention(
            num_heads=self.config.num_heads,
            key_dim=self.config.key_dim
        )

        self.add1 = tf.keras.layers.Add()

        self.mask_blk1 = MaskBlock()

        self.causal_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.config.num_heads,
            key_dim=self.config.key_dim
        )

        self.add2 = tf.keras.layers.Add()

        self.mask_blk2 = MaskBlock()

        self.mlp = tf.keras.layers.Dense(
            units=self.config.hidden_dim,
            activation='relu'
        )

        self.add3 = tf.keras.layers.Add()

    @tf.function
    def call(self, x, query, mask, causal_mask, training=True):
        """
        :param x: (batch,n,key_dim)=(None,15,64)
        :param query: (batch,n,hidden_dim)=(None,15,256)
        :param mask: (batch,n), bool
        :param causal_mask: (batch,n,n), bool
        :return: (batch,n,hidden_dim)=(None,15,256)
        """

        # Make causal_attention mask
        alive_mask = make_attention_mask(mask)  # (b,n,n)=(1,15,15), bool
        alive_mask = tf.cast(alive_mask, dtype=tf.int32)  # (b,n,n)=(1,15,15), int32

        causal_mask = tf.cast(causal_mask, dtype=tf.int32)  # (b,n,n)=(1,15,15), int32

        causal_attention_mask = alive_mask * causal_mask  # (b,n,n)=(1,15,15), int32
        causal_attention_mask = \
            tf.cast(causal_attention_mask, dtype=tf.bool)  # (b,n,n)=(1,15,15), ool

        """ Causal Self-attention + Skip connection """
        feat1, score_causal_self_att = \
            self.causal_self_attenion(
                query=x,
                key=x,
                value=x,
                attention_mask=causal_attention_mask,
                return_attention_scores=True
            )
        # (None,n,key_dim),(None,num_heads,n,n)=(None,15,64),(None,2,15,15)

        feat2 = self.add1([x, feat1])  # (None,n,key_dim)=(None,15,64)

        feat2 = self.mask_blk1(feat2, mask)  # (None,n,key_dim)=(None,15,64)

        """ Causal attention + Skip connection """
        feat3, score_causal_att = \
            self.causal_attention(
                query=query,
                key=feat2,
                value=feat2,
                attention_mask=causal_attention_mask,
                return_attention_scores=True
            )
        # (None,n,hidden_dim),(None,num_heads,n,n)=(None,15,256),(None,2,15,15)

        feat4 = self.add2([feat3, query])  # (None,n,hidden_dim)=(None,15,256)

        feat4 = self.mask_blk2(feat4, mask)  # (None,n,hidden_dim)=(None,15,256)

        """ mlp + skip connection """
        feat5 = self.mlp(feat4)  # (None,n,hidden_dim)=(None,15,256)
        feat6 = self.add3([feat5, feat4])  # (None,n,hidden_dim)=(None,15,256)

        return feat6, [score_causal_self_att, score_causal_att]
        # (None,n,hidden_dim)=(None,15,256)
        # [(None,num_heads,n,n),(None,num_heads,n,n)]=[(None,2,15,15),(None,2,15,15)]


class QLogitsBlock(tf.keras.models.Model):
    """
    Very simple dense model, output is logits

    :param action_dim=5
    :param hidden_dim=256
    :param max_num_agents=15=n
    :return: (None,n,action_dim)=(None,15,5)

    Model: "q_logits_block"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     time_distributed_12 (TimeDi  multiple                 197376
     stributed)

     dropout_3 (Dropout)         multiple                  0

     time_distributed_13 (TimeDi  multiple                 196864
     stributed)

     time_distributed_14 (TimeDi  multiple                 1285
     stributed)

    =================================================================
    Total params: 395,525
    Trainable params: 395,525
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(QLogitsBlock, self).__init__(**kwargs)

        self.config = config

        self.dense1 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.hidden_dim * 3,
                    activation='relu',
                )
            )

        self.dropoout1 = tf.keras.layers.Dropout(rate=self.config.dropout_rate)

        self.dense2 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.hidden_dim,
                    activation='relu',
                )
            )

        self.dense3 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.action_dim,
                    activation=None,
                )
            )

    @tf.function
    def call(self, inputs, mask, training=True):
        # inputs: (None,n,hidden_dim)=(None,15,256)
        # mask: (None,n)=(None,15), bool

        x1 = self.dense1(inputs)  # (None,n,hidden_dim*3)

        x1 = self.dropoout1(x1, training=training)

        x1 = self.dense2(x1)  # (None,n,hidden_dim)

        logits = self.dense3(x1)  # (None,n,action_dim)

        return logits

    def build_graph(self, mask):
        x = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.hidden_dim)
        )  # (None,n,256)

        model = tf.keras.models.Model(
            inputs=[x],
            outputs=self.call(x, mask),
            name='q_net'
        )

        return model


class DecoderBlock(tf.keras.models.Model):
    """
    Model: "decoder_block"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     dense_6 (Dense)             multiple                  448

     add_5 (Add)                 multiple                  0

     mask_block_2 (MaskBlock)    multiple                  0

     attention_block (AttentionB  multiple                 181568
     lock)

     mask_block_5 (MaskBlock)    multiple                  0

     q_logits_block (QLogitsBloc  multiple                 395525
     k)

     mask_block_6 (MaskBlock)    multiple                  0

    =================================================================
    Total params: 577,541
    Trainable params: 577,541
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config):
        super(DecoderBlock, self).__init__()

        self.config = config

        self.pos_encoding = PositionalEncoding(
            seq_len=self.config.max_num_red_agents,
            depth=self.config.key_dim
        )

        self.action_emb = \
            tf.keras.layers.Dense(
                units=self.config.key_dim,
                activation='linear'
            )
        self.add1 = tf.keras.layers.Add()
        self.mask_blk1 = MaskBlock()

        self.attention_blk1 = AttentionBlock(self.config)
        self.mask_blk2 = MaskBlock()

        self.qlogits = QLogitsBlock(self.config)
        self.mask_blk3 = MaskBlock()

    @tf.function
    def call(self, x, query, mask, training=True):
        """
        :param x: [a_0, ..., a_n-1], (b,n)=(1,15), int
        :param query: (b,n,hidden_dim)=(1,15,256)
        :param mask: (b,n)=(1,15), bool
        :return: Q-logits, (b,n,action_dim)
        """

        # Position encoding block
        positions = \
            tf.expand_dims(self.pos_encoding.pos_encoding, axis=0)  # (1,n,key_dim)=(1,15,64)

        # One_hot action
        x = tf.one_hot(indices=x, depth=self.config.action_dim+1)  # (b,n,action_dim+1)=(1,15,6)

        # Input action sequence encoding
        features = self.action_emb(x)  # (b,n,key_dim)=(1,15,64)

        # add (with relative scale of features and pos_encoding) & mask process
        features = self.add1([features * tf.math.sqrt(tf.cast(self.config.key_dim, tf.float32)),
                              positions])  # (b,15,64)
        features = self.mask_blk1(features, mask)  # (b,n,key_dim)=(1,15,64)

        # Make causal mask
        batch_size = features.shape[0]
        num_agents = self.config.max_num_red_agents
        causal_mask = make_causal_mask(batch_size, num_agents)  # (b,n,n)=(1,15,15), bool

        # Attention block 1
        features1, scores1 = \
            self.attention_blk1(
                features,
                query,
                mask,
                causal_mask,
                training=training
            )
        # (b,n,hidden_dim)=(1,15,256)
        # [(None,num_heads,n,n),(None,num_heads,n,n)]=[(None,2,15,15),(None,2,15,15)]

        features = self.mask_blk2(features1, mask)  # (b,n,hidden_dim)=(1,15,256)

        # Q-logits
        qlogits = self.qlogits(features, mask, training=training)  # (b,n,action_dim)=(1,15,5)
        qlogits = self.mask_blk3(qlogits, mask)  # (b,n,action_dim)=(1,15,5)

        return qlogits, scores1  # (b,n,action_dim)=(1,15,5)


def main():
    dir_name = './models_architecture'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    config = Config()

    grid_size = config.grid_size
    ch = config.observation_channels
    n_frames = config.n_frames

    obs_shape = (grid_size, grid_size, ch * n_frames)

    max_num_agents = config.max_num_red_agents

    # Define alive_agents_ids & raw_obs
    alive_agents_ids = [0, 2]
    raw_obs = []

    for a in alive_agents_ids:
        obs_a = np.random.rand(obs_shape[0], obs_shape[1], obs_shape[2])
        raw_obs.append(obs_a)

    # Get padded_obs and mask
    padded_obs = make_padded_obs(max_num_agents, obs_shape, raw_obs)  # (1,n,g,g,ch*n_frames)

    mask = make_mask(alive_agents_ids, max_num_agents)  # (1,n)

    # Make input action sequence
    input_actions = [0] * config.max_num_red_agents
    input_actions[0] = 5
    input_actions[1] = 3
    input_actions[2] = 1

    input_actions = np.array(input_actions, dtype=np.int32)  # (n,)
    input_actions = np.expand_dims(input_actions, axis=0)  # (b,n)=(1,15)

    """ cnn_model """
    cnn = CNNBlock(config=config)

    features_cnn = cnn(padded_obs)  # Build, (1,n,hidden_dim)

    """ remove tf.function for summary """
    """
    cnn.build_graph(mask).summary()

    tf.keras.utils.plot_model(
        cnn.build_graph(mask),
        to_file=dir_name + '/cnn_model',
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )
    """

    """ position encoding """
    pos_encoding = PositionalEncoding(seq_len=config.max_num_red_agents, depth=config.key_dim)

    pos_encoding.pos_encoding

    """ mha model """
    mha = SelfAttentionBlock(config=config)

    features_mha, score = mha(features_cnn, mask)  # Build, (None,n,hidden_dim),(1,num_heads,n,n)

    """ remove tf.function for summary """
    """
    idx = 1
    mha.build_graph(mask, idx).summary()

    tf.keras.utils.plot_model(
        mha.build_graph(mask, idx),
        to_file=dir_name + '/mha_model_' + str(idx),
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )
    """

    """ encoder block """
    encoder = EncoderBlock(config=config)

    query, enc_score = encoder(x=padded_obs, mask=mask, training=True)

    """ decoder block """
    decoder = DecoderBlock(config)
    qlogits, dec_scores = decoder(input_actions, query, mask)  # (b,n,action_dim)=(1,15,5)

    print(qlogits[0, :, :])


if __name__ == '__main__':
    main()

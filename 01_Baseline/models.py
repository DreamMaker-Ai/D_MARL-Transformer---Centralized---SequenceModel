import numpy as np
import tensorflow as tf
import os

from config import Config
from sub_models import EncoderBlock, DecoderBlock
from utils_transformer import make_mask, make_padded_obs, make_causal_mask, make_attention_mask


class MarlTransformerSequenceModel(tf.keras.models.Model):
    """
    :inputs: padded state (None,n,g,g,ch*n_frames)=(None,15,15,15,6), n=max_num_agents=15
             input_actions (None,n)=(None,15), int32
             mask (None,n)=(None,15), bool
    :return: Q logits (None,n,action_dim)=(None,15,5),
             scores [enc_score, dec_scores]
                enc_score: (None,num_heads,n,n), num_heads=2, n=15
                dec_scores: [(None,num_heads,n,n),(None,num_heads,n,n)]

    Model: "marl_transformer_sequence_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     encoder_block (EncoderBlock  multiple                 657408
     )

     decoder_block (DecoderBlock  multiple                 578437
     )

    =================================================================
    Total params: 1,235,845
    Trainable params: 1,235,845
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(MarlTransformerSequenceModel, self).__init__(**kwargs)

        self.config = config

        self.encoder = EncoderBlock(config=config)

        self.decoder = DecoderBlock(config=config)

    @tf.function
    def call(self, padded_state, input_actions, mask, training=True):
        query, enc_score = self.encoder(padded_state, mask, training)

        q_logits, dec_scores = self.decoder(input_actions, query, mask, training)

        return q_logits, [enc_score, dec_scores]


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

    """ Make model """
    marl_transformer = MarlTransformerSequenceModel(config=config)

    q_logits, scores = marl_transformer(padded_obs, input_actions, mask, training=True)

    marl_transformer.encoder.summary()
    print('\n')
    """
    Model: "encoder_block"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding (Embedding)       multiple                  3840      
                                                                     
     cnn_block (CNNBlock)        multiple                  258944    
                                                                     
     dropout (Dropout)           multiple                  0         
                                                                     
     mask_block (MaskBlock)      multiple                  0         
                                                                     
     self_attention_block (SelfA  multiple                 394624    
     ttentionBlock)                                                  
                                                                     
     mask_block_1 (MaskBlock)    multiple                  0         
                                                                     
    =================================================================
    Total params: 657,408
    Trainable params: 657,408
    Non-trainable params: 0
    _________________________________________________________________
    """

    marl_transformer.decoder.summary()
    print('\n')
    """
    Model: "decoder_block"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding_1 (Embedding)     multiple                  960       
                                                                     
     embedding_2 (Embedding)     multiple                  384       
                                                                     
     add_2 (Add)                 multiple                  0         
                                                                     
     mask_block_2 (MaskBlock)    multiple                  0         
                                                                     
     attention_block (AttentionB  multiple                 181568    
     lock)                                                           
                                                                     
     mask_block_5 (MaskBlock)    multiple                  0         
                                                                     
     q_logits_block (QLogitsBloc  multiple                 395525    
     k)                                                              
                                                                     
     mask_block_6 (MaskBlock)    multiple                  0         
                                                                     
    =================================================================
    Total params: 578,437
    Trainable params: 578,437
    Non-trainable params: 0
    _________________________________________________________________
    """

    marl_transformer.summary()


if __name__ == '__main__':
    main()

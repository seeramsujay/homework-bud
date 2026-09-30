from collections import namedtuple

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

from tf_utils import dense_layer, shape


LSTMAttentionCellState = namedtuple(
    'LSTMAttentionCellState',
    ['h1', 'c1', 'h2', 'c2', 'h3', 'c3', 'alpha', 'beta', 'kappa', 'w', 'phi']
)


class LSTMAttentionCell(tf.nn.rnn_cell.RNNCell):

    def __init__(
        self,
        lstm_size,
        num_attn_mixture_components,
        attention_values,
        attention_values_lengths,
        num_output_mixture_components,
        bias,
        reuse=None,
    ):
        self.reuse = reuse
        self.lstm_size = lstm_size
        self.num_attn_mixture_components = num_attn_mixture_components
        self.attention_values = attention_values
        self.attention_values_lengths = attention_values_lengths
        self.window_size = shape(self.attention_values, 2)
        self.char_len = tf.shape(attention_values)[1]
        self.batch_size = tf.shape(attention_values)[0]
        self.num_output_mixture_components = num_output_mixture_components
        self.output_units = 6*self.num_output_mixture_components + 1
        self.bias = bias

    @property
    def state_size(self):
        return LSTMAttentionCellState(
            self.lstm_size,
            self.lstm_size,
            self.lstm_size,
            self.lstm_size,
            self.lstm_size,
            self.lstm_size,
            self.num_attn_mixture_components,
            self.num_attn_mixture_components,
            self.num_attn_mixture_components,
            self.window_size,
            self.char_len,
        )

    @property
    def output_size(self):
        return self.lstm_size

    def zero_state(self, batch_size, dtype):
        return LSTMAttentionCellState(
            tf.zeros([batch_size, self.lstm_size]),
            tf.zeros([batch_size, self.lstm_size]),
            tf.zeros([batch_size, self.lstm_size]),
            tf.zeros([batch_size, self.lstm_size]),
            tf.zeros([batch_size, self.lstm_size]),
            tf.zeros([batch_size, self.lstm_size]),
            tf.zeros([batch_size, self.num_attn_mixture_components]),
            tf.zeros([batch_size, self.num_attn_mixture_components]),
            tf.zeros([batch_size, self.num_attn_mixture_components]),
            tf.zeros([batch_size, self.window_size]),
            tf.zeros([batch_size, self.char_len]),
        )

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__, reuse=tf.AUTO_REUSE):

            # lstm 1
            s1_in = tf.concat([state.w, inputs], axis=1)
            cell1 = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
            s1_out, s1_state = cell1(s1_in, state=(state.c1, state.h1))

            # attention
            attention_inputs = tf.concat([state.w, inputs, s1_out], axis=1)
            attention_params = dense_layer(attention_inputs, 3*self.num_attn_mixture_components, scope='attention')
            alpha, beta, kappa = tf.split(tf.nn.softplus(attention_params), 3, axis=1)
            kappa = state.kappa + kappa / 25.0
            beta = tf.clip_by_value(beta, .01, np.inf)

            kappa_flat, alpha_flat, beta_flat = kappa, alpha, beta
            kappa, alpha, beta = tf.expand_dims(kappa, 2), tf.expand_dims(alpha, 2), tf.expand_dims(beta, 2)

            enum = tf.reshape(tf.range(self.char_len), (1, 1, self.char_len))
            u = tf.cast(tf.tile(enum, (self.batch_size, self.num_attn_mixture_components, 1)), tf.float32)
            phi_flat = tf.reduce_sum(alpha*tf.exp(-tf.square(kappa - u) / beta), axis=1)

            phi = tf.expand_dims(phi_flat, 2)
            sequence_mask = tf.cast(tf.sequence_mask(self.attention_values_lengths, maxlen=self.char_len), tf.float32)
            sequence_mask = tf.expand_dims(sequence_mask, 2)
            w = tf.reduce_sum(phi*self.attention_values*sequence_mask, axis=1)

            # lstm 2
            s2_in = tf.concat([inputs, s1_out, w], axis=1)
            cell2 = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
            s2_out, s2_state = cell2(s2_in, state=(state.c2, state.h2))

            # lstm 3
            s3_in = tf.concat([inputs, s2_out, w], axis=1)
            cell3 = tf.nn.rnn_cell.LSTMCell(self.lstm_size)
            s3_out, s3_state = cell3(s3_in, state=(state.c3, state.h3))

            new_state = LSTMAttentionCellState(
                s1_state.h,
                s1_state.c,
                s2_state.h,
                s2_state.c,
                s3_state.h,
                s3_state.c,
                alpha_flat,
                beta_flat,
                kappa_flat,
                w,
                phi_flat,
            )

            return s3_out, new_state

    def output_function(self, state):
        params = dense_layer(state.h3, self.output_units, scope='gmm', reuse=tf.AUTO_REUSE)
        pis, mus, sigmas, rhos, es = self._parse_parameters(params)
        mu1, mu2 = tf.split(mus, 2, axis=1)
        sigma1, sigma2 = tf.split(sigmas, 2, axis=1)

        # Standard 2D Gaussian sampling via reparameterization trick
        # z1, z2 ~ Normal(0, 1)
        z1 = tf.random.normal(tf.shape(mu1))
        z2 = tf.random.normal(tf.shape(mu2))
        x1 = mu1 + sigma1 * z1
        x2 = mu2 + sigma2 * (rhos * z1 + tf.sqrt(tf.maximum(1.0 - tf.square(rhos), 1e-6)) * z2)
        sampled_coords = tf.stack([x1, x2], axis=2)  # [batch, components, 2]

        # Categorical sampling for mixture component
        # Gumbel-max trick for sampling from categorical distribution without tf.contrib
        gumbel_noise = -tf.log(-tf.log(tf.random.uniform(tf.shape(pis), minval=1e-6, maxval=1.0) + 1e-8) + 1e-8)
        sampled_idx = tf.argmax(tf.log(pis + 1e-8) + gumbel_noise, axis=1)

        # Bernoulli sampling for end of stroke
        rand_u = tf.random.uniform(tf.shape(es))
        sampled_e = tf.cast(rand_u < es, tf.float32)

        idx = tf.stack([tf.range(self.batch_size), tf.cast(sampled_idx, tf.int32)], axis=1)
        coords = tf.gather_nd(sampled_coords, idx)
        return tf.concat([coords, sampled_e], axis=1)

    def termination_condition(self, state):
        char_idx = tf.cast(tf.argmax(state.phi, axis=1), tf.int32)
        final_char = char_idx >= self.attention_values_lengths - 1
        past_final_char = char_idx >= self.attention_values_lengths
        output = self.output_function(state)
        es = tf.cast(output[:, 2], tf.int32)
        is_eos = tf.equal(es, np.ones_like(es))
        return tf.logical_or(tf.logical_and(final_char, is_eos), past_final_char)

    def _parse_parameters(self, gmm_params, eps=1e-8, sigma_eps=1e-4):
        pis, sigmas, rhos, mus, es = tf.split(
            gmm_params,
            [
                1*self.num_output_mixture_components,
                2*self.num_output_mixture_components,
                1*self.num_output_mixture_components,
                2*self.num_output_mixture_components,
                1
            ],
            axis=1
        )

        pis = tf.nn.softmax(pis * (1 + self.bias))
        sigmas = tf.exp(sigmas - self.bias) + sigma_eps
        rhos = tf.clip_by_value(tf.tanh(rhos), -1.0 + eps, 1.0 - eps)
        es = tf.clip_by_value(tf.nn.sigmoid(es)*(1 + self.bias), eps, 1.0 - eps)

        return pis, mus, sigmas, rhos, es

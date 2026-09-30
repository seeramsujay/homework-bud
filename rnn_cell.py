from collections import namedtuple

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

from tf_utils import dense_layer, shape


LSTMAttentionCellState = namedtuple(
    'LSTMAttentionCellState',
    ['h1', 'c1', 'h2', 'c2', 'h3', 'c3', 'alpha', 'beta', 'kappa', 'w', 'phi']
)


def custom_lstm_step(inputs, h, c, scope_name, input_dim, units=400):
    """
    Exact mathematical equivalent of standard LSTMCell.
    Uses variable names 'kernel' and 'bias' under scope_name so pre-trained weights
    load identically without any dependence on tf.nn.rnn_cell.RNNCell or Keras 3.
    gate order in TF1 LSTMCell: [i, j, f, o]
    """
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        kernel = tf.get_variable(
            name='kernel',
            shape=[input_dim + units, 4 * units],
            dtype=tf.float32
        )
        bias = tf.get_variable(
            name='bias',
            shape=[4 * units],
            dtype=tf.float32,
            initializer=tf.zeros_initializer()
        )
        concat_in = tf.concat([inputs, h], axis=1)
        gates = tf.matmul(concat_in, kernel) + bias
        i, j, f, o = tf.split(gates, 4, axis=1)
        new_c = tf.nn.sigmoid(f + 1.0) * c + tf.nn.sigmoid(i) * tf.nn.tanh(j)
        new_h = tf.nn.sigmoid(o) * tf.nn.tanh(new_c)
        return new_h, new_c


class LSTMAttentionCell:

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

            # lstm 1 (scope: lstm_cell)
            s1_in = tf.concat([state.w, inputs], axis=1)
            # input_dim = window_size (73) + inputs (3) = 76
            s1_h, s1_c = custom_lstm_step(s1_in, state.h1, state.c1, 'lstm_cell', input_dim=76, units=self.lstm_size)

            # attention
            attention_inputs = tf.concat([state.w, inputs, s1_h], axis=1)
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

            # lstm 2 (scope: lstm_cell_1)
            # inputs (3) + s1_h (400) + w (73) = 476
            s2_in = tf.concat([inputs, s1_h, w], axis=1)
            s2_h, s2_c = custom_lstm_step(s2_in, state.h2, state.c2, 'lstm_cell_1', input_dim=476, units=self.lstm_size)

            # lstm 3 (scope: lstm_cell_2)
            # inputs (3) + s2_h (400) + w (73) = 476
            s3_in = tf.concat([inputs, s2_h, w], axis=1)
            s3_h, s3_c = custom_lstm_step(s3_in, state.h3, state.c3, 'lstm_cell_2', input_dim=476, units=self.lstm_size)

            new_state = LSTMAttentionCellState(
                s1_h,
                s1_c,
                s2_h,
                s2_c,
                s3_h,
                s3_c,
                alpha_flat,
                beta_flat,
                kappa_flat,
                w,
                phi_flat,
            )

            return s3_h, new_state

    def output_function(self, state):
        params = dense_layer(state.h3, self.output_units, scope='gmm', reuse=tf.AUTO_REUSE)
        pis, mus, sigmas, rhos, es = self._parse_parameters(params)
        mu1, mu2 = tf.split(mus, 2, axis=1)
        sigma1, sigma2 = tf.split(sigmas, 2, axis=1)

        # Standard 2D Gaussian sampling via reparameterization trick
        z1 = tf.random.normal(tf.shape(mu1))
        z2 = tf.random.normal(tf.shape(mu2))
        x1 = mu1 + sigma1 * z1
        x2 = mu2 + sigma2 * (rhos * z1 + tf.sqrt(tf.maximum(1.0 - tf.square(rhos), 1e-6)) * z2)
        sampled_coords = tf.stack([x1, x2], axis=2)

        # Categorical sampling for mixture component
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

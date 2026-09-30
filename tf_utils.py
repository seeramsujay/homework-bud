import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def dense_layer(inputs, output_units, bias=True, activation=None, batch_norm=None,
                dropout=None, scope='dense-layer', reuse=False):
    """
    Applies a dense layer to a 2D tensor of shape [batch_size, input_units]
    to produce a tensor of shape [batch_size, output_units].
    """
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=tf.variance_scaling_initializer(),
            shape=[shape(inputs, -1), output_units]
        )
        z = tf.matmul(inputs, W)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b

        if batch_norm is not None:
            z = tf.layers.batch_normalization(z, training=batch_norm, reuse=reuse)

        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        return z


def time_distributed_dense_layer(
        inputs, output_units, bias=True, activation=None, batch_norm=None,
        dropout=None, scope='time-distributed-dense-layer', reuse=False):
    """
    Applies a shared dense layer to each timestep of a tensor of shape
    [batch_size, max_seq_len, input_units] to produce a tensor of shape
    [batch_size, max_seq_len, output_units].
    """
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=tf.variance_scaling_initializer(),
            shape=[shape(inputs, -1), output_units]
        )
        z = tf.einsum('ijk,kl->ijl', inputs, W)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b

        if batch_norm is not None:
            z = tf.layers.batch_normalization(z, training=batch_norm, reuse=reuse)

        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        return z


def shape(tensor, dim=None):
    """Finds the static or dynamic shape of a tensor."""
    if dim is None:
        return [shape(tensor, i) for i in range(len(tensor.shape))]
    dim_val = tensor.shape[dim]
    return dim_val if dim_val is not None else tf.shape(tensor)[dim]

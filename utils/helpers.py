
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def shape_(inputs, name=None):
    name = "shape" if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return shape


def norm(tensor, ord='euclidean', axis=None, keepdims=None, name=None):
    try:
        return tf.norm(tensor, ord=ord, axis=axis, keepdims=keepdims, name=name)
    except:
        return tf.norm(tensor, ord=ord, axis=axis, keep_dims=keepdims, name=name)


def transforming(inputs, num_outputs, out_caps_dims, name=None):
    """
    Args:
        inputs: A 4-D or 6-D tensor, [batch_size, num_inputs] + in_caps_dims or [batch_size, height, width, channels] + in_caps_dims.
        num_outputs: Integer, the number of output capsules.
        out_caps_dims: A list of 2 integers. The dimensions of output capsule, e.g. out_caps_dims=[4, 4].
        name: String, a name for this operation.

    Returns:
        votes: A 5-D or 7-D tensor, [batch_size, num_inputs, num_outputs] + out_caps_dims or [batch_size, height, width, channels, num_outputs] + out_caps_dims.
    """
    name = "transforming" if name is None else name
    with tf.variable_scope(name) as scope:
        input_shape = shape_(inputs)
        prefix_shape = [1 for i in range(len(input_shape) - 3)] + input_shape[-3:-2] + [num_outputs]
        in_caps_dims = input_shape[-2:]
        if in_caps_dims[0] == out_caps_dims[1]:
            shape = prefix_shape + [out_caps_dims[0], 1, in_caps_dims[1]]
            expand_axis = -3
            reduce_sum_axis = -1
        elif in_caps_dims[1] == out_caps_dims[0]:
            shape = prefix_shape + [in_caps_dims[0], 1, out_caps_dims[1]]
            expand_axis = -1
            reduce_sum_axis = -3
        elif in_caps_dims[0] == out_caps_dims[0]:
            shape = prefix_shape + [1, out_caps_dims[1], in_caps_dims[1]]
            expand_axis = -2
            reduce_sum_axis = -1
        elif in_caps_dims[1] == out_caps_dims[1]:
            shape = prefix_shape + [in_caps_dims[0], out_caps_dims[0], 1]
            expand_axis = -2
            reduce_sum_axis = -3
        else:
            raise TypeError("out_caps_dims must have at least one value being the same with the in_caps_dims")
        in_pose = tf.expand_dims(inputs, axis=-3)
        ones = tf.ones(shape=prefix_shape + [1, 1])
        in_pose = tf.expand_dims(in_pose * ones, axis=expand_axis)
        transform_mat = tf.get_variable("transformation_matrix", shape=shape)
        votes = tf.reduce_sum(in_pose * transform_mat, axis=reduce_sum_axis)

        return votes


def space_to_batch_nd(input, kernel_size, strides, name=None):
    """ Space to batch with strides. Different to tf.space_to_batch_nd.
        for convCapsNet model: memory 4729M, speed 0.165 sec/step, similiar to space_to_batch_nd_v1
    Args:
        input: A Tensor. N-D with shape input_shape = [batch] + spatial_shape + remaining_shape, where spatial_shape has M dimensions.
        kernel_size: A sequence of len(spatial_shape)-D positive integers specifying the spatial dimensions of the filters.
        strides: A sequence of len(spatial_shape)-D positive integers specifying the stride at which to compute output.
    Returns:
        A Tensor.
    """
    assert len(kernel_size) == 3
    assert len(strides) == 3
    name = "space_to_batch_nd" if name is None else name
    with tf.name_scope(name):
        input_shape = shape_(input)
        h_steps = int((input_shape[1] - kernel_size[0]) / strides[0] + 1)
        w_steps = int((input_shape[2] - kernel_size[1]) / strides[1] + 1)
        d_steps = int((input_shape[3] - kernel_size[2]) / strides[2] + 1)
        blocks = []  # each element with shape [batch, h_kernel_size * w_kernel_size * d_kernel_size] + remaining_shape
        for d in range(d_steps):
            d_s = d * strides[2]
            d_e = d_s + kernel_size[2]
            h_blocks = []
            for h in range(h_steps):
                h_s = h * strides[0]
                h_e = h_s + kernel_size[0]
                w_blocks = []
                for w in range(w_steps):
                    w_s = w * strides[1]
                    w_e = w_s + kernel_size[1]
                    block = input[:, h_s:h_e, w_s:w_e, d_s:d_e]
                    # block = tf.reshape(block, shape=[tf.shape(input)[0], np.prod(kernel_size)] + input_shape[4:])
                    w_blocks.append(block)
                h_blocks.append(tf.concat(w_blocks, axis=2))
            blocks.append(tf.concat(h_blocks, axis=1))
        return tf.concat(blocks, axis=0)


def reduce_sum(input_tensor,
               axis=None,
               keepdims=None,
               name=None):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims, name=name)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims, name=name)


def divide(x, y, safe_mode=True, epsilon=None, name=None):
    """ A wrapper of `tf.divide`, computes Python style division of x by y but extends safe divide support.
        If safe_mode is `True` or epsilon is given(a small float number), the absolute value of denominator
        in the division will be clip to make sure it's bigger than epsilon(default is 1e-13).
    Args:
        safe_mode: Use safe divide mode.
        epsilon: Float number. Default is `1e-13`.
    """
    if not safe_mode and epsilon is None:
        return tf.divide(x, y, name=name)
    else:
        epsilon = 1e-20 if epsilon is None else epsilon
        name = "safe_divide" if name is None else name
        with tf.name_scope(name):
            y = tf.where(tf.greater(tf.abs(y), epsilon), y, y + tf.sign(y) * epsilon)
            return tf.divide(x, y)


def log(x, epsilon=1e-20, name=None):
    """ A wrapper of `tf.log`, computes natural logarithm of x element-wise but extends safe log support.
        If epsilon is given as a positive float, x will be clipped to bigger than epsilon before doing computing.
    """
    if isinstance(epsilon, float) and epsilon > 0:
        return tf.log(tf.maximum(x, epsilon), name=name)
    else:
        return tf.log(x, name=name)

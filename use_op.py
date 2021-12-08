import tensorflow as tf
gelu_module = tf.load_op_library('./gelu.so')
print(gelu_module)
gelu_module.gelu([2.0, 3.0])
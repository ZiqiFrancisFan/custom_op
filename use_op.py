import tensorflow as tf
gelu_module = tf.load_op_library('./gelu.so')
print(gelu_module)
gelu_module.gelu_op([2.0, 3.0])
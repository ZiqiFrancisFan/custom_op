import tensorflow as tf
gelu_module = tf.load_op_library('./gelu.so')
print(gelu_module)
print(gelu_module.gelu_op([0.7, 0.5]).numpy())
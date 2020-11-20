import tensorflow as tf

print(tf.__version__)

v1 = tf.constant(8.0, dtype=tf.dtypes.float32)
v2 = tf.constant(2.0, dtype=tf.dtypes.float32)
v3 = tf.math.multiply(v1, v2)

@tf.function  # 这种装饰器写法能加快运行速度
def mm():
    print(f"result = {v3}")

mm()
print(f"result = {v3}")

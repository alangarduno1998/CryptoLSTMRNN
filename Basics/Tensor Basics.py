import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #supresses error message
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#Tensor is a n dimension array of vector

#Initialization of Tensors
x= tf.constant(4.0, shape=(1,1), dtype=tf.float32)
x= tf.constant([[1,2,3],[4,5,6]])

x = tf.ones((3,3))
x = tf.zeros((2, 3))
x = tf.eye(3)
x = tf.random.normal((3,3), mean=0, stddev=1)
x= tf.random.uniform((1,3), minval=0, maxval=1)
x= tf.range(start=1, limit=10, delta=2)
x = tf.cast(x, dtype=tf.float64)
#tf.float (16, 32, 64), tf.int (8,16, 32, 64)

# Mathematical Operations
x = tf.constant([1,2,3])
y = tf.constant([9,8,7])
z = tf.add(x,y)
z = x+y
z = tf.subtract(x,y)
z= x-y
z = tf.multiply(x,y)
z = x*y
z = tf.divide(x,y)
z= x/y
z= tf.tensordot(x,y, axes=1)
z= tf.reduce_sum(x*y, axis=0)
z= tf.matmul(x,y)
print(z)
# Indexing


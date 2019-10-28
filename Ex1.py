import  tensorflow as tf
import numpy as np

graph = tf.Graph()
session = tf.InteractiveSession(graph=graph)

x = tf.placeholder(shape=[1, 10], dtype=tf.float32, name='x')
W = tf.Variable(tf.random_uniform(shape=[10,5], minval=-0.1, maxval=0.1, dtype=tf.float32),name='W') 
# Variable
b = tf.Variable(tf.zeros(shape=[5],dtype=tf.float32),name='b') 

h = tf.nn.sigmoid(tf.matmul(x,W) + b) # Operation to be performed

# Executing operations and evaluating nodes in the graph
tf.global_variables_initializer().run() # Initialize the variables

# Run the operation by providing a value to the symbolic input x
h_eval = session.run(h,feed_dict={x: np.random.rand(1,10)}) 
print(h_eval)
session.close()

  
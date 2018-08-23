import tensorflow as tf

sess = tf.InteractiveSession()
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

################ Function for Jacobian calculation ################
def jacobian_matrix(y_flat, x, num_classes):
    for i in range(num_classes):
        if i==0:
            Jacobian = tf.gradients(y_flat[i],x)
        else:
            Jacobian = tf.concat([Jacobian, tf.gradients(y_flat[i],x)],axis=0)
    return Jacobian

########################## CNN functions ##########################
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
  
######################### GRAPH #########################   
x = tf.placeholder(tf.float32, shape=[None,784])
y_ = tf.placeholder(tf.float32, shape=[None,10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

classification_f = tf.reduce_max(tf.nn.softmax(logits=y_conv))
classification_f = tf.reshape(classification_f,[1])

y_flat = tf.reshape(y_conv, (-1,))

num_classes = 10
Jacobian = jacobian_matrix(y_flat, x, num_classes)
Jacobian_loss = tf.norm(Jacobian) # Frobenius norm is default for matrices

# Adding L2 Regularization
L2_Regularization = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_fc1)

# Loss
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
loss = cross_entropy_loss + (1e-3)*L2_Regularization
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

loss_post_processing = loss + (1e-1)*Jacobian_loss
train_step_post_processing = tf.train.AdamOptimizer(1e-4).minimize(loss_post_processing)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

num_of_iterations = 20001
num_of_post_processing_iterations = 4500
train_batch_size = 500

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  
  for i in range(num_of_iterations + num_of_post_processing_iterations):
    batch = mnist.train.next_batch(train_batch_size) 
    X_to_train = batch[0]
    Y_to_train = batch[1]
        
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x: X_to_train, y_: Y_to_train, keep_prob: 1.0})
      print('step %d, training accuracy %g percent' % (i, 100*train_accuracy))
      curr_Jacobian_loss = sess.run(Jacobian_loss, feed_dict={x: X_to_train, y_: Y_to_train, keep_prob: 1.0})
      print("The current Jacobian loss is ", curr_Jacobian_loss)   
      curr_loss = sess.run(loss, feed_dict={x: X_to_train, y_: Y_to_train, keep_prob: 1.0})
      print("The current general loss is ", curr_loss, "\n")
      
    if i < num_of_iterations:
        train_step.run(feed_dict={x: X_to_train, y_: Y_to_train, keep_prob: 0.5})
    else:
        train_step_post_processing.run(feed_dict={x: X_to_train, y_: Y_to_train, keep_prob: 0.5})

  print('Test accuracy: %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
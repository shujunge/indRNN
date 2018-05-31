#coding:utf-8


import numpy as np
import tensorflow as tf
import csv
from ind_rnn_cell import IndRNNCell
from time import time

# import argparse
#
#
# parser = argparse.ArgumentParser(description='using IndRNNCell to solve the addition problem',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--count', type=int,default=200)
# args = parser.parse_args()


TIME_STEPS = 300
NUM_UNITS = 128
LEARNING_RATE_INIT = 0.0002
LEARNING_RATE_DECAY_STEPS = 20000
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)

BATCH_SIZE = 50


def get_batch():
	"""生成adding problem 数据集,每个批次大小为BATCH_SIZE,序列长度为TIME_STEPS,返回data"""
	add_values = np.random.rand(BATCH_SIZE, TIME_STEPS)
	
	# Build the second sequence with one 1 in each half and 0s otherwise
	add_indices = np.zeros_like(add_values)
	half = int(TIME_STEPS / 2)
	for i in range(BATCH_SIZE):
		first_half = np.random.randint(half)
		second_half = np.random.randint(half, TIME_STEPS)
		add_indices[i, [first_half, second_half]] = 1
	
	# Zip the values and indices in a third dimension:
	# inputs has the shape (batch_size, time_steps, 2)
	inputs = np.dstack((add_values, add_indices))
	targets = np.sum(np.multiply(add_values, add_indices), axis=1)
	return inputs, targets


def indrnn_model(first_input_init, inputs_ph):
	"""indrnn模型:搭建两层indrnn模型,每层神经元的数量为NUM_UNITS，两层总参数为：TIME_STEPS*NUM_UNITS+NUM_UNITS*NUM_UNITS+2*NUM_UNITS
	
	"""
	first_layer = IndRNNCell(NUM_UNITS, recurrent_max_abs=RECURRENT_MAX,
	                         recurrent_kernel_initializer=first_input_init)
	
	second_layer = IndRNNCell(NUM_UNITS, recurrent_max_abs=RECURRENT_MAX)
	
	cell = tf.nn.rnn_cell.MultiRNNCell([first_layer, second_layer])
	# cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_UNITS) uncomment this for LSTM runs
	output, state = tf.nn.dynamic_rnn(cell, inputs_ph, dtype=tf.float32)
	last = output[:, -1, :]
	
	weight = tf.get_variable("softmax_weight", shape=[NUM_UNITS, 1])
	bias = tf.get_variable("softmax_bias", shape=[1],
	                       initializer=tf.constant_initializer(0.1))
	prediction = tf.squeeze(tf.matmul(last, weight) + bias)
	return prediction


def lstm_model(first_input_init, inputs_ph):
	"""lstmrnn模型:搭建两层lstmrnn模型,每层神经元的数量为NUM_UNITS,
	两层总参数为：TIME_STEPS*NUM_UNITS+3*NUM_UNITS
		"""
	first_layer = tf.nn.rnn_cell.LSTMCell(NUM_UNITS)
	second_layer = tf.nn.rnn_cell.LSTMCell(NUM_UNITS)
	
	cell = tf.nn.rnn_cell.MultiRNNCell([first_layer, second_layer])
	# cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_UNITS) uncomment this for LSTM runs
	output, state = tf.nn.dynamic_rnn(cell, inputs_ph, dtype=tf.float32)
	last = output[:, -1, :]
	
	weight = tf.get_variable("softmax_weight", shape=[NUM_UNITS, 1])
	bias = tf.get_variable("softmax_bias", shape=[1],
	                       initializer=tf.constant_initializer(0.1))
	prediction = tf.squeeze(tf.matmul(last, weight) + bias)
	return prediction


def rnn_model(first_input_init, inputs_ph):
	"""lstmrnn模型:搭建两层lstmrnn模型,每层神经元的数量为NUM_UNITS,
	两层总参数为：TIME_STEPS*NUM_UNITS+3*NUM_UNITS,激活函数用relu代替sigmod函数和tanh函数
	"""
	first_layer = tf.nn.rnn_cell.BasicRNNCell(NUM_UNITS, activation=tf.nn.relu)
	second_layer = tf.nn.rnn_cell.BasicRNNCell(NUM_UNITS, activation=tf.nn.relu)
	
	cell = tf.nn.rnn_cell.MultiRNNCell([first_layer, second_layer])
	# cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_UNITS) uncomment this for LSTM runs
	output, state = tf.nn.dynamic_rnn(cell, inputs_ph, dtype=tf.float32)
	last = output[:, -1, :]
	
	weight = tf.get_variable("softmax_weight", shape=[NUM_UNITS, 1])
	bias = tf.get_variable("softmax_bias", shape=[1],
	                       initializer=tf.constant_initializer(0.1))
	prediction = tf.squeeze(tf.matmul(last, weight) + bias)
	return prediction


def main():
	"""adding problem实验主程序,使用３种模型对数据集进行训练和测试,即indrnn,Irnn,lstm网络.
	"""
	# Placeholders for training data
	inputs_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, TIME_STEPS, 2))
	targets_ph = tf.placeholder(tf.float32, shape=BATCH_SIZE)
	
	# Build the graph
	first_input_init = tf.random_uniform_initializer(-RECURRENT_MAX,
	                                                 RECURRENT_MAX)
	
	prediction = rnn_model(first_input_init, inputs_ph)
	
	loss_op = tf.losses.mean_squared_error(tf.squeeze(targets_ph), prediction)
	
	global_step = tf.get_variable("global_step", shape=[], trainable=False,
	                              initializer=tf.zeros_initializer)
	learning_rate = tf.train.exponential_decay(LEARNING_RATE_INIT, global_step,
	                                           LEARNING_RATE_DECAY_STEPS, 0.1,
	                                           staircase=True)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	optimize = optimizer.minimize(loss_op, global_step=global_step)
	isend = False
	# Train the model
	csvfile = open("./result/indrnn_%s_result.csv" % TIME_STEPS, "w")
	writer = csv.writer(csvfile)
	start = time()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		step = 0
		while isend == False:
			losses = []
			for _ in range(100):
				# Generate new input data
				inputs, targets = get_batch()
				loss, _ = sess.run([loss_op, optimize],
				                   {inputs_ph: inputs, targets_ph: targets})
				losses.append(loss)
				writer.writerow([loss])
				if loss <= 5e-2:
					isend = True
					break
				step += 1
			print("Step [x100] {} MSE {}".format(int(step / 100), np.mean(losses)))
		print("*********************************************")
		writer.writerow(["%s" % (time() - start)])
		csvfile.close()


if __name__ == "__main__":
	main()

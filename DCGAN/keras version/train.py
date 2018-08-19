
import glob
import numpy as np 
from scipy import misc
import tensorflow as tf 

from network import *

def train():
	data = []
	for image in glob.glob("data/images/*"):
		image_data = misc.imread(image)
		data.append(image_data)

	input_data = np.array(data)

	input_data = (input_data.astype(np.float(32)) - 127.5)/ 127.5
	g = generator_model()
	d = discriminator_model()

	d_after_g = generator_containing_discriminator(g,d)

	g_optimizer = tf.keras.optimizers.Adam(lr = LEARNING_RATE, beta_1 = BETA_1)
	d_optimizer = tf.keras.optimizers.Adam(lr = LEARNING_RATE, beta_1 = BETA_1)

	g.compile(loss = "binary_crossentropy", optimizer = g_optimizer)
	d_after_g.compile(loss = "binary_crossentropy", optimizer = g_optimizer)
	d.trainable = True
	d.compile(loss = "binary_crossentropy", optimizer = d_optimizer)

	for epoch in range(EPOCH):
		for index in range(int(input_data.shape[0] / BATCH_SIZE)):
			input_batch = input_data[index * BATCH_SIZE, (index + 1) * BATCH_SIZE]

			random_data = np.random.uniform(-1, 1, size = (Batch_SIZE, 100))
			generated_images = g.predict(random_data, verbose = 0)

			input_batch = np.concatenate((input_batch, generated_images))
			output_batch = [1] * BATCH_SIZE + [0] * BATCH_SIZE

			d_loss = d.train_on_batch(input_batch, output_batch)
			d.trainable = False

			random_data = np.random.uniform(-1, 1, size = (BATCH_SIZE, 100))
			g_loss = d_on_g.train_on_batch(random_data, [1] * BATCH_SIZE)

			d.trainable = True

            print("Epch %d Step %d Generator Loss: %f Discriminator Loss: %f" % (epoch, index, g_loss, d_loss))

        if epoch % 10 == 9:
        	g.save_weights("generator_weight", True)
        	d.save_weights("discriminator_weight", True)


if __name__ == "__main__":
	train()

	

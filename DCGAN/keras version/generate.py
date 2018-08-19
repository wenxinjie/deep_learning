
import numpy as np 
from PIL import Image
import tensorflow as tf 

from network import *

def generate():
	g = generator_model()
	g.compile(loss = "binary_crossentropy", optimizer = tf.keras.optimizers.Adam(lr = LEARNING_RATE, beta_1 = BETA_1))

	g.load_weights("generator_model")
	random_data = np.random.uniform(-1, 1, size = (BATCH_SIZE, 100))
	images = g.predict(random_data)

	for i in range(BATCH_SIZE):
		image = images[i] * 127.5 + 127.5
		Image.fromarray(image.astype(np.unit8)).save("image-%s.png" % i)

if __name__ == "__main__":
	generate()
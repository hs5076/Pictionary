from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import random
import threading
import time


global circle
global triangle
global square
global imagelist
global modelscore
global model2score
circle = "/home/wnlab4/Downloads/Circle1.jpg"
square = "/home/wnlab4/Downloads/square1.jpg"
triangle = "/home/wnlab4/Downloads/Triangle1.jpg"

imagelist = [circle, square, triangle]
newitem = threading.Event()
model1w = threading.Event()
model2w = threading.Event()
statusready = threading.Event()
gamestart1 = threading.Event()
gamestart2 = threading.Event()
model1loss = threading.Event()
model2loss = threading.Event()
confirm1 = threading.Event()
confirm2 = threading.Event()
modelscore = 0
model2score = 0
model1w.clear()
model2w.clear()

global model
global class_names
global data
global model2
global class_names2

np.set_printoptions(suppress=True)
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

model2 = load_model("keras_Model2.h5", compile=False)
class_names2 = open("labels2.txt", "r").readlines()

gamestart1.clear()
gamestart2.clear()


def statuscheck():
	global ranimage
	global correct
	global modelscore
	global model2score
	while True:
		if newitem.is_set():
			statusready.set()
			confirm1.wait()
			confirm2.wait()
			ranimage = random.choice(imagelist)
			if ranimage == imagelist[0]:
				correct = 'Circle'
			if ranimage == imagelist[1]:
				correct = 'Square'
			if ranimage == imagelist[2]:
				correct = 'Triangle'
			print('\nNew word: ' + correct)
			time.sleep(3)
			gamestart1.set()
			gamestart2.set()
			newitem.clear()
			continue
		if model1w.is_set():
			gamestart1.clear()
			gamestart2.clear()
			modelscore += 1
			print("model 1 has won, score is \nModel 1: " + str(modelscore) + "\nModel 2: " + str(model2score))
			confirm1.clear()
			confirm2.clear()
			time.sleep(1)
			model1w.clear()
			newitem.set()
			continue
		if model2w.is_set():
			gamestart1.clear()
			gamestart2.clear()
			model2score += 1
			print("model 2 has won, score is \nModel 1: " + str(modelscore) + "\nModel 2: " + str(model2score))
			confirm1.clear()
			confirm2.clear()
			time.sleep(1)
			model2w.clear()
			newitem.set()
			continue
		if model1loss.is_set() & model2loss.is_set():
			model1loss.clear()
			model2loss.clear()
			gamestart1.set()
			gamestart2.set()

def modelgame():
	statusready.wait()
	while True:
		confirm1.set()
		if gamestart1.is_set() == True and model1w.is_set() == False and model2w.is_set() == False:
			time.sleep(1)
			model1loss.clear()
			print('Model 1 acquired word')
			image = Image.open(ranimage).convert("RGB")
			size = (224, 224)
			image = ImageOps.fit(image, size, Image.LANCZOS)
			image_array = np.asarray(image)
			normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
			data[0] = normalized_image_array
			prediction = model.predict(data)
			index = np.argmax(prediction)
			class_name = class_names[index]
			confidence_score = prediction[0][index]
			if confirm1.is_set():
				print('Model 1 prediction is ' + class_name[2:])
			if str(class_name[2:].strip()) != correct:
				print('Model 1 guessed incorrectly')
				gamestart1.clear()
				model1loss.set()
				continue
			if str(class_name[2:].strip()) == correct:
				if model2w.is_set():
					continue
				else:
					model1w.set()
			if model2w.is_set():
				continue
				
def model2game():
	statusready.wait()
	while True:
		confirm2.set()
		if gamestart2.is_set() == True and model1w.is_set() == False and model2w.is_set() == False:
			time.sleep(1)
			model2loss.clear()
			print('Model 2 acquired word')
			image = Image.open(ranimage).convert("RGB")
			size = (224, 224)
			image = ImageOps.fit(image, size, Image.LANCZOS)
			image_array = np.asarray(image)
			normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
			data[0] = normalized_image_array
			prediction = model2.predict(data)
			index = np.argmax(prediction)
			class_name = class_names2[index]
			confidence_score = prediction[0][index]
			if confirm2.is_set():
				print('Model 2 prediction is ' + class_name[2:])
			if str(class_name[2:].strip()) != correct:
				print('Model 2 guessed incorrectly')
				gamestart2.clear()
				model2loss.set()
				continue
			if str(class_name[2:].strip()) == correct:
				if model1w.is_set():
					continue
				else:
					model2w.set()
			if model1w.is_set():
				continue
				
# Print prediction and confidence score
#print("Class:", class_name[2:], end="")
#print("Confidence Score:", confidence_score)

y = threading.Thread(target=modelgame)
z = threading.Thread(target=statuscheck)
j = threading.Thread(target=model2game)

z.start()
j.start()
y.start()

newitem.set()


import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
import random
import time
import sys
import os

circle = "/home/wnlab4/Downloads/Circle1.jpg"
square = "/home/wnlab4/Downloads/square1.jpg"
triangle = "/home/wnlab4/Downloads/Triangle1.jpg"

np.set_printoptions(suppress=True)
class_names = open("labelsedge.txt", "r").readlines()
class_names2 = open("labelsn.txt", "r").readlines()
imagelist = [circle, square, triangle]

n = 0
z = 0

script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_directory)

interpreter1 = tflite.Interpreter(model_path='model_unquant.tflite')
interpreter2 = tflite.Interpreter(model_path='model.tflite')

global vimage
global correct
vimage = random.choice(imagelist)
if vimage == imagelist[0]:
	correct = 'circle'
if vimage == imagelist[1]:
	correct = 'square'
if vimage == imagelist[2]:
	correct = 'triangle'
print(correct)

def modelpred(interpreter, classlist):
	global vimage
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	height = input_details[0]['shape'][1]
	width = input_details[0]['shape'][2]
	img = Image.open(vimage).resize((width, height))
	img = img.convert('RGB')
	if interpreter == interpreter1:
		input_data = np.array(img, dtype=np.float32)  # Convert img to NumPy array
		input_data = np.expand_dims(input_data / 255.0, axis=0) 
	else:
		input_data = np.array(img, dtype=np.uint8)
		input_data = np.expand_dims(input_data, axis=0)
	start_time = time.time()
	interpreter.set_tensor(input_details[0]['index'], input_data)
	interpreter.invoke()
	prediction = interpreter.get_tensor(output_details[0]['index'])
	time_total = time.time() - start_time
	maximum = np.argmax(prediction)
	class_name = classlist[maximum]
	if interpreter == interpreter1:
		confidence_score = prediction[0][maximum]
	else:
		confidence_score = prediction[0][maximum] / 255.0
	if float(confidence_score) <= 0.65:
		return['none', time_total]
	else:
		return[class_name[2:].strip(), time_total]
while True:
	x = modelpred(interpreter1, class_names)
	print('Model 1 : ')
	print(x)
	y = modelpred(interpreter2, class_names2)
	print('Model 2 : ')
	print(y)
	time.sleep(1)
	if x[0] == correct and y[0] != correct:
		print('Model 1 wins')
		break
	elif x[0] != correct and y[0] == correct:
		print('Model 2 wins')
		break
	elif x[0] == correct and y[0] == correct:
		if x[1] < y[1]:
			print('Model 1 wins by time')
			print( 'Model 1 time : ' + str(x[1]) + '\nModel 2 time : ' + str(y[1]))
			break
		else:
			print('Model 2 wins by time')
			print( 'Model 1 time : ' + str(x[1]) + '\nModel 2 time : ' + str(y[1]))
			break
	elif x[0] != correct and y[0] != correct:
		if x[0] == 'none' and y[0] == 'none':
			print('both have submitted none')
			continue
		elif x[0] == 'none' and y[0] != 'none':
			z += 1
		elif x[0] != 'none' and y[0] == 'none':
			n += 1
		elif n == 2:
			print('Model 2 wins by forfeit')
			break
		elif z == 2:
			print('Model 1 wins by forfeit')
			break	

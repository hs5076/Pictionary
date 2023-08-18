import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
import random
import time
import sys
import os
import cv2 as cv

#circle = "/home/wnlab4/Downloads/Circle1.jpg"
#square = "/home/wnlab4/Downloads/square1.jpg"
#triangle = "/home/wnlab4/Downloads/Triangle1.jpg"

#circle = 'https://www.youtube.com/watch?v=zR3wbEudD1I'
#square = 'https://www.youtube.com/watch?v=QwgWhYr8hfA'
#triangle = 'https://www.youtube.com/watch?v=W77eGvscpmU'

circle = '/home/wnlab4/circlevid.webm'
square = '/home/wnlab4/squarevid.webm'
triangle = '/home/wnlab4/trianglevid.webm'

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

vimage = random.choice(imagelist)
if vimage == imagelist[0]:
	correct = 'circle'
if vimage == imagelist[1]:
	correct = 'square'
if vimage == imagelist[2]:
	correct = 'triangle'
print(correct)


def modelpred(interpreter, classlist, minconf, image):
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	height = input_details[0]['shape'][1]
	width = input_details[0]['shape'][2]	
	
	img = image.resize((width, height))
	input_data = np.array(img).reshape(input_details[0]['shape'])/255.0 # reshape eliminates needs for expand_dims
	if input_details[0]['quantization_parameters']['scales'].size > 0: # if it is quantized, divide by scaling parameter
		input_data = np.array(input_data/input_details[0]['quantization_parameters']['scales'])
	input_data = np.array(input_data, dtype = input_details[0]['dtype']) # either way, make sure dtype is correct
	start_time = time.time()
	interpreter.set_tensor(input_details[0]['index'], input_data)
	interpreter.invoke()
	prediction = interpreter.get_tensor(output_details[0]['index'])
	time_total = time.time() - start_time
	maximum = np.argmax(prediction)
	class_name = classlist[maximum]
	confidence_score = prediction[0][maximum]
	if output_details[0]['quantization_parameters']['scales'].size > 0:
		confidence_score = confidence_score * output_details[0]['quantization_parameters']['scales']
	if float(confidence_score) <= minconf:
		return [None, time_total]
	else:
		return[class_name[2:].strip(), time_total]
cap = cv.VideoCapture(vimage)
while True:
	ret,frame = cap.read()
	if ret:
		print('One frame read')
	pil_image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
	x = modelpred(interpreter1, class_names, 0.65, pil_image)
	print('Model 1 : ')
	print(x)
	y = modelpred(interpreter2, class_names2, 0.65, pil_image)
	print('Model 2 : ')
	print(y)
	if x[0] == correct and y[0] != correct:
		print('Model 1 wins')
		break
	elif x[0] != correct and y[0] == correct:
		print('Model 2 wins')
		break
	elif x[0] == correct and y[0] == correct and x[1] < y[1]:
		print('Model 1 wins by time')
		print( 'Model 1 time : ' + str(x[1]) + '\nModel 2 time : ' + str(y[1]))
		break
	elif x[0] == correct and y[0] == correct and x[1] > y[1]:
		print('Model 2 wins by time')
		print( 'Model 1 time : ' + str(x[1]) + '\nModel 2 time : ' + str(y[1]))
		break
	elif x[0] == correct and y[0] == correct and x[1] == y[1]:
		print('Somehow its a tie! Trying again')
		print( 'Model 1 time : ' + str(x[1]) + '\nModel 2 time : ' + str(y[1]))
		continue
	elif not x[0] and not y[0]:
		print('both have submitted no answer')
		continue
	elif not x[0] and y[0]:
		z += 1
	elif x[0] and not y[0]:
		n += 1
	elif x[0] and y[0]:
		z += 1
		n += 1
		print('Both have submitted a wrong guess')
	if n == 2:
		print('Model 2 wins by forfeit')
		break
	if z == 2:
		print('Model 1 wins by forfeit')
		break	
	if cv.waitKey(20) & 0xFF == ord('q'):
		break    

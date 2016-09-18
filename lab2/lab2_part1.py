import cv2
import numpy as np
import numpy.linalg as la

def calculate_hue(c_max_string, r_val, g_val, b_val, delta):
	hue_value = 0.0

	if delta == 0:
		return hue_value
	elif c_max_string == 'r_val':
		hue_value = (((g_val-b_val)/delta)%6)*60
	elif c_max_string == 'g_val':
		hue_value = (((b_val-r_val)/delta)+2)*60	
	elif c_max_string == 'b_val':
		hue_value = (((r_val-g_val)/delta)+4)*60

	hue_value = (hue_value/360.0)*255
	return hue_value	

def find_saturation(c_max, delta):
	saturation_value = 0.0

	if c_max == 0:
		return saturation_value
	else:
		saturation_value = (float(delta)/float(c_max))*255
		return saturation_value	

file_name = raw_input('Enter file name: ')
file_string = 'lab2_pictures/' + file_name
image = cv2.imread(file_string)
rows_for_image = len(image[:,0])
columns_for_image = len(image[0,:])
hue_image_array = np.zeros([rows_for_image, columns_for_image])
saturation_array = np.zeros([rows_for_image, columns_for_image])
v_array = np.zeros([rows_for_image, columns_for_image])

for i in range(rows_for_image):
	for j in range(columns_for_image):
		current_pixel = image[i][j]
		red = current_pixel[0]
		green = current_pixel[1]
		blue = current_pixel[2]

		r_val = red / 255.0
		g_val = green / 255.0
		b_val = blue / 255.0

		dict_val = {'r_val':r_val, 'g_val':g_val, 'b_val':b_val}

		c_max = max(r_val, g_val, b_val)
		c_min = min(r_val, g_val, b_val)
		c_max_string = max(dict_val, key=dict_val.get)
		c_min_string = min(dict_val, key=dict_val.get)
		delta = c_max - c_min

		hue_value = calculate_hue(c_max_string, r_val, g_val, b_val, delta)
		saturation_value = find_saturation(c_max, delta)
		v_value = c_max*255

		print hue_value
		print saturation_value
		print v_value

		hue_image_array[i][j] = hue_value
		saturation_array[i][j] = saturation_value
		v_array[i][j] = v_value

if file_name == 'concert.jpg':
	cv2.imwrite('concert_hue.jpg', hue_image_array)
	cv2.imwrite('concert_saturation.jpg', saturation_array)
	cv2.imwrite('concert_brightness.jpg', v_array)
elif file_name == 'sea1.jpg':
	cv2.imwrite('sea1_hue.jpg', hue_image_array)
	cv2.imwrite('sea1_saturation.jpg', saturation_array)
	cv2.imwrite('sea1_brightness.jpg', v_array)	
elif file_name == 'sea2.jpg':
	cv2.imwrite('sea2_hue.jpg', hue_image_array)
	cv2.imwrite('sea2_saturation.jpg', saturation_array)
	cv2.imwrite('sea2_brightness.jpg', v_array)			

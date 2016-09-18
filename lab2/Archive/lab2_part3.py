import cv2
import numpy as np
import numpy.linalg as la
import math

def determine_values(current_h_val, c_val, x_val):
	r_val = g_val = b_val = 0.0
	if current_h_val >= 0 and current_h_val < 60:
		r_val = c_val
		g_val = x_val
	elif current_h_val >= 60 and current_h_val < 120:
		r_val = x_val
		g_val = c_val
	elif current_h_val >= 120 and current_h_val < 180:
		g_val = c_val
		b_val = x_val
	elif current_h_val >= 180 and current_h_val < 240:
		g_val = x_val
		b_val = c_val
	elif current_h_val >= 240 and current_h_val < 300:
		r_val = x_val
		b_val = c_val
	elif current_h_val >= 300 and current_h_val < 360:
		r_val = c_val
		b_val = x_val
	return r_val, g_val, b_val


brightness_file_name = raw_input('Brightness file to be histogram equalized: ')

brightness_image = cv2.imread(brightness_file_name)

rows_for_image = len(brightness_image[:,0])
columns_for_image = len(brightness_image[0,:])

total_pixels = rows_for_image*columns_for_image

histogram_equalized_array = np.zeros(256)

for i in range(rows_for_image):
	for j in range(columns_for_image):

		v_value = brightness_image[i][j][0]
		histogram_equalized_array[int(v_value)] += 1

cumulative_sum = 0.0

for k in range(len(histogram_equalized_array)):

	current_pmf = float(histogram_equalized_array[k])/float(total_pixels)
	cumulative_sum = cumulative_sum + current_pmf
	new_value = cumulative_sum*255
	histogram_equalized_array[k] = math.floor(new_value)

new_brightness_image = np.zeros([rows_for_image, columns_for_image])

for l in range(rows_for_image):
	for m in range(columns_for_image):

		v_value = brightness_image[l][m][0]
		new_brightness_image[l][m] = histogram_equalized_array[int(v_value)]

if 'concert' in brightness_file_name:
	hue_file_name = 'concert_hue.jpg'
	saturation_file_name = 'concert_saturation.jpg'
elif 'sea1' in brightness_file_name:
	hue_file_name = 'sea1_hue.jpg'
	saturation_file_name = 'sea1_saturation.jpg'
elif 'sea2' in brightness_file_name:
	hue_file_name = 'sea2_hue.jpg'
	saturation_file_name = 'sea2_saturation.jpg'

hue_image = cv2.imread(hue_file_name)
saturation_image = cv2.imread(saturation_file_name)

rgb_array = np.zeros([rows_for_image, columns_for_image, 3])

for i in range(rows_for_image):
	for j in range(columns_for_image):
		current_h_val = hue_image[i][j][0]
		current_s_val = saturation_image[i][j][0]
		current_v_val = new_brightness_image[i][j]

		current_h_val = (current_h_val/255.0)*360.0
		current_s_val = current_s_val/255.0
		current_v_val = current_v_val/255.0

		c_val = current_s_val*current_v_val
		x_val = c_val*(1 - abs(((current_h_val/60)%2)-1))
		m_val = current_v_val - c_val
		
		r_val, g_val, b_val = determine_values(current_h_val, c_val, x_val)
		red = (r_val + m_val)*255
		green = (g_val + m_val)*255
		blue = (b_val + m_val)*255

		rgb_array[i][j][0] = red
		rgb_array[i][j][1] = green
		rgb_array[i][j][2] = blue			

if 'concert' in brightness_file_name:
	cv2.imwrite('concert_histeq.jpg', rgb_array)
elif 'sea1' in brightness_file_name:
	cv2.imwrite('sea1_histeq.jpg', rgb_array)
elif 'sea2' in brightness_file_name:
	cv2.imwrite('sea2_histeq.jpg', rgb_array)		


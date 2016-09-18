import cv2
import numpy as np
import numpy.linalg as la

def MyConvolve(img, ff):
	
	convoluted_array = np.zeros(img.shape)

	rows_for_img = len(img[:,0])
	columns_for_img = len(img[0,:])
	ff = np.fliplr(ff)
	ff = np.flipud(ff)

	convoluted_array[0][0] = [0.0, 0.0, 0.0]
	convoluted_array[0][columns_for_img-1] = [0.0, 0.0, 0.0]
	convoluted_array[rows_for_img-1][0] = [0.0, 0.0, 0.0]
	convoluted_array[rows_for_img-1][columns_for_img-1] = [0.0, 0.0, 0.0]

	for k in range(1, rows_for_img-1):
		for h in range(1, columns_for_img-1):

			convolution_val = (ff[0][0]*img[k-1][h-1]) + (ff[0][1]*img[k-1][h]) + (ff[0][2]*img[k-1][h+1]) + (ff[1][0]*img[k][h-1]) + (ff[1][2]*img[k][h+1]) + (ff[2][0]*img[k+1][h-1]) + (ff[2][1]*img[k+1][h]) + (ff[2][2]*img[k+1][h+1])
			convoluted_array[k][h] = convolution_val

	return convoluted_array


def combine_edges(horizontal_edge_strength, vertical_edge_strength, filter_function):

	edge_detected_array = np.zeros(horizontal_edge_strength.shape)

	rows_for_img = len(horizontal_edge_strength[:,0])
	columns_for_img = len(horizontal_edge_strength[0,:])

	if filter_function.lower() == 'prewitt':
		first_val = 255 * 3
		squared = np.square(first_val)
		double = squared * 2
		normalization_factor = np.sqrt(double)

	elif filter_function.lower() == 'sobel':
		first_val = 255 * 4
		squared = np.square(first_val)
		double = squared * 2
		normalization_factor = np.sqrt(double)	

	for i in range(rows_for_img):
		for j in range(columns_for_img):
			g_y = horizontal_edge_strength[i][j]
			g_x = vertical_edge_strength[i][j]

			addition_of_squares = np.square(g_y) + np.square(g_x)
			square_root = np.sqrt(addition_of_squares)
			normalized = (square_root / normalization_factor)*255.0


			edge_detected_array[i][j] = normalized
	return edge_detected_array		


def edge_thinning(edge_detected_array):
	rows_for_img = len(edge_detected_array[:,0])
	columns_for_img = len(edge_detected_array[0,:])
	thinned_edge = np.zeros(edge_detected_array.shape)

	for i in range(1, rows_for_img - 1):
		for j in range(1, columns_for_img - 1):
			current_pixel_val = edge_detected_array[i][j][0]
			left_pixel_val = edge_detected_array[i][j-1][0]
			right_pixel_val = edge_detected_array[i][j+1][0]

			max_value_horizontal = max(current_pixel_val, left_pixel_val, right_pixel_val)

			if current_pixel_val == max_value_horizontal:
				thinned_edge[i][j] = edge_detected_array[i][j]
			else:
				up_pixel_val = edge_detected_array[i-1][j][0]
				down_pixel_val = edge_detected_array[i+1][j][0]

				max_value_vertical = max(current_pixel_val, up_pixel_val, down_pixel_val)

				if current_pixel_val == max_value_vertical:
					thinned_edge[i][j] = edge_detected_array[i][j]
				else:
					thinned_edge[i][j] = [0.0, 0.0, 0.0]
	return thinned_edge					


def get_grayscale_image(img):
	rows_for_img = len(img[:,0])
	columns_for_img = len(img[0,:])
	grayscale_image = np.zeros(img.shape)

	for i in range(rows_for_img):
		for j in range(columns_for_img):
			current_pixel = img[i][j]
			red = current_pixel[0]
			green = current_pixel[1]
			blue = current_pixel[2]

			r_val = red / 255.0
			g_val = green / 255.0
			b_val = blue / 255.0

			c_max = max(r_val, g_val, b_val)
			v_value = c_max*255

			grayscale_image[i][j] = v_value

	return grayscale_image		

file_name = raw_input('Enter file name: ')
filter_function = raw_input('Enter filter function (Prewitt or Sobel): ')
thinning_function = raw_input('Thinning? (Y/N): ')
prewitt_horizontal_edge_filter = np.array([[1, 1, 1],[0, 0, 0], [-1, -1, -1]])
prewitt_vertical_edge_filter = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
sobel_horizontal_edge_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
sobel_vertical_edge_filter = np.array([[1, 0, -1],[2, 0, -2], [1, 0, -1]])

image = cv2.imread(file_name)

#we first obtain the grayscale image, i.e. V value like lab 2
grayscale_image = get_grayscale_image(image)

#then we pass through the convolution function
if filter_function.lower() == 'prewitt':
	horizontal_edge_strength = MyConvolve(grayscale_image, prewitt_horizontal_edge_filter)
	vertical_edge_strength = MyConvolve(grayscale_image, prewitt_vertical_edge_filter)
	edge_detected_array = combine_edges(horizontal_edge_strength, vertical_edge_strength, filter_function)

elif filter_function.lower() == 'sobel':
	horizontal_edge_strength = MyConvolve(grayscale_image, sobel_horizontal_edge_filter)
	vertical_edge_strength = MyConvolve(grayscale_image, sobel_vertical_edge_filter)
	edge_detected_array = combine_edges(horizontal_edge_strength, vertical_edge_strength, filter_function)	

if filter_function.lower() == 'prewitt':
	if file_name == 'example.jpg':
		cv2.imwrite('example_prewitt.jpg', edge_detected_array)
	elif file_name == 'test1.jpg':
		cv2.imwrite('test1_prewitt.jpg', edge_detected_array)
	elif file_name == 'test2.jpg':
		cv2.imwrite('test2_prewitt.jpg', edge_detected_array)
	elif file_name == 'test3.jpg':
		cv2.imwrite('test3_prewitt.jpg', edge_detected_array)			
elif filter_function.lower() == 'sobel':
	if file_name == 'example.jpg':
		cv2.imwrite('example_sobel.jpg', edge_detected_array)
	elif file_name == 'test1.jpg':
		cv2.imwrite('test1_sobel.jpg', edge_detected_array)
	elif file_name == 'test2.jpg':
		cv2.imwrite('test2_sobel.jpg', edge_detected_array)
	elif file_name == 'test3.jpg':
		cv2.imwrite('test3_sobel.jpg', edge_detected_array)		

if thinning_function.lower() == 'y':
	if filter_function.lower() == 'prewitt':
		if file_name == 'example.jpg':
			thinned_edge = edge_thinning(edge_detected_array)
			cv2.imwrite('example_prewitt_thinning.jpg', thinned_edge)
		elif file_name == 'test1.jpg':
			thinned_edge = edge_thinning(edge_detected_array)
			cv2.imwrite('test1_prewitt_thinning.jpg', thinned_edge)
		elif file_name == 'test2.jpg':
			thinned_edge = edge_thinning(edge_detected_array)
			cv2.imwrite('test2_prewitt_thinning.jpg', thinned_edge)
		elif file_name == 'test3.jpg':
			thinned_edge = edge_thinning(edge_detected_array)
			cv2.imwrite('test3_prewitt_thinning.jpg', thinned_edge)			
	elif filter_function.lower() == 'sobel':
		if file_name == 'example.jpg':
			thinned_edge = edge_thinning(edge_detected_array)
			cv2.imwrite('example_sobel_thinning.jpg', thinned_edge)	
		elif file_name == 'test1.jpg':
			thinned_edge = edge_thinning(edge_detected_array)
			cv2.imwrite('test1_sobel_thinning.jpg', thinned_edge)	
		elif file_name == 'test2.jpg':
			thinned_edge = edge_thinning(edge_detected_array)
			cv2.imwrite('test2_sobel_thinning.jpg', thinned_edge)
		elif file_name == 'test3.jpg':
			thinned_edge = edge_thinning(edge_detected_array)
			cv2.imwrite('test3_sobel_thinning.jpg', thinned_edge)


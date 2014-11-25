import cv2
import numpy as np

TEST_IMAGE = "/home/qk/Projects/music_recognition_project/test_data/taylor_swift_shake_it_off/1.png"

def eliminateStaffLines(img):
	"""
		Eliminates staff lines from a single-channel binary image of a music sheet.

		Args:
			img: Single channel binary image of a music sheet.

		Returns:
			Modified image.
	"""
	img_height = img.shape[0]
	img_width = img.shape[1]

	staff_line_idx = []

	for row in range(img_height):
		if np.bincount(img[row])[0] > img_width / 2:
			# This is an important row.
			staff_line_idx.append(row)

	count = 0
	for row in staff_line_idx:
		if staff_line_idx[min(count+1, len(staff_line_idx) - 1)] == row + 1:
			dist = 2
			del staff_line_idx[min(count+1, len(staff_line_idx))]
		else:
			dist = 1

		for col in range(img_width):
			if img[row - 1][col] == 255 and img[row + dist][col] == 255:
				img[row][col] = 255
				img[row + dist / 2][col] = 255
		count += 1
	return thresh

def floodDetection(img):
	"""
		Performs flood detection on a single-channel binary image of a music sheet.

		Args:
			img: Single channel binary image of a music sheet.

		Returns:
			Array of bounding boxes for each of the objects detected.
	"""
	img_height = img.shape[0]
	img_width = img.shape[1]
	temp_image = np.copy(img)
	flood_window = [-2, -1, 0, 1, 2]
	detected_rois = []

	for row in range(img_height):
		for col in range(img_width):
			# Basic flood detection algorithm.
			if temp_image[row][col] == 0:
				min_x = -1
				min_y = -1
				max_x = -1
				max_y = -1

				pixels = []
				pixels.append([row, col])
				for pixel in pixels:
					for sub_row in flood_window:
						for sub_col in flood_window:
							# Disregard 0,0.
							if sub_row == sub_col and sub_row == 0:
								pass
							# Disregard image borders.
							elif pixel[0] + sub_row > img_height or pixel[1] + sub_col > img_width:
								pass
							# If you find a black pixel, append.
							elif temp_image[pixel[0] + sub_row][pixel[1] + sub_col] == 0:
								# Check if that pixel is not already in the pixels.
								new_pixel = [pixel[0] + sub_row,pixel[1] + sub_col]
								if new_pixel not in pixels:
									pixels.append(new_pixel)
									if new_pixel[1] < min_x or min_x < 0:
										min_x = new_pixel[1]
									if new_pixel[1] > max_x or max_x < 0:
										max_x = new_pixel[1]
									if new_pixel[0] < min_y or min_y < 0:
										min_y = new_pixel[0]
									if new_pixel[0] > max_y or max_y < 0:
										max_y = new_pixel[0]
				# Whiten out the pixels we have already selected.
				cv2.rectangle(temp_image, (min_x, min_y), (max_x, max_y), 255, cv2.FILLED)

				# Add to list of ROIs.
				detected_rois.append([min_x, min_y, max_x, max_y])
	return detected_rois

# TODO(dcastro): Change 0 to cv2.grayscale or whichever it is.
img = cv2.imread(TEST_IMAGE, 0)

# Threshold the image.
ret, thresh = cv2.threshold(img, 255 * 2 / 3, 255, cv2.THRESH_BINARY)

thresh = eliminateStaffLines(thresh)

rois = floodDetection(thresh)

for roi in rois:
	cv2.rectangle(thresh, (roi[0], roi[1]), (roi[2], roi[3]), 128, 1)

cv2.imwrite("out.png", thresh)
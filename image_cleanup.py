import cv2
import numpy as np

TEST_IMAGE = "/home/qk/Projects/music_recognition_project/test_data/taylor_swift_shake_it_off/1.png"

# TODO(dcastro): Change 0 to cv2.grayscale or whichever it is.
img = cv2.imread(TEST_IMAGE, 0)

# Threshold the image.
ret, thresh = cv2.threshold(img, 255 * 2 / 3, 255, cv2.THRESH_BINARY)

img_height = thresh.shape[0]
img_width = thresh.shape[1]

staff_line_idx = []

for row in range(len(thresh)):
	if np.bincount(thresh[row])[0] > img_width / 2:
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
		if thresh[row - 1][col] == 255 and thresh[row + dist][col] == 255:
			thresh[row][col] = 255
			thresh[row + dist / 2][col] = 255
	count += 1

cv2.imwrite("out.png", thresh)
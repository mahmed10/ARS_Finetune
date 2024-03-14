import numpy as np
def colorize_mask(mask, palette=None):
	"""
	Colorize the given segmentation mask.

	Args:
		mask (np.ndarray): A 2D numpy array of shape (H, W) containing class labels as values.
		palette (dict, optional): A dictionary mapping class labels to RGB colors. 
		If None, a default palette will be used.

	Returns:
		np.ndarray: An RGB image of shape (H, W, 3).
	"""
	# If no custom palette is provided, use a default one
	if palette is None:
		palette = {
			0: [128, 64, 128],
			1: [244, 35, 232],
			2: [70, 70, 70],
			3: [102, 102, 156],
			4: [190, 153, 153],
			5: [153, 153, 153],
			6: [250, 170, 30],
			7: [220, 220, 0],
			8: [107, 142, 35],
			9: [152, 251, 152],
			10: [70, 130, 180],
			11: [220, 20, 60],
			12: [255, 0, 0],
			13: [0, 0, 142],
			14: [0, 0, 70],
			15: [0, 60, 100],
			16: [0, 80, 100],
			17: [0, 0, 230],
			18: [119, 11, 32],
			255: [0, 0, 0]
		}

	# Create an empty RGB image with the same spatial dimensions as the mask
	colorized = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

	# For each class label, populate the RGB image with the corresponding color
	for label, color in palette.items():
		colorized[mask == label] = color

	return colorized
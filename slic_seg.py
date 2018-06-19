from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import cv2
import os 
import argparse
folder = 'test'
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True, help = "Path to the image")
# args = vars(ap.parse_args())
file_list = os.listdir(folder)
for filename in file_list:
# load the image and convert it to a floating point data type
	image = img_as_float(io.imread(os.path.join(folder,filename)))
	 
	# loop over the number of segments
	numSegments = [100, 200, 300]
		# apply SLIC and extract (approximately) the supplied number
		# of segments
	segments = slic(image, n_segments = numSegments[1], sigma = 5)
 
	# show the output of SLIC
	#fig = plt.figure("Superpixels -- %d segments" % (numSegments))
	#ax = fig.add_subplot(1, 1, 1)
	cv2.imshow('re',mark_boundaries(image, segments))
	#plt.axis("off")
	cv2.waitKey(0)	 
	# show the plots
	#plt.show()
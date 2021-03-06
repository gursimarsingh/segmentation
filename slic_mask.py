from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage.draw import circle, line_aa, polygon
#from  raw_read import get_mask
import cv2 
import numpy as np
import generate_mask as gm
import os 
import argparse
img_src = 'Lifting/images'
csv_src  = 'Lifting/csv'
dst = 'grab_cut/Lifting2'
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True, help = "Path to the image")
# args = vars(ap.parse_args())
file_list = os.listdir(img_src)
file_list.sort()
def read_csv(filename):
	
	f=filename.split('.')
	with open(csv_src + '/' + f[0]+'.csv', 'rb') as csvfile:
		pt=[]
		for line in csvfile.readlines():	
			array = line.split(',')
		for k in range(len(array)):
			if k%3==0:
				x=int(round(float(array[k])*w))
				y=int(round(float(array[k+1])*h))
				
				pt.append([x,y,1])

	return pt

def create_rect(pt,h,w):

	rect=[]
	sort_x= sorted(pt,key= lambda x:x[0])
	sort_y= sorted(pt,key= lambda y:y[1])
	for item in sort_x:
		if item[0]!=0 and item[1]!=0:
			rect.append(item[0]-int(0.1*w))
			break

	for item in sort_y:
		if item[0]!=0 and item[1]!=0:
			rect.append(item[1]-int(0.1*h))
			break
	#print rect
	rect.append(sort_x[-1][0] +int(0.1*w))
  	rect.append(sort_y[-1][1] +int(0.1*h))
	
	if rect[0]<0:
		rect[0]=0
	if rect[1]<0:
		rect[1]=0

	if rect[2]>w:
		rect[2]=w
	if rect[3]>h:
		rect[3]=h

	return rect
def add_limb(kp1,kp2,mask,point_radius=7):
	MISSING_VALUE =0
	from_missing = kp1[0] == MISSING_VALUE or kp1[1] == MISSING_VALUE
	to_missing = kp2[0] == MISSING_VALUE or kp2[1] == MISSING_VALUE
		#from_missing = kp1[2] == MISSING_VALUE
		#to_missing = kp2[2] == MISSING_VALUE
	if from_missing or to_missing:
	    return mask
	img_size = (h,w)
	kp1 = np.asarray(kp1[0:2])
	kp2 = np.asarray(kp2[0:2])
	norm_vec = kp1 - kp2
	norm_vec = np.array([-norm_vec[1],norm_vec[0]])
	norm_vec = point_radius * norm_vec / np.linalg.norm(norm_vec)


	vetexes = np.array([
	    kp1 + norm_vec,
	    kp1 - norm_vec,
	    kp2 - norm_vec,
	    kp2 + norm_vec
	])
#pdb.set_trace()
	yy, xx = polygon(vetexes[:, 1], vetexes[:, 0], shape=img_size)
	mask[yy, xx] = 255

	yy, xx = circle(kp1[1], kp1[0], radius=point_radius, shape=img_size)
	mask[yy, xx] = 255
	yy, xx = circle(kp2[1], kp2[0], radius=point_radius, shape=img_size)
	mask[yy, xx] = 255

	return mask
def ret_mid_pt(kp):
	# rsh = np.asarray(kp[2]) 
	# lsh = np.asarray(kp[5])
	# lhip = np.asarray(kp[11])
	# rhip= np.asarray(kp[8])
	# pt = [rsh[0:2],lsh[0:2],lhip[0:2],rhip[0:2]]
	bck = np.asarray([0,0])
	i=0
	for p in kp:
		if p[0]!=0 and p[1]!=0:
			bck = bck+np.asarray(p[0:2])
			i=i+1
	bck = bck/i
	#kp.append(list(bck.astype(np.uint8)))
	return list(bck.astype(np.uint8))
# def read_raw(filename,r,c):
# 	fd = open(os.path.join(mask_src,filename.split('.')[0] + '.raw' ), 'rb')

# 	f = np.fromfile(fd, dtype=np.uint8,count=r*c)
# 	im = f.reshape((r,c)) #notice row, column format
# 	fd.close()
# 	return im
g=gm.GenerateMask()
for filename in file_list:
# load the image and convert it to a floating point data type
	image = img_as_float(io.imread(os.path.join(img_src,filename)))
	img = cv2.imread(os.path.join(img_src,filename))
	#image =  cv2.resize(image,None,fx=2.5,fy=2.5,interpolation=cv2.INTER_CUBIC)
	#image =  cv2.resize(image,None,fx=2.5,fy=2.5,interpolation=cv2.INTER_CUBIC)
	#img =  cv2.resize(img,None,fx=2.5,fy=2.5,interpolation=cv2.INTER_CUBIC).astype(np.uint8)
	temp = image.copy().astype('uint8')
	h,w,d=image.shape

	

	kp=read_csv(filename) 
	bg,sure_fg = g.mask_generate(kp,h,w)

	mid_hip = ret_mid_pt([kp[8],kp[11]])
	sure_fg = add_limb(kp[1],mid_hip,sure_fg)
	# sure_fg2 = add_limb(kp[1],kp[11],sure_fg)
	# sure_fg2 = add_limb(kp[1],kp[8],sure_fg2)
	sure_fg = add_limb(kp[5],kp[8],sure_fg)
	sure_fg = add_limb(kp[2],kp[11],sure_fg)
	# sure_fg2 = add_limb(kp[2],kp[8],sure_fg2)
	# sure_fg2 = add_limb(kp[5],kp[11],sure_fg2)

	fg_orig = sure_fg.copy()
	#fg_orig2 = sure_fg2.copy()
	sure_fg2 = sure_fg.copy()
	cv2.imshow('s',sure_fg2)
	# loop over the number of segments
	numSegments = [100, 200, 300]
		
	segments = slic(image, n_segments = numSegments[1], sigma = 5)
	#print segments
	fg_mask= segments.copy()
	#cv2.imshow('ss',fg_mask)
	fg=[]
	for (i, segVal) in enumerate(np.unique(segments)):
		mask = np.zeros(image.shape[:2], dtype = "uint8")
		mask[segments == segVal] = 255
		_, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		seg_area = cv2.contourArea(contours[0])
		mask2 =  cv2.bitwise_and(sure_fg,mask)
		_, contours, _ = cv2.findContours(mask2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		if len(contours)>0:
			inter_area= cv2.contourArea(contours[0])
			if inter_area>seg_area*0.4:
				sure_fg2[mask==255] =255

		#cv2.imshow('ssds',mask)
		
		# _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		# #print contours
		# #m =cv2.drawContours(temp,contours[0],-1,255,2,8)
		# # cv2.imshow('m',m)
		# # cv2.waitKey(0)
		# #pts = np.where((segments==segVal))
		# for k in kp:
		# 	pt = (k[0],k[1])
		# 	if cv2.pointPolygonTest(contours[0], pt, False)>=0:
		# 		#print 'in'
				
		# 		fg.append(segVal)
		# 		fg_mask =  np.where(fg_mask[:,:]==segVal,300,fg_mask[:,:])


	# fg_mask =  np.where(fg_mask[:,:]==300,255,0)
	# mask2 =  cv2.bitwise_or(sure_fg.astype('uint8'),fg_mask.astype('uint8'))
	# mask2 =  np.where(mask2[:,:]==255,1,0)
	kernel =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

	sure_fg2 = cv2.morphologyEx(sure_fg2,cv2.MORPH_CLOSE,kernel,iterations=2)
	mask2 =  np.where(sure_fg2[:,:]==255,1,0).astype('uint8')
	img2 = image*mask2[:,:,np.newaxis]

##########################################################################


	init_mask = np.zeros(image.shape[:2], dtype = "uint8")
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	rect=create_rect(kp,h,w)
	
	tp_c=(rect[0],rect[1]) 
	bt_c=(rect[2],rect[3])
	init_mask, bgdModel, fgdModel = cv2.grabCut(img,init_mask,tuple(rect),bgdModel,fgdModel,3,cv2.GC_INIT_WITH_RECT)
	sure_bg = np.where((init_mask==2)|(init_mask==0),0,1).astype('uint8')
	init_mask = np.where((init_mask==1)|(init_mask==3),0,init_mask).astype('uint8')
	

	init_mask[mask2==1]=3
	
	
	init_mask[mask2==0]=2
	init_mask[sure_bg==0]=0
	init_mask[bg==0]=0
	init_mask[fg_orig==255]=1
	mask, bgdModel, fgdModel = cv2.grabCut(img,init_mask,None,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_MASK)
	mask = np.where((mask==2)|(mask==0),0,mask).astype('uint8')
	mask = np.where((mask==3)|(mask==1),3,mask).astype('uint8')

	#mask[sure_fg==1]=3

	mask[fg_orig==255]=1

	mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_MASK)
	mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')

	# mm = cv2.bitwise_and(mask,mask2)
	#mask= cv2.bitwise_or(mask,mask2)
	img1 = img*mask[:,:,np.newaxis]
	re=cv2.resize(img1,(64,128),interpolation = cv2.INTER_AREA)
	mask = np.where((mask==1),255,0).astype('uint8')
	# mm = get_mask(filename)
	# bgdModel = np.zeros((1,65),np.float64)
	# fgdModel = np.zeros((1,65),np.float64)
	# init_mask= np.zeros((h,w),dtype='uint8')

	# mm_img = image*mm[:,:,np.newaxis]
	# mm[fg_mask==1]=1
	# mask, bgdModel, fgdModel = cv2.grabCut(mm_img.astype('uint8'),mm,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
	# mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	# img3 = image*mask[:,:,np.newaxis]
	# show the output of SLIC
	#fig = plt.figure("Superpixels -- %d segments" % (numSegments))
	#ax = fig.add_subplot(1, 1, 1)
	#cv2.imshow('re',mark_boundaries(image, segments))
	cv2.imshow('mask',img1)
	cv2.imwrite(os.path.join(dst,filename),mask)
	cv2.imshow('mask222',sure_fg)
	#plt.axis("off")
	cv2.waitKey(1)

	# show the plots
#plt.show()
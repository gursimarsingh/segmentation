import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from  sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
data_file =  'file.csv'

f = open(data_file,newline ='')
reader  =  csv.reader(f,delimiter=',')
limbs =[]
gts =[]
bboxs_h = []
bboxs_w = []
lin = LinearRegression()

for row in reader:
    img_name = row[0]
    limb_mean = float(row[1])
    bbox_h = float(row[2])
    bbox_w = float(row[3])
    gt_rad = float(row[4])
    
    limbs.append(limb_mean)
    bboxs_h.append(bbox_h*bbox_w)
   # bboxs_w.append(bbox_w)
    gts.append(gt_rad)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
bboxs_h = np.array(bboxs_h)
#print (np.min(bboxs_h))
limbs = np.array(limbs)
bboxs_h = bboxs_h/(np.max(bboxs_h)-np.min(bboxs_h))
limbs  = limbs /(np.max(limbs)-np.min(limbs))

#print (np.max(bboxs_h))
X = np.concatenate((bbox_h.reshape(-1,1),limbs.reshape(-1,1)),axis=1)
Y  = np.array(gts).reshape(-1,1)
lin.fit(X,Y)

ax.scatter(bboxs_h,limbs,np.array(gts))
plt.show()


  

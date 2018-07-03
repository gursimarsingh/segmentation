import numpy as np
import skimage.io 
from sklearn import mixture
import pycoop
import pycoop.potentials as potentials
import os
import cv2

folder = '../images'
files = os.listdir(folder)
files.sort()
# load image and foreground / background label marks
for filename in files:
    im = cv2.imread('../images/'+filename).astype(np.float64)
    mark_im = cv2.imread('../export_mask/'+filename.split('.')[0]+'.png')
    h,w,d = im.shape
    
    # show image and labels
    cv2.imshow('or',im.astype('uint8'))
    cv2.imshow('mask',mark_im)
    # GMM example: learn 5 component model on pixels, show cluster assignments
    gmm = mixture.GMM(n_components=2, covariance_type='diag')
    pix = im.reshape((-1, 3))
    gmm.fit(pix)
    gmm_im = gmm.predict(pix).reshape(im.shape[:-1])
    gmm_color = cv2.applyColorMap((gmm_im*50).astype('uint8'), cv2.COLORMAP_JET)


    cv2.imshow("gmm",gmm_color)

    fg_pix, bg_pix = potentials.extract_pix_from_marks(im, mark_im)
    fg_gmm = potentials.learn_gmm(fg_pix,n_comp=2)
    bg_gmm = potentials.learn_gmm(bg_pix,n_comp=5)
    fg_un, bg_un = potentials.make_gmm_unaries(im.reshape((-1, 3)), fg_gmm, bg_gmm)
    print np.unique(fg_un)
    fg_res= fg_un.reshape(im.shape[0:2])
    bg_res= bg_un.reshape(im.shape[0:2])
   
    # show mask where the foreground models wins over the background, and the log ratio for the foreground model
    un =  np.where((fg_res[:,:] > bg_res[:,:]),255,0).astype('uint8')

    print np.unique(fg_un - bg_un)
    cv2.imshow('fg_un > bg_un',un)
    
    cv2.imshow('fg_un - bg_un',(fg_un - bg_un).reshape(im.shape[:-1]))

    ig = pycoop.InputGraph(im)

    edge_cluster_classes, edge_centroids = potentials.cluster_edges(ig.edges, k=5)
    
    

    ig.setClasses(edge_cluster_classes, 5)
    ig.setUnaries(fg_un, bg_un)

    label_im, cc_cost, cc_cut = pycoop.segment(ig, lambda_=2.5, theta=0.01, max_iter=12)
    print np.unique(label_im)
    result_cc = np.ones_like(im)
    result_cc[label_im] = im[label_im]
    print np.unique(result_cc)
    cv2.imshow('coop_cut',result_cc.astype('uint8'))

    label_im, cost, cut = pycoop.segment(ig, lambda_=2.5, theta=1, max_iter=12)
    result_gc = np.ones_like(im)
    result_gc[label_im] = im[label_im]
    
    cv2.imshow('graph_cut',result_gc.astype('uint8'))

    label_im, cost, cut = pycoop.segment(ig, lambda_=2.5, theta=-1, max_iter=12)
    result_submod = np.ones_like(im)
    result_submod[label_im] = im[label_im]
    
    cv2.imshow('submod',result_submod.astype('uint8'))
    #skimage.io.imsave(filename.split('.')[0]+'.png', result_cc)
    cv2.waitKey(0)
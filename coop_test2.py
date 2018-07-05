import numpy as np
import skimage.io 
from sklearn import mixture
import pycoop
import pycoop.potentials as potentials
import os
import cv2
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
folder = '../images'
files = os.listdir(folder)
files.sort()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# load image and foreground / background label marks
for filename in files:
    im = cv2.imread('../images/'+filename).astype(np.float64)
    im_rgb = im.copy()
    im = cv2.cvtColor(im.astype(np.uint8),cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    im[:,:,2] = clahe.apply(im[:,:,2])
    im = cv2.cvtColor(im.astype(np.uint8),cv2.COLOR_HSV2BGR).astype(np.float64)

    mark_im = cv2.imread('../export_mask/'+filename.split('.')[0]+'.png')
    h,w,d = im.shape
    
    # show image and labels
    cv2.imshow('pro',im.astype('uint8'))
    #cv2.imshow('or',im_rgb.astype('uint8'))
    cv2.imshow('mask',mark_im)
    # GMM example: learn 5 component model on pixels, show cluster assignments
    gmm = mixture.GMM(n_components=2, covariance_type='diag')
    pix = im.reshape((-1, 3))
    gmm.fit(pix)
    gmm_im = gmm.predict(pix).reshape(im.shape[:-1])
    gmm_color = cv2.applyColorMap((gmm_im*50).astype('uint8'), cv2.COLORMAP_JET)


    #cv2.imshow("gmm",gmm_color)

    fg_pix, bg_pix = potentials.extract_pix_from_marks(im, mark_im)
    fg_gmm = potentials.learn_gmm(fg_pix,n_comp=5)
    bg_gmm = potentials.learn_gmm(bg_pix,n_comp=5)
    fg_un, bg_un = potentials.make_gmm_unaries(im.reshape((-1, 3)), fg_gmm, bg_gmm)
    #print np.unique(fg_un)
    fg_res= fg_un.reshape(im.shape[0:2])
    bg_res= bg_un.reshape(im.shape[0:2])
   
    # show mask where the foreground models wins over the background, and the log ratio for the foreground model
    un =  np.where((fg_res[:,:] > bg_res[:,:]),255,0).astype('uint8')

    #print np.unique(fg_un - bg_un)
    # cv2.imshow('fg_un > bg_un',un)
    
    # cv2.imshow('fg_un - bg_un',(fg_un - bg_un).reshape(im.shape[:-1]))

    ig = pycoop.InputGraph(im)

    edge_cluster_classes, edge_centroids = potentials.cluster_edges(ig.edges, k=8)
    
    

    ig.setClasses(edge_cluster_classes, 8)
    ig.setUnaries(fg_un, bg_un)

    # label_im, cc_cost, cc_cut = pycoop.segment(ig, lambda_=2.5, theta=0.001, max_iter=12)
    # print (np.unique(label_im))
    # result_cc = np.ones_like(im)
    # result_cc[label_im] = im[label_im]
    # # print np.unique(result_cc)
    # cv2.imshow('coop_cut',result_cc.astype('uint8'))
    # label_im, cc_cost, cc_cut = pycoop.segment(ig, lambda_=2, theta=0.001, max_iter=12)
    # #print np.unique(label_im)
    # result_cc = np.ones_like(im)
    # result_cc[label_im] = im[label_im]
    # # print np.unique(result_cc)
    # cv2.imshow('coop_cut2',result_cc.astype('uint8'))
    # #cv2.imshow('coop_cut',result_cc.astype('uint8'))
    # label_im, cc_cost, cc_cut = pycoop.segment(ig, lambda_=0.5, theta=-0.01, max_iter=12)
    # #print np.unique(label_im)
    # result_cc = np.ones_like(im)
    # result_cc[label_im] = im[label_im]
    # # print np.unique(result_cc)
    # cv2.imshow('coop_cut2',result_cc.astype('uint8'))
    label_im, cc_cost, cc_cut = pycoop.segment(ig, lambda_=0.5, theta=0.001, max_iter=12)
    #print np.unique(label_im)
    print label_im.shape
    result_cc = np.zeros(im.shape)
    t= np.zeros(im.shape)
    t.fill(255)
    result_cc[label_im] = t[label_im]
    result_cc = result_cc[:,:,0].copy()
    #result_cc = cv2.morphologyEx(result_cc[:,:,0],cv2.MORPH_CLOSE,kernel,iterations=1)
    mask = np.where(((mark_im[:,:,0]==255) & (mark_im[:,:,1] ==0) & (mark_im[:,:,2]==0)),255,0)
    bg = np.where(((mark_im[:,:,0]==0) & (mark_im[:,:,1] ==0) & (mark_im[:,:,2]==255)),255,0)
    result_cc[mask==255]=255 
    result_cc[bg==255]=0
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    init_mask =np.zeros((h,w),dtype='uint8')

    init_mask[result_cc==255] = 3
    init_mask[bg==255] = 0
    init_mask[mask==255] = 1
    init_mask[bg+result_cc ==0]=2
    mask_color = cv2.applyColorMap((init_mask*50).astype('uint8'), cv2.COLORMAP_JET)
    cv2.imshow('m',mask_color)
    mask, bgdModel, fgdModel = cv2.grabCut(im.astype('uint8'),init_mask,None,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask==0)|(mask==2),0,1)
    fg_index = np.where(mask==1)

    fg  = im[fg_index[0],fg_index[1],:]
    print fg.shape
    bg  = im[mask == 0]
    img1 = im*mask[:,:,np.newaxis]
    fg_gmm = potentials.learn_gmm(fg,n_comp=5)
    bg_gmm = potentials.learn_gmm(bg,n_comp=5)

    fg_un = fg_gmm.score(fg.reshape((-1, 3)))
    bg_un = bg_gmm.score(fg.reshape((-1, 3))) 
    p_min= np.min(bg_un)
    bg_pr = bg_un-p_min
    bg_pr = bg_pr/np.max(bg_pr)*1.0
    fp_min= np.min(fg_un)
    fg_pr = fg_un-fp_min
    fg_pr = fg_pr/np.max(fg_pr)*1.0
    print np.max(fg_pr)
    print np.median(fg_pr)

    #bg_un = potentials.make_gmm_unaries(fg.reshape((-1, 3)), fg_gmm, bg_gmm)
    #print np.unique(fg_un)
    #print np.unique(fg_un)
    # fg_res= fg_un.reshape(im.shape[0:2])
    # bg_res= bg_un.reshape(im.shape[0:2])
   
    # show mask where the foreground models wins over the background, and the log ratio for the foreground model
    un =  np.where((bg_pr < 0.8))
    print un[0].shape
    fg_ind  = np.concatenate((fg_index[0].reshape(-1,1),fg_index[1].reshape(-1,1)),axis=1)
    #print fg_ind.shape
    fg_cor = fg_ind[un[0],:]
    #print fg_cor.shape
    fin_mask = np.zeros(im.shape,dtype=np.uint8)

    fin_mask[fg_cor[:,0],fg_cor[:,1]] = im[fg_cor[:,0],fg_cor[:,1]]

    #print np.unique(fg_un - bg_un)
    
    # print np.unique(result_cc)
    #print result_cc.shape
    # gmm = mixture.GMM(n_components=3, covariance_type='diag')
    # pix = img1.reshape((-1, 3)).astype(np.float64)
    # gmm.fit(pix)
    # gmm_im = gmm.predict(pix).reshape(img1.shape[:-1])
    # gmm_color = cv2.applyColorMap((gmm_im*50).astype('uint8'), cv2.COLORMAP_JET)
    # cv2.imshow("gmm",gmm_color)
    cv2.imshow('coop_cut3',result_cc.astype('uint8'))
    cv2.imshow('mm',img1.astype('uint8'))
    cv2.imshow('fin_mask',fin_mask)

    # label_im, cost, cut = pycoop.segment(ig, lambda_=2.5, theta=1, max_iter=12)
    # result_gc = np.ones_like(im)
    # result_gc[label_im] = im[label_im]
    
    # cv2.imshow('graph_cut',result_gc.astype('uint8'))

    # label_im, cost, cut = pycoop.segment(ig, lambda_=2.5, theta=-1, max_iter=12)
    # result_submod = np.ones_like(im)
    # result_submod[label_im] = im[label_im]
    
    # cv2.imshow('submod',result_submod.astype('uint8'))
    #skimage.io.imsave(filename.split('.')[0]+'.png', result_cc)
    cv2.waitKey(0)
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import generate_binary_structure,gaussian_filter,label,binary_closing,zoom
from skimage.measure import regionprops

def z_score_RF(A,pix_len,rf_len,abs=True):
    # get x and y positions of pixs
    n = A.shape[0]
    xs,ys = np.meshgrid(np.arange(n)*pix_len,np.arange(n)*pix_len)
    
    # calculate maximum pix location in image
    if abs:
        max_idxs = np.unravel_index(np.argmax(np.abs(A)),(n,n))
    else:
        max_idxs = np.unravel_index(np.argmax(A),(n,n))
    max_x,max_y = max_idxs[0]*pix_len,max_idxs[1]*pix_len
    
    non_rf = np.sqrt((xs-max_x)**2 + (ys-max_y)**2) > rf_len
    
    return (A - np.mean(A[non_rf]))/np.std(A[non_rf])

def binary_rm_sml(A,pix_area,cutoff):
    clean_A = np.zeros_like(A).astype(int)
    
    # label all connected components
    lab_A, nfeat_A = label(A,generate_binary_structure(2,2))
    
    # only keep components with area larger than cutoff
    for feat in range(1,nfeat_A+1):
        if pix_area*np.sum(lab_A==feat) > cutoff:
            clean_A += (lab_A==feat).astype(int)
            
    return clean_A

def binary_rm_far(A,pix_len,cutoff):
    clean_A = np.zeros_like(A).astype(int)
    
    # get x and y positions of pixs
    n = A.shape[0]
    xs,ys = np.meshgrid(np.arange(n)*pix_len,np.arange(n)*pix_len)
    
    # label all connected components
    lab_A, nfeat_A = label(A,generate_binary_structure(2,2))
    if nfeat_A==0:
        return clean_A
    
    # # calculate size of all components
    # sizes = np.zeros(nfeat_A)
    # for feat in range(1,nfeat_A+1):
    #     sizes[feat-1] = np.sum(lab_A==feat)
    # big_feat = np.argmax(sizes) + 1
    
    # # calculate center of largest component
    # big_cent_x = np.mean(xs[lab_A==big_feat])
    # big_cent_y = np.mean(ys[lab_A==big_feat])
    # clean_A += (lab_A==big_feat).astype(int)
    
    # calculate center of mass of image
    big_cent_x = np.mean(xs[A==1])
    big_cent_y = np.mean(ys[A==1])
    
    # only keep components with center within cutoff of largest component center
    for feat in range(1,nfeat_A+1):
        # if feat==big_feat:
        #     continue
        sml_cent_x = np.mean(xs[lab_A==feat])
        sml_cent_y = np.mean(ys[lab_A==feat])
        if np.sqrt((sml_cent_x-big_cent_x)**2+(sml_cent_y-big_cent_y)**2) < cutoff:
            clean_A += (lab_A==feat).astype(int)
            
    return clean_A

def binary_clean_SR(A,pix_len):
    pix_area = pix_len**2
    
    # remove all connected subregions with area less than 8 deg^2
    clean_A = binary_rm_sml(A,pix_area,8)
    
    # remove all connected subregions that are greater than 22.5 deg from center of largest subregion
    clean_A = binary_rm_far(clean_A,pix_len,22.5)
            
    # smooth shape of connected subregions
    # clean_A = binary_closing(clean_A,structure=generate_binary_structure(2,2),iterations=int(np.round(2/l))).astype(int)
    # clean_A = binary_closing(clean_A,structure=generate_binary_structure(2,2),iterations=int(np.round(2/l))).astype(int)
    
    return clean_A

def binary_clean_RF(A,B,pix_len):
    clean_A = A.copy()
    clean_B = B.copy()
    
    pix_area = pix_len**2
    
    # remove all connected subregions with area less than 8 deg^2
    clean_A = binary_rm_sml(clean_A,pix_area,8)
    clean_B = binary_rm_sml(clean_B,pix_area,8)
    
    # remove all connected subregions that are greater than 22.5 deg from center of largest subregion
    clean_AB = binary_rm_far(np.fmin(1,clean_A+clean_B).astype(int),pix_len,22.5)
    clean_A = (clean_A*clean_AB).astype(int)
    clean_B = (clean_B*clean_AB).astype(int)
    
    # smooth shape of connected subregions
    # clean_A = binary_closing(clean_A,structure=generate_binary_structure(2,2),iterations=int(np.round(2/l))).astype(int)
    # clean_B = binary_closing(clean_B,structure=generate_binary_structure(2,2),iterations=int(np.round(2/l))).astype(int)
    
    return clean_A,clean_B

def gaussian_clean(A,pix_len,blur_len):
    # Gaussian blur image with sigma = 1 deg
    return gaussian_filter(A,blur_len/pix_len)

def find_cent_idx(A):
    n = A.shape[0]
    
    # # get x and y positions of pixs
    # n = A.shape[0]
    # xs,ys = np.meshgrid(np.arange(n),np.arange(n))
    
    # # label all connected components
    # lab_A, nfeat_A = label(A,generate_binary_structure(2,2))
    # if nfeat_A==0:
    #     raise Exception('Image Empty')
    
    # # calculate size of all components
    # sizes = np.zeros(nfeat_A)
    # for feat in range(1,nfeat_A+1):
    #     sizes[feat-1] = np.sum(lab_A==feat)
    # big_feat = np.argmax(sizes) + 1
    
    # # calculate center of largest component
    # big_cent_x = int(np.round(np.mean(xs[lab_A==big_feat])+0.5))
    # big_cent_y = int(np.round(np.mean(ys[lab_A==big_feat])+0.5))
    
    # # Calculate pix edges closest to the largest component's center (in pixs)
    # xs,ys = np.meshgrid(np.arange(n),np.arange(n))
    # big_cent_x = int(np.round(np.mean(np.unique(xs[A==1]))+0.5))
    # big_cent_y = int(np.round(np.mean(np.unique(ys[A==1]))+0.5))
    
    # Calculate pix edges closest to the largest component's center of mass (in pixs)
    xs,ys = np.meshgrid(np.arange(n),np.arange(n))
    big_cent_x = int(np.round(np.mean(xs[A==1])+0.5))
    big_cent_y = int(np.round(np.mean(ys[A==1])+0.5))
    
    return big_cent_x,big_cent_y

def center_rescale_RF(A,B,pix_len,cent_idx,l_size=28,s_size=25,deg_res=0.5):    
    # center RFs in l_sizexl_size deg window
    clean_A = A[cent_idx[1]-int(np.round(l_size/2/pix_len)):cent_idx[1]+int(np.round(l_size/2/pix_len)),
                cent_idx[0]-int(np.round(l_size/2/pix_len)):cent_idx[0]+int(np.round(l_size/2/pix_len))]
    clean_B = B[cent_idx[1]-int(np.round(l_size/2/pix_len)):cent_idx[1]+int(np.round(l_size/2/pix_len)),
                cent_idx[0]-int(np.round(l_size/2/pix_len)):cent_idx[0]+int(np.round(l_size/2/pix_len))]
    
    # resize RFs to have 0.5 deg resolution
    clean_A = zoom(clean_A,pix_len/deg_res)
    clean_B = zoom(clean_B,pix_len/deg_res)
    
    # center RFs in s_sizexs_size deg window
    idx_cut = int(np.round((l_size-s_size)/2/deg_res))
    clean_A = clean_A[idx_cut:-idx_cut,idx_cut:-idx_cut]
    clean_B = clean_B[idx_cut:-idx_cut,idx_cut:-idx_cut]
    
    # remove pixels outside of s_size deg diameter circular window
    n = int(np.round(s_size/deg_res))
    xs,ys = np.meshgrid((np.arange(n)+0.5)*deg_res - s_size/2,
                        (np.arange(n)+0.5)*deg_res - s_size/2)
    in_rf = (np.sqrt(xs**2 + ys**2) <= s_size/2).astype(int)
    clean_A = clean_A*in_rf
    clean_B = clean_B*in_rf
    
    return clean_A,clean_B

def ellipse_feats(A):
    n = lab_A.shape[0]
    
    xs,ys = np.meshgrid(np.arange(n),np.arange(n))
    
    # label all connected components
    lab_A, nfeat_A = label(A,generate_binary_structure(2,2))
    
    # get region properties from labeled image
    props = regionprops(lab_A)
    
    # build labeled image of elliptical features
    ellipse = np.zeros_like(lab_A)
    for feat in range(nfeat_A):
        cent = props[feat]['centroid']
        a = props[feat]['axis_major_length']
        b = props[feat]['axis_minor_length']
        ori = np.pi/2-props[feat]['orientation']
        ellipse += (feat+1)*(((((xs-cent[1])*np.cos(ori)+(ys-cent[0])*np.sin(ori))/a)**2 +\
            ((-(xs-cent[1])*np.sin(ori)+(ys-cent[0])*np.cos(ori))/b)**2) < 0.5**2)
        
    return ellipse

def ellipse_feats_from_labeled(lab_A):
    n = lab_A.shape[0]
    xs,ys = np.meshgrid(np.arange(n),np.arange(n))
    
    # extract number of labels
    nfeat_A = np.max(lab_A)
    
    # get region properties from labeled image
    props = regionprops(lab_A)
    
    # build labeled image of elliptical features
    ellipse = np.zeros_like(lab_A)
    for feat in range(nfeat_A):
        cent = props[feat]['centroid']
        a = props[feat]['axis_major_length']
        b = props[feat]['axis_minor_length']
        ori = np.pi/2-props[feat]['orientation']
        ellipse += (feat+1)*(((((xs-cent[1])*np.cos(ori)+(ys-cent[0])*np.sin(ori))/a)**2 +\
            ((-(xs-cent[1])*np.sin(ori)+(ys-cent[0])*np.cos(ori))/b)**2) < 0.5**2)
        
    return ellipse

def ellipse_gauss_func(X, a, b, tht, A, x0, y0):
    x, y = X
    dx_rot = (x - x0) * np.cos(tht) + (y - y0) * np.sin(tht)
    dy_rot = -(x - x0) * np.sin(tht) + (y - y0) * np.cos(tht)
    return A * np.exp(-(dx_rot**2 / a**2 + dy_rot**2 / b**2))

def ellipse_gauss_fit(field):
    n = field.shape[0]
    x,y = np.mgrid[:n,:n]
    props = regionprops((field > 0).astype(int))[0]
    
    p0 = (props['major_axis_length']/2,
          props['minor_axis_length']/2,
          props['orientation'],
          np.max(field),
          props['centroid'][0],
          props['centroid'][1])
    
    popt,pcov = curve_fit(ellipse_gauss_func, np.array([x.flatten(), y.flatten()]), field.flatten(),
                          p0=p0, maxfev=1000000)
    
    popt[0] = np.abs(popt[0])
    popt[1] = np.abs(popt[1])
    if popt[0] < popt[1]:
        popt[0], popt[1] = popt[1], popt[0]
        popt[2] = popt[2] + np.pi/2
    popt[2] = np.mod(popt[2], np.pi)
    if popt[2] > np.pi/2:
        popt[2] -= np.pi
    
    popt[0] *= np.sqrt(8)
    popt[1] *= np.sqrt(8)
    
    return popt,pcov

def gaussprops(field):
    popt,_ = ellipse_gauss_fit(field)
    return {
        'axis_major_length': popt[0],
        'axis_minor_length': popt[1],
        'orientation': popt[2],
        'centroid': (popt[4], popt[5]),
        'area': np.pi * popt[0] * popt[1]
    }
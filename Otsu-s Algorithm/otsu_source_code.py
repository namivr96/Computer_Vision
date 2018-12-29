import numpy as np
import cv2
#function that implements otsu's algorithm 
def otsu(image,tri):
    L = 256
    p_tot = image.shape[0]*image.shape[1]
    p_count = np.zeros(L)
    pi = np.zeros(L)
    for ind in range (0,L):
        pi[ind] = ind  
        
    for a in range (0, L):
        t = np.where(image == a, 1, 0) 
        p_count[a] = np.sum(t)
    p_den = np.divide(p_count, p_tot)
    
    k = 0
    sigma_b = []
    for tri in range (0,tri):
        for i in range (0,L):
            w0 = np.sum(p_den[0:i])
            w1 = np.sum(p_den[i:L])
            
            m_num = np.multiply(pi,p_den)
    
            m_i0 = m_num[0:i]
            m_i1 = m_num[i:L]
            
            m0 = np.sum(m_i0)/w0
            m1 = np.sum(m_i1)/w1
            
            sigma_b.append(w0*w1*np.square(m1-m0))
            
        p_new = []
        k = sigma_b.index(max(sigma_b[1:L]))
    
        for r in range (0, image.shape[0]): 
            for c in range (0, image.shape[1]):
                if (image[r][c] > k) :
                    p_new.append(image[r][c])
        
        L = len(set(p_new))
        p_tot = len(p_new)
        p_count = np.zeros(L)
        pi = np.zeros(L)
        for ind in range (0,L):
            pi[ind] = ind  
        a = 0    
        for a in range (0, L):
            t = np.where(image == a, 1, 0) 
            p_count[a] = np.sum(t)
        p_den = np.divide(p_count, p_tot)
       
    mask = image.copy()
    for row in range (0, mask.shape[0]): 
        for col in range (0, mask.shape[1]):
            if (mask[row][col] > k ):
                mask[row][col] = 255
            else:
                mask[row][col] = 0
    return(mask)
#Function to implement RGB based image segmentation     
def seg_color(img_1, N):
    b = img_1.copy()
    b[:, :, 1] = 0
    b[:, :, 2] = 0
    b = cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
    g = img_1.copy()
    g[:, :, 0] = 0
    g[:, :, 2] = 0
    g = cv2.cvtColor(g,cv2.COLOR_BGR2GRAY)
    r = img_1.copy()
    r[:, :, 0] = 0
    r[:, :, 1] = 0 
    r = cv2.cvtColor(r,cv2.COLOR_BGR2GRAY)
    R = otsu(r,3)
    B = otsu(b,3)
    G = otsu(g,3)
    
    if (N==1):
        mask = R & (~B) & (~G)
        cv2.imwrite('LC_red.jpg', R)
        cv2.imwrite('LC_blue.jpg', G)
        cv2.imwrite('LC_green.jpg', B)
        cv2.imwrite('LC_overall.jpg', mask)
    if (N==2):
        mask = ~B & ~G
        cv2.imwrite('babyC_red.jpg', R)
        cv2.imwrite('babyC_blue.jpg', G)
        cv2.imwrite('babyC_green.jpg', B)
        cv2.imwrite('babyC_overall.jpg', mask)
    if (N==3):
        mask = R & (~G)|(~B)
        mask = mask & R
        cv2.imwrite('skiC_red.jpg', R)
        cv2.imwrite('skiC_blue.jpg', G)
        cv2.imwrite('skiC_green.jpg', B)
        cv2.imwrite('skiC_overall.jpg', mask)
    return (mask)
#Function for creating texture channels 
def createtext(image, image_new, N, o_tri):
    w = int(N/2)
    for row in range (w, image_new.shape[0]-w): 
        for col in range (w, image_new.shape[1]-w):
            image_new [row][col] = int(np.var(image[row-w:row+w+1,col-w:col+w+1]))
    mask = otsu(image_new,o_tri)
    return mask
#Function for texture based image segmentation
def seg_text(image,name):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    o_tri = 3
    image_new = np.zeros_like(image)
    N = 3
    m3 = createtext(image, image_new, N,o_tri)
    cv2.imwrite('%s_3.jpg'%(name), m3)

    
    image_new = np.zeros_like(image)
    N = 5
    m5 = createtext(image, image_new, N,o_tri)
    cv2.imwrite('%s_5.jpg'%(name), m5)

    
    image_new = np.zeros_like(image)
    N = 7
    m7 = createtext(image, image_new, N,o_tri)
    cv2.imwrite('%s_7.jpg'%(name), m7)
 
    mask = m3&m5&m7
    cv2.imwrite('%s_overall.jpg'%(name), mask)
    
    return mask
#Function for contour extraction     
def contour(image,name):
    image_new = np.zeros_like(image)
    for row in range (0, image_new.shape[0]-1): 
        for col in range (0, image_new.shape[1]-1):
            if (image[row][col]!= 0):
                if image[row-1][col] == 0 or image[row+1][col] == 0 or image[row][col-1] == 0 or image[row][col+1] == 0:
                    image_new[row][col] = 255
    cv2.imwrite('%s.jpg' %(name), image_new)

#Function calls for RGB and Texture based image segmentation            
img_1 = cv2.imread('lighthouse.jpg')
LC_overall = seg_color(img_1,1)
name = 'LC_contour'
contour (LC_overall,name)

img_2 = cv2.imread('baby.jpg')
bc_over = seg_color(img_2,2)
name = 'BC_contour'
contour (bc_over,name)

img_3 = cv2.imread('ski.jpg')
sc_over = seg_color(img_3,3)
name = 'SC_contour'
contour (sc_over,name)

img_1 = cv2.imread('lighthouse.jpg')
name1 = 'Ltext'
Lt_mask = seg_text(img_1,name1)
name = 'LText_contour7'
contour (Lt_mask,name)
    
img_2 = cv2.imread('baby.jpg')
name2 = 'babytext'
ski_t_mask = seg_text(img_2,name2)
name = 'babyText_contour'
contour (ski_t_mask,name)

img_3 = cv2.imread('ski.jpg')
name3 = 'skitext'
ski_t_mask = seg_text(img_3,name3)
name = 'skiText_contour'
contour (ski_t_mask,name)

'''
1.	Using Point to Point Correspondence
The source code below was used to remove the projective and the affine distortion the from the first image using point to point correspondence. The code used to remove the distortions from the other input images are very similar and hence it has not been included.
'''
import cv2
import numpy as np
import math

img_1 = cv2.imread('1.jpg', 1)

#Points from the distorted image
pt_img1 = np.array ([[814,1141,1], [759,1257,1], [990,1122,1], [942,1241,1]]) 
#Read world coordinates
pt_rw1 = np.array ([[814,1141,1], [814,1201,1], [894,1141,1], [894,1201,1]])

#Building the homography matrix
def homography (pt_f, pt_i):
T = np.zeros((8,8))
s = np.zeros((8,1))
for i in range (1,5):
T[2*i-2] = [pt_i[i-1][0],pt_i[i-1][1],1, 0, 0, 0, (-1*pt_i[i-1][0]*pt_f[i-1][0]), (-1*pt_i[i-1][1]*pt_f[i-1][0])]
T[2*i-1] = [0, 0, 0, pt_i[i-1][0], pt_i[i-1][1], 1, (-1*pt_i[i-1][0]*pt_f[i-1][1]), (-1*pt_i[i-1][1]*pt_f[i-1][1])]
s [2*i-2] = pt_f[i-1][0]
s [2*i-1] = pt_f[i-1][1]

H = np.zeros((3,3))
inv_T = np.linalg.inv(T)
h = np.matmul(inv_T, s)
H[0] = h[0:3,0]
H[1][0] = h[3][0]
H[1][1] = h[4][0]
H[1][2] = h[5][0]
H[2][0] = h[6][0]
H[2][1] = h[7][0]
H[2][2] = 1
return H 

H_1 = homography(pt_rw1, pt_img1)

 #Creating the bounding image
P = np.array([0,0,1])
Q = np.array([0,2592,1])
R = np.array([1944,0,1])
S = np.array([1944,2592,1])

p_h = np.matmul(H_1,P)
q_h = np.matmul(H_1,Q)
r_h = np.matmul(H_1,R)
s_h = np.matmul(H_1,S)

x_min = min (p_h[0]/p_h[2],q_h[0]/q_h[2],r_h[0]/r_h[2],s_h[0]/s_h[2])
x_max = max (p_h[0]/p_h[2],q_h[0]/q_h[2],r_h[0]/r_h[2],s_h[0]/s_h[2])
y_min = min (p_h[1]/p_h[2],q_h[1]/q_h[2],r_h[1]/r_h[2],s_h[1]/s_h[2])
y_max = max (p_h[1]/p_h[2],q_h[1]/q_h[2],r_h[1]/r_h[2],s_h[1]/s_h[2])

width = x_max - x_min
height = y_max - y_min 

#Creating a new image without the projective and affine distortions 
new = np.zeros((int(width), int(height), 3), dtype ='uint8' )

temp = np.ones(3)
for row in range (0, new.shape[0]-1): 
for col in range (0, new.shape[1]-1):
temp[0] = row + x_min
temp[1] = col+ y_min
new_rc = np.matmul(np.linalg.inv(H_1),temp)
new_rc = np.divide(new_rc,new_rc[2])
if (new_rc[0]>0) and (new_rc[0]<img_1.shape[0])and (new_rc[1]>0) and (new_rc[1]<img_1.shape[1]):
new[row][col] = img_1[math.floor(new_rc[0]),math.floor(new_rc[1])]
cv2.imwrite('t1_i3.jpg',new)
'''
2.	Two Step Method
The source code below was used to remove the projective and the affine distortion the from the first image using the two-step method. The code used to remove the distortions from the other input images are very similar and hence it has not been included.
'''
import cv2
import numpy as np
import math

#import img1
img_1 = cv2.imread('1.jpg', 1)

#Calculating the vanishing line
p_1 = np.array ([814,1141,1])
p_2 = np.array ([990,1122,1])
p_3 = np.array ([759,1257,1])
p_4 = np.array ([942,1241,1])

l_1 = np.cross(p_1,p_2)
l_2 = np.cross(p_3,p_4)
p_v = np.cross(l_1,l_2)

m_1 = np.cross(p_1,p_3)
m_2 = np.cross(p_2,p_4)
q_v = np.cross(m_1,m_2)

q_v = q_v/q_v[2]
p_v = p_v/p_v[2]

vl = np.cross(p_v,q_v)
vl = vl / vl[2]

#Calculating the homography matrix
H_p = np.zeros((3,3))
H_p[0][0] = 1
H_p[1][1] = 1
H_p[2][0] = vl[0]
H_p[2][1] = vl[1]
H_p[2][2] = vl[2]



p_h = np.matmul(H_p,P)
q_h = np.matmul(H_p,Q)
r_h = np.matmul(H_p,R)
s_h = np.matmul(H_p,S)

x_min = min (p_h[0]/p_h[2],q_h[0]/q_h[2],r_h[0]/r_h[2],s_h[0]/s_h[2])
x_max = max (p_h[0]/p_h[2],q_h[0]/q_h[2],r_h[0]/r_h[2],s_h[0]/s_h[2])
y_min = min (p_h[1]/p_h[2],q_h[1]/q_h[2],r_h[1]/r_h[2],s_h[1]/s_h[2])
y_max = max (p_h[1]/p_h[2],q_h[1]/q_h[2],r_h[1]/r_h[2],s_h[1]/s_h[2])

width = x_max - x_min
height = y_max - y_min 

#Scaling the image 
scale_x = img_1.shape[1]/width
scale_y = img_1.shape[0]/height 

w_scale = int (width*scale_x)
h_scale = int (height*scale_y)
new = np.zeros((w_scale, h_scale, 3), dtype ='uint8' ) 

# Removing the projective distortion 
 temp = np.ones(3)
 for row in range (0, new.shape[0]-1): 
for col in range (0, new.shape[1]-1):
temp[0] = row/scale_x + x_min
temp[1] = col/scale_y + y_min
new_rc = np.matmul(np.linalg.inv(H_p),temp)
new_rc = np.divide(new_rc,new_rc[2])
if (new_rc[0]>0) and (new_rc[0]<img_1.shape[0])and (new_rc[1]>0) and (new_rc[1]<img_1.shape[1]):
new[row][col] = img_1[math.floor(new_rc[0]),math.floor(new_rc[1])]
cv2.imwrite('task2_proj.jpg',new)

#Creating affine homography matrix 
B = np.zeros((2,2))

p_1 = np.matmul(np.linalg.inv(H_p),p_1)
p_2 = np.matmul(np.linalg.inv(H_p),p_2)
p_3 = np.matmul(np.linalg.inv(H_p),p_3)
p_4 = np.matmul(np.linalg.inv(H_p),p_4)

l_1 = np.cross(p_1,p_2)
m_1 = np.cross(p_2,p_4)
l_2 = np.cross(p_1,p_3)
m_2 = np.cross(p_3,p_4)

B[0][0] = l_1[0]*m_1[0]
B[1][0] = l_2[0]*m_2[0]
B[0][1] = l_1[0]*m_1[1]+l_1[1]*m_1[0]
B[1][1] = l_2[0]*m_2[1]+l_2[1]*m_2[0]
C = np.array([(-1)*l_1[1]*m_1[1],(-1)*l_2[1]*m_2[1]])
C = C.reshape(2,1)
S = np.matmul(np.linalg.inv(B),C)

X = np.zeros((2,2))
X[0][0] = S[0]
X[0][1] = S[1]
X[1][0] = S[1]
X[1][1] = 1

U, s, V = np.linalg.svd(X, full_matrices = True)

D = np.zeros((2,2))
D[0][0] = math.sqrt(s[0])
D[1][1] = math.sqrt(s[1])

A = np.matmul(U,np.matmul(D,np.transpose(U)))

H_a = np.zeros((3,3))
H_a[0][0] = A[0][0]
H_a[0][1] = A[0][1]
H_a[1][0] = A[1][0]
H_a[1][1] = A[1][1]
H_a[2][2] = 1

#Importing image without projective distortion
img_1 = cv2.imread('task2_proj.jpg', 1)

#The Bounding points of image 1
P = np.array([0,0,1])
Q = np.array([0,1943,1])
R = np.array([2592,0,1])
S = np.array([1944,2592,1])

p_h = np.matmul(H_new,P)
q_h = np.matmul(H_new,Q)
r_h = np.matmul(H_new,R)
s_h = np.matmul(H_new,S)

x_min = min (p_h[0]/p_h[2],q_h[0]/q_h[2],r_h[0]/r_h[2],s_h[0]/s_h[2])
x_max = max (p_h[0]/p_h[2],q_h[0]/q_h[2],r_h[0]/r_h[2],s_h[0]/s_h[2])
y_min = min (p_h[1]/p_h[2],q_h[1]/q_h[2],r_h[1]/r_h[2],s_h[1]/s_h[2])
y_max = max (p_h[1]/p_h[2],q_h[1]/q_h[2],r_h[1]/r_h[2],s_h[1]/s_h[2])

width = x_max - x_min
height = y_max - y_min 

#Scaling the image 
scale_x = img_1.shape[1]/width
scale_y = img_1.shape[0]/height 

w_scale = int (width*scale_x)
h_scale = int (height*scale_y)
new = np.zeros((w_scale, h_scale, 3), dtype ='uint8' ) 

# Removing affine distortion 
temp = np.ones(3)
for row in range (0, new.shape[0]-1): 
for col in range (0, new.shape[1]-1):
temp[0] = row/scale_x + x_min
temp[1] = col/scale_y + y_min
new_rc = np.matmul(np.linalg.inv(H_new),temp)
new_rc = np.divide(new_rc,new_rc[2])
if (new_rc[0]>0) and (new_rc[0]<img_1.shape[0])and (new_rc[1]>0) and (new_rc[1]<img_1.shape[1]):
new[row][col] = img_1[math.floor(new_rc[0]),math.floor(new_rc[1])]
cv2.imwrite('t2_img1_com.jpg',new)

'''
3.	One Step Method
The source code below was used to remove the projective and the affine distortion the from the first image using the one-step method. The code used to remove the distortions from the other input images are very similar and hence it has not been included.
'''
import cv2
import numpy as np
import math

#import image
img_1 = cv2.imread('10.jpg', 1)

#IMG 1
p_1 = np.array ([1114,236,1])
p_2 = np.array ([1749,102,1])
p_3 = np.array ([12,2208,1])
p_4 = np.array ([1397,2338,1])
q_1 = np.array ([1356,225,1])
q_2 = np.array ([1257,470,1])
q_3 = np.array ([1519,190,1])

l_1 = np.cross(p_1,p_2)
m_1 = np.cross(p_2,p_4)
l_1 = l_1/l_1[2]
m_1 = m_1/m_1[2]

l_2 = np.cross(p_1,p_3)
m_2 = np.cross(p_3,p_4)
l_2 = l_2/l_2[2]
m_2 = m_2/m_2[2]

l_3 = np.cross(q_1,q_2)
m_3 = np.cross(q_1,q_3)
l_3 = l_3/l_3[2]
m_3 = m_3/m_3[2]

l_4 = np.cross(q_1,q_2)
m_4 = np.cross(p_1,p_2)
l_4 = l_4/l_4[2]
m_4 = m_4/m_4[2]

l_5 = np.cross(q_1,q_3)
m_5 = np.cross(p_1,p_3)
l_5 = l_5/l_5[2]
m_5 = m_5/m_5[2]

B = np.zeros((5,5))

B[0][0] = l_1[0]*m_1[0]
B[0][1] = (l_1[0]*m_1[1]+l_1[1]*m_1[0])/2
B[0][2] = l_1[1]*m_1[1] 
B[0][3] = (l_1[0]*m_1[2]+l_1[2]*m_1[0])/2
B[0][4] = (l_1[1]*m_1[2]+l_1[2]*m_1[1])/2


B[1][0] = l_2[0]*m_2[0]
B[1][1] = (l_2[0]*m_2[1]+l_2[1]*m_2[0])/2
B[1][2] = l_2[1]*m_2[1]
B[1][3] = (l_2[0]*m_2[2]+l_2[2]*m_2[0])/2
B[1][4] = (l_2[1]*m_2[2]+l_2[2]*m_2[1])/2

B[2][0] = l_3[0]*m_3[0]
B[2][1] = (l_3[0]*m_3[1]+l_3[1]*m_3[0])/2
B[2][2] = l_3[1]*m_3[1]
B[2][3] = (l_3[0]*m_3[2]+l_3[2]*m_3[0])/2
B[2][4] = (l_3[1]*m_3[2]+l_3[2]*m_3[1])/2

B[3][0] = l_4[0]*m_4[0]
B[3][1] = (l_4[0]*m_4[1]+l_4[1]*m_4[0])/2
B[3][2] = l_4[1]*m_4[1]
B[3][3] = (l_4[0]*m_4[2]+l_4[2]*m_4[0])/2
B[3][4] = (l_4[1]*m_4[2]+l_4[2]*m_4[1])/2

B[4][0] = l_5[0]*m_5[0]
B[4][1] = (l_5[0]*m_5[1]+l_5[1]*m_5[0])/2
B[4][2] = l_5[1]*m_5[1]
B[4][3] = (l_5[0]*m_5[2]+l_5[2]*m_5[0])/2
B[4][4] = (l_5[1]*m_5[2]+l_5[2]*m_5[1])/2

C = np.array([(-1)*l_1[2]*m_1[2],(-1)*l_2[2]*m_2[2],(-1)*l_3[2]*m_3[2],(-1)*l_4[2]*m_4[2],(-1)*l_5[2]*m_5[2]])
C = C.reshape(5,1)

S = np.matmul(np.linalg.inv(B),C)

CC = np.zeros((3,3))
CC[0][0]= S[0][0]
CC[0][1]= S[1][0]/2
CC[0][2]= S[3][0]/2
CC[1][0]= S[1][0]/2
CC[1][1]= S[2][0]
CC[1][2]= S[4][0]/2
CC[2][0]= S[3][0]/2
CC[2][1]= S[4][0]/2
CC[2][2]= 1

X = np.zeros((2,2))
X[0][0] = CC[0][0]
X[0][1] = CC[0][1]
X[1][0] = CC[1][0]
X[1][1] = CC[1][1]

U, s, V = np.linalg.svd(X, full_matrices=False)

D = np.zeros((2,2))
D[0][0] = math.sqrt(s[0])
D[1][1] = math.sqrt(s[1])

A = np.matmul(U,np.matmul(D,V))
t = np.array([[CC[0][2]],[CC[1][2]]])

VT = np.matmul(np.linalg.inv(A),t)

H_new = np.zeros((3,3))
H_new[0][0] = A[0][0]
H_new[0][1] = A[0][1]
H_new[1][0] = A[1][0]
H_new[1][1] = A[1][1]
H_new[2][0] = VT[0][0]
H_new[2][1] = VT[1][0]
H_new[2][2] = 1

# The Bounding points of image 1
P = np.array([0,0,1])
Q = np.array([0,2592,1])
R = np.array([1944,0,1])
S = np.array([1944,2592,1])

P = np.array([0,0,1])
Q = np.array([0,1000,1])
R = np.array([750,0,1])
S = np.array([750,1000,1])

p_h = np.matmul(np.linalg.inv(H_new),P)
q_h = np.matmul(np.linalg.inv(H_new),Q)
r_h = np.matmul(np.linalg.inv(H_new),R)
s_h = np.matmul(np.linalg.inv(H_new),S)

x_min = min (p_h[0]/p_h[2],q_h[0]/q_h[2],r_h[0]/r_h[2],s_h[0]/s_h[2])
x_max = max (p_h[0]/p_h[2],q_h[0]/q_h[2],r_h[0]/r_h[2],s_h[0]/s_h[2])
y_min = min (p_h[1]/p_h[2],q_h[1]/q_h[2],r_h[1]/r_h[2],s_h[1]/s_h[2])
y_max = max (p_h[1]/p_h[2],q_h[1]/q_h[2],r_h[1]/r_h[2],s_h[1]/s_h[2])

width = x_max - x_min
height = y_max - y_min 

#Scaling the image 
scale_x = img_1.shape[1]/width
scale_y = img_1.shape[0]/height 
w_scale = int (width*scale_x)
h_scale = int (height*scale_y)
new = np.zeros((w_scale, h_scale, 3), dtype ='uint8' ) 

#Mapping Removing the projective & affine distortion
temp = np.ones(3)
for row in range (0, new.shape[0]-1): 
for col in range (0, new.shape[1]-1):
temp[0] = row/scale_x + x_min
temp[1] = col/scale_y + y_min 
new_rc = np.matmul(H_new,temp)
new_rc = np.divide(new_rc,new_rc[2])
if (new_rc[0]>0) and (new_rc[0]<img_1.shape[0])and (new_rc[1]>0) and (new_rc[1]<img_1.shape[1]):
new[row][new.shape[1]-col] = img_1[math.floor(new_rc[0]),math.floor(new_rc[1])]
cv2.imwrite('task3_img1.jpg',new)




















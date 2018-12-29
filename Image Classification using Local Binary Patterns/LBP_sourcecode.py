import math
import BitVector
import cv2
import numpy as np

#Function to classify test images
def test(name, hist_tot):
    label = np.zeros(5)
    for i in range (0,5):  
        image = cv2.imread('%s%d.jpg'%(name,i+1)) 
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        hist = LBP_texture(image)
        print('done with test hist %d'%(i+1))
        label[i] = NN_classifier(hist, hist_tot)
        print('label %d done'%(i+1) )
    return (label)
    
#Function to implement NN classifier
def NN_classifier (hist, hist_tot):
    label = np.zeros(5)
    E = np.zeros(len(hist_tot))
    for i in range(0, len(hist_tot)):
        E[i] = np.linalg.norm(hist - hist_tot[i])

    ind = np.argsort(E)[0:5]
    for j in range(0, len(ind)):
        if ind[j] < 20:
            label[0] = label[0] + 1
            
        elif  ind[j] > 20 and ind[j] < 40:               
            label[1] = label[1] + 1
            
        elif  ind[j] > 40 and ind[j] < 60:               
            label[2] = label[2] + 1
            
        elif  ind[j] > 60 and ind[j] < 80:               
            label[3] = label[3] + 1
            
        elif  ind[j] > 80 and ind[j] < 100:               
            label[4] = label[4] + 1
            
    return (np.argmax(label)+1)
	
#Function to create LBP histogram             
def LBP_texture(image):       
    R = 1                 
    P = 8                    
    hist = {h:0 for h in range(0, P+2)}
    hist_temp = np.zeros(10)
    for r in range(R,image.shape[0]-R):  
        for c in range(R,image.shape[1]-R):
            pattern =[]
            for p in range(0,P):
                d_u = float(R*math.cos(2*math.pi*p/P))
                d_v = float(R*math.sin(2*math.pi*p/P))
                img_p = weighted(r,c,image,d_u,d_v)
                if img_p < image[r][c]:
                    pattern.append(0)
                else:
                    pattern.append(1) 
            bv = BitVector.BitVector(bitlist=pattern)
            intval_cs = [int(bv << 1) for _ in range(P)]
            minbv = BitVector.BitVector(intVal=min(intval_cs),size=P)
            bvruns = minbv.runs()
            encode = 11             
            if len(bvruns) == 1 and bvruns[0][0] == '1':
                encode = P
            elif len(bvruns) == 1 and bvruns[0][0] == '0':
                encode = 0
            elif len(bvruns)>2:
                encode = P+1
            else:
                encode = len(bvruns[1])
            hist[encode] = hist[encode] +1
            hist_temp[encode] = hist_temp[encode] + 1 
    print(hist)
    return(hist_temp)

#Function to calculate gray scale value using bilinear interpolation
def weighted(row, col, image, d_u, d_v):
    thresh = 0.001 
    u = row + d_u
    v = col + d_v
    delta_u = u - int(u)
    delta_v = v - int(v)
    if (delta_u < thresh) and (delta_v < thresh):
       img_p = float(image[int(u)][int(v)])
    elif (delta_u < thresh):
        img_p = (1-delta_v)*image[int(u)][int(v)] + delta_v*image[int(u)][int(v)+1]
    elif (delta_v < thresh):
        img_p = (1-delta_u)*image[int(u)][int(v)] + delta_u*image[int(u)+1][int(v)]
    else:
        img_p = (1-delta_u)*(1-delta_v)*image[int(u)][int(v)]+ (1-delta_u)*delta_v*image[int(u)][int(v)+1]+ delta_u*delta_v*image[int(u)+1][int(v)+1]+delta_u*(1-delta_v)*image[int(u)+1][int(v)]
    return img_p

#Function to create LBP histogram for training images
def create_hist(name):
    hist_np = np.zeros((20,10))
    for i in range(0,20):
        image = cv2.imread('%s/%d.jpg'%(name,i+1)) 
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        hist_np[i] = LBP_texture(image)
        print('done with hist %d'%(i+1))
    return(hist_np)
        
beach = 'C:/Users/namra/Desktop/Fall 2018/ECE 661/hw7/imagesDatabaseHW7/training/beach'
building = 'C:/Users/namra/Desktop/Fall 2018/ECE 661/hw7/imagesDatabaseHW7/training/building'
car = 'C:/Users/namra/Desktop/Fall 2018/ECE 661/hw7/imagesDatabaseHW7/training/car'
mountain = 'C:/Users/namra/Desktop/Fall 2018/ECE 661/hw7/imagesDatabaseHW7/training/mountain'
tree = 'C:/Users/namra/Desktop/Fall 2018/ECE 661/hw7/imagesDatabaseHW7/training/tree'           

#Function calls to calculate LBP histogram for training images
hist_beach = create_hist(beach)
print('hist beach done')

hist_build = create_hist(building)
print('hist build done')

hist_car = create_hist(car)
print('hist car done')

hist_mount = create_hist(mountain)
print('hist mount done')

hist_tree = create_hist(tree)
print('hist tree done')

#Combining the LBP histogram for all training images
hist_tot = np.concatenate((hist_beach, hist_build, hist_car, hist_mount, hist_tree), axis=0)
print('hist_tot done')

beach_test = 'C:/Users/namra/Desktop/Fall 2018/ECE 661/hw7/imagesDatabaseHW7/testing/beach_'
build_test = 'C:/Users/namra/Desktop/Fall 2018/ECE 661/hw7/imagesDatabaseHW7/testing/building_'
car_test = 'C:/Users/namra/Desktop/Fall 2018/ECE 661/hw7/imagesDatabaseHW7/testing/car_'
mount_test = 'C:/Users/namra/Desktop/Fall 2018/ECE 661/hw7/imagesDatabaseHW7/testing/mountain_'
tree_test = 'C:/Users/namra/Desktop/Fall 2018/ECE 661/hw7/imagesDatabaseHW7/testing/tree_'

#Function calls to classify test images
beach_t= test(beach_test, hist_tot)
print('beach test')
print(beach_t)
build_t= test(build_test, hist_tot)
print('building test')
print(build_t)
car_t = test(car_test, hist_tot)
print('car test')
print(car_t)
mount_t = test(mount_test, hist_tot)
print('mountain test')
print(mount_t)
tree_t = test(tree_test, hist_tot)
print('tree test')
print(tree_t)



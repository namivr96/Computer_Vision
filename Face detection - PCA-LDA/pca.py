import numpy as np
import cv2
from sklearn import neighbors
import matplotlib.pyplot as plt
#Function to vectorize the training and testing images and calculate the mean 
def fetchimages(N, P, name):
    tot = N*P
    
    img1 = cv2.imread('C:/Users/namra/Desktop/Fall 2018/ECE 661/hw10/train/01_01.png')
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    
    tot_pt = img1.shape[0]*img1.shape[1]
    img_list = np.zeros((tot_pt, tot))
    
    for i in range(1,P+1):
        
        for j in range(1,N+1): 
            
            if j < 10 and i < 10:
                img = cv2.imread('C:/Users/namra/Desktop/Fall 2018/ECE 661/hw10/%s/0%d_0%d.png'%(name,i,j))
            elif (j > 10 or j == 10) and i < 10:
                img = cv2.imread('C:/Users/namra/Desktop/Fall 2018/ECE 661/hw10/%s/0%d_%d.png'%(name,i,j))
            elif  j < 10 and (i > 10 or i == 10):
                img = cv2.imread('C:/Users/namra/Desktop/Fall 2018/ECE 661/hw10/%s/%d_0%d.png'%(name,i,j))
            else:
                img = cv2.imread('C:/Users/namra/Desktop/Fall 2018/ECE 661/hw10/%s/%d_%d.png'%(name,i,j))
            
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            temp = np.reshape(gray,tot_pt)
            img_list[:,((i-1)*N)+(j-1)] = temp
        
    img_list_norm = np.linalg.norm (img_list, axis = 0)
    img_normalize = np.divide(img_list, img_list_norm)
    mean = np.average(img_normalize, axis = 1)#overall mean
    for i in range (len(img_list)):
        img_normalize[i,:] = img_normalize[i,:] - mean[i]
    return img_normalize
#Function that conducts eigen value decomposition eigen vectors of the projection matrix

def eig_vectors(img_list):
    C = np.matmul(np.transpose(img_list),img_list)
    w,v = np.linalg.eig(C)
    w_sort_ind = np.argsort(np.dot(w, -1))
    eig_vec = [v[:,i] for i in w_sort_ind]    
    W = np.dot(img_list, eig_vec)
    W_n = np.linalg.norm (W, axis = 0)
    W_norm = np.divide(W, W_n)
    
    neigh = 0
    for i in w:
        if i > 0.5:
            neigh = neigh + 1 
    return W_norm, neigh
#Function to  vectorize training images
N = 21
P = 30
tot = N*P
name = 'train'
img_list_train = fetchimages(N, P, name)
#Function call to get eigen vectors of the projection matrix
W_norm, neigh_train = eig_vectors(img_list_train)

name = 'test'
img_list_test = fetchimages(N, P, name)
#Function to label for the testing and training data 
train_y = np.zeros(tot)
for i in range(0, tot):
    train_y [i] = np.ceil((i/N))+1
	
# The following code projects the train and test data into a low dimensional space and classifies them using KNN neighbors  
acc_total = []
neigh_train = 29
for i in range (1, neigh_train):
    eig_sect = W_norm[:, 0:i]
    train_proj = np.zeros((i,tot))
    for j in range(0, tot):
        train_proj [:,j] = np.dot(np.transpose(eig_sect), img_list_train[:,j])
        
    test_proj = np.zeros((i,tot))
    for k in range(0,tot):
        test_proj [:,k] = np.dot(np.transpose(eig_sect), img_list_test[:,k])
    
    clf = neighbors.KNeighborsClassifier(n_neighbors=1)
    clf.fit(np.transpose(train_proj), train_y)
    Z = clf.predict(np.transpose(test_proj))
        
    corr = 1
    for c in range(len(Z)):
        if Z[c] == train_y[c]:
            corr = corr + 1
    #Calculating the accuracy
    acc = float(corr)/float(tot)
    acc_total.append(acc)

x_axis = []
for i in range (1, neigh_train):
    x_axis.append(i)
   #Plotting the accuracy of PCA
plt.rcParams.update({'font.size': 22})    
acc_plot = np.multiply(acc_total, 100)
plt.plot(x_axis, acc_plot, color='blue', linewidth=3, label='PCA')
plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
plt.show()


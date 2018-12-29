import numpy as np
import cv2
from sklearn import neighbors
import matplotlib.pyplot as plt
#Function to vectorize the training and testing images 
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
			
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#
            temp = np.reshape(gray,tot_pt)            
            img_list[:,((i-1)*N)+(j-1)] = temp
    
    img_list_norm = np.linalg.norm (img_list, axis = 0)
    norm_img_list = np.divide(img_list, img_list_norm)
    mean = np.mean(norm_img_list, axis = 1)
    return norm_img_list, mean
#Function that conducts eigen value decomposition of the Sb and calculate the matrix Z and the eigen vectors of the projection matrix
def project(img_list_train,mean_g ):
    N = 21
    P = 30
    mean_c = np.zeros((img_list_train.shape[0], P))
    for i in range (0,P):
        temp = img_list_train[:, i*N:i*N+N]
        mean_temp = np.mean(temp, axis = 1 )
        mean_c[:,i] = mean_temp
    mean_gc = mean_c    
    for i in range (0, mean_c.shape[0]):
        mean_gc[i,:] = mean_gc[i,:] - mean_g[i]
        
    SB = np.divide(np.matmul(mean_gc,np.transpose(mean_gc)), P)
    C = np.matmul(np.transpose(mean_gc),mean_gc)
    w,v = np.linalg.eig(C)
    
    w_ind = np.argsort(w)
    w_sort = [w[i] for i in w_ind]
    vec_sort = [v[:,i] for i in w_ind] 
    
    w_f = w_sort[1:]
    vec_n = vec_sort[1:]
    vec_f = np.reshape (np.transpose(vec_n),(30,-1))
    
    V = np.matmul(mean_gc,vec_f)#eigen vectors of SB
    
    Y = V
    db = np.diag(np.power(w_f,-0.5))
    Z = np.matmul(Y, db)
    
    x_mean = img_list_train
    x_temp = np.zeros((img_list_train.shape[0], N))
    for i in range (0,1): 
        temp2 = x_mean[:, i*N:i*N+N] 
        m_temp = mean_c[:,i]
        for j in range (0,N):
            diff = np.subtract(temp2[:,j],m_temp)
            x_temp[:,j] = diff
        x_mean[:, i*N:i*N+N] = x_temp
    Zx_mean = np.matmul(np.transpose(Z),x_mean)
    w,v = np.linalg.eig(np.matmul(Zx_mean,np.transpose(Zx_mean)))
    w_diag = np.diag(w)
    
    return w_diag, Z
#Function to  vectorize training images
img = cv2.imread('C:/Users/namra/Desktop/Fall 2018/ECE 661/hw10/train/01_01.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_size = len(img)*len(img[0])

N = 21
P = 30
tot = N*P
name = 'train'
img_list_train,mean_g = fetchimages(N, P, name)
#Function call to get matrix Z and the eigen vectors 
w, Z_proj = project(img_list_train,mean_g )

name = 'test'
img_list_test, mean_test = fetchimages(N, P, name)
train_y = np.zeros(tot)

#Function to label for the testing and training data 
for i in range(0, tot):
    train_y [i] = np.ceil((i/N))+1
    
acc_total = []

train_mean = img_list_train
for i in range(0, img_list_train.shape[0]):
    train_mean[i,:]  = img_list_train[i,:] - mean_g[i]

test_mean = img_list_test
for i in range(0, img_list_test.shape[0]):
    test_mean[i,:]  = img_list_test[i,:] - mean_test[i]

# The following code projects the train and test data into a low dimensional space and classifies them using KNN neighbors
for i in range (1, len(w[0])):
    eig_sect = w[:, 0:i]
    W_temp = np.matmul(Z_proj, eig_sect)
    W_temp_norm = np.linalg.norm (W_temp, axis = 0)
    W_norm = np.divide(W_temp, W_temp_norm) 
  
    train_proj = np.zeros((i,tot))
    for j in range(0, tot):
        train_proj [:,j] = np.dot(np.transpose(W_norm), train_mean[:,j])
        
    test_proj = np.zeros((i,tot))
    for k in range(0,tot):
        test_proj [:,k] = np.dot(np.transpose(W_norm), test_mean[:,k])
    
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
for i in range (1, len(w[0])):
    x_axis.append(i)
	
#Plotting the accuracy of LDA 
plt.rcParams.update({'font.size': 22})
acc_plot = np.multiply(acc_total, 100)
plt.clf() 
plt.plot(x_axis, acc_plot, color='green', linewidth=3, label='LDA')
plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
plt.xlabel('Number of eigen vectors')
plt.ylabel('Accuracy')
plt.show()


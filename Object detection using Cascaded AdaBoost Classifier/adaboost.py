import numpy as np
import cv2
from sklearn import neighbors
import matplotlib.pyplot as plt
import glob
#Function to calculate sum of pixels in a box
def sumrect(img, box):
    A = float(img[box[0],box[1]])
    B = float(img[box[0],box[1]+box[2]])
    C = float(img[box[0]+box[3],box[1]])
    D = float(img[box[0]+box[3],box[1]+box[2]])
    return (D+A-(C+B))
#Function to calculate Haar features for an image 
def haar (img_int):

    feat = []
	#Calcuculates horizontal features
    for h in range (1,21):
        for w in range (1,21):
            for i in range (0,21-h):
                for j in range (0,41-2*w):
                    box1 = [i,j,w,h]
                    box2 = [i,j+w,w,h]                        
                    diff1 = sumrect(img_int, box1)
                    diff2 = sumrect(img_int, box2)
                    diff_final = diff2-diff1
                    feat.append(diff_final)  
    #Calcuculates vertical features                   
    for h in range (1,11):
        for w in range (1,41):
            for i in range (0,21-2*h):
                for j in range (0,41-w):
                    box1 = [i,j,w,h]
                    box2 = [i+h,j,w,h]                        
                    diff1 = sumrect(img_int, box1)
                    diff2 = sumrect(img_int, box2)                       
                    diff_final = diff1 - diff2
                    feat.append(diff_final)
    return np.transpose(feat)
#Function to extract features from testing and training data
def feat_extract(name):
    num_pics = len(glob.glob(name))
    feature = np.zeros ((166000,num_pics))
    c = 0 
    for i in sorted(glob.glob(name)):    
        img = cv2.imread(i)      
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_int = cv2.integral(gray)
        feat = haar(img_int)
        feature[:,c] = feat
        c = c+1
        print (c)
    return feature, num_pics
#Function to combine features from positive and negative images
def feat_comb(pos, neg):
    comb = np.concatenate((pos,neg), axis=1)
    return comb
#Function to find the best weak classifier
def best_classifier(features, weights, labels, num_pos):
    num_feat = features.shape[0]
    num_img = features.shape[1]
        
    min_val = np.inf
    T_pos = np.sum(weights[0:num_pos])
    T_neg = np.sum(weights[num_pos:num_img])
    
    for i in range (0, num_feat):
        feat_ind = features[i]
        sort_idx = np.argsort(feat_ind)
        sort_weights = weights[sort_idx]
        sort_labels = labels[sort_idx]
        sort_feat = np.sort(feat_ind)
                
        S_pos = np.cumsum(sort_weights*sort_labels)
        S_neg = np.cumsum(sort_weights)-S_pos
        
        err = np.zeros((len(feat_ind),2))
        err[:,0] = S_pos + (T_neg - S_neg)
        err[:,1] = S_neg + (T_pos - S_pos) 
        err_arr = err.min(1)
        
        err_min = min(err_arr)
        
        err_ind_temp = np.where(err_arr==err_min)
        err_ind = err_ind_temp[0][0]
        
        
        if err_min < min_val:            
            min_val = err_min        
            result = np.zeros(num_img)
            result_temp = np.zeros(num_img)
            #calc result
            if err[err_ind,0] <= err[err_ind,1]:
                p = -1
                result_temp[err_ind:] = 1
                for j in range (0, len(result)):
                    result[sort_idx[j]] = result_temp[j]
            else:
                p = 1
                result_temp[:err_ind] = 1
                for j in range (0, len(result)):
                    result[sort_idx[j]] = result_temp[j]
            #calc theta
            if err_ind == 0:
                theta = sort_feat[0]
            elif err_ind == len(feat_ind) - 1:
                theta = sort_feat[err_ind]
            else:
                theta = ((sort_feat[err_ind]+sort_feat[err_ind+1])/2) 
            
            feat_idx = i 
            
    return feat_idx, theta, p, result, min_val
#Function to build a strong classifier 
def cascade(feat,num_pos,index,stage):
    num_neg = len(index) - num_pos
    num_img = len(index)
    features = feat[:,index]   
    weights = np.zeros(num_img)
    labels = np.zeros (num_img)
    for i in range (0, num_img):
        if i < num_pos:          
            weights[i] = 0.5/num_pos
            labels[i] = 1
        else:
            weights[i] = 0.5/num_neg 
    
    T = 100 #change T
    strong_class = np.zeros(num_img)
    alpha = np.zeros(T)
    h_res = np.zeros((num_img,T))
    pos_acc = []
    fp_acc = []
    theta_all  = []
    p_all = []
    feat_idx_all = []
    thresh_all = []
    
    rt = open('C:/Users/namra/Desktop/Fall 2018/ECE 661/hw10/result_file.txt','a')     
    print ("Stage no %d"%(stage))    
    rt.write("Stage no %f\n"%(stage))
    
    for j in range (0,T):
        weights = np.divide(weights,np.sum(weights))
        feat_idx, theta, p, result, min_err = best_classifier(features, weights, labels, num_pos)
        print ("After %d classifier"%(j))
        theta_all.append(theta)
        p_all.append(p)
        feat_idx_all.append(feat_idx)
        
        h_res [:,j] = result
        beta = min_err/(1-min_err)
        alpha[j] = np.log(1/beta)
        
        strong_class = np.dot(h_res[:, 0:j+1],alpha[0:j+1])
        thresh = min(strong_class[0:num_pos])
        thresh_all.append(thresh)
        for k in range (len(strong_class)):
            if (strong_class[k]> thresh)or(strong_class[k]==thresh) : 
                strong_class[k] = 1    
            else: 
                strong_class[k] = 0  
                
        p_acc = np.sum(strong_class[0:num_pos])/num_pos
        fp = np.sum(strong_class[num_pos:num_img])/num_neg   
        pos_acc.append(p_acc)
        fp_acc.append(fp)  

        print ("True positive rate")
        print (p_acc)
        print ("False positive rate")
        print (fp) 
        rt.write("classifier number : %f\n"%(j))
        rt.write("True detection positive rate : %f\n"%(p_acc))  
        rt.write("False positive rate : %f\n"%(fp))

        for l in range (len(weights)):
            if labels[l] != result[l]: temp = 1
            else : temp = 0 
            weights[l] = np.multiply(weights[l],np.power(beta, (1-temp)))
             
        if p_acc == 1 and fp <= 0.5 :#change later
            break
   
    rt.close()         
    np.savez('C:/Users/namra/Desktop/Fall 2018/ECE 661/hw10/data/feature_idx%d'%(stage),feat_idx_all)
    np.savez('C:/Users/namra/Desktop/Fall 2018/ECE 661/hw10/data/polarity%d'%(stage),p_all)
    np.savez('C:/Users/namra/Desktop/Fall 2018/ECE 661/hw10/data/theta%d'%(stage),theta_all)
    np.savez('C:/Users/namra/Desktop/Fall 2018/ECE 661/hw10/data/alpha%d'%(stage),alpha)  
    np.savez('C:/Users/namra/Desktop/Fall 2018/ECE 661/hw10/data/threshold%d'%(stage),thresh_all) 
    np.savez('C:/Users/namra/Desktop/Fall 2018/ECE 661/hw10/data/num_class%d'%(stage),j)#number of classifiers is that plus one
       
    sort_neg = np.sort(strong_class[num_pos:num_img])
    sort_neg_ind = np.argsort(strong_class[num_pos:num_img])
    new_neg = []
    for i in range (0,num_neg):
        if (sort_neg[i]>0):
            new_neg = sort_neg_ind[i:]
            break
    new_index = np.arange (0,num_pos,1)
    for i in range (len(new_neg)):
        new_index = np.append(new_index, [num_pos+int(new_neg[i])]) 
        
    return new_index, pos_acc, fp_acc]

#Function for classifying test images
def testing(features, feature_idx, p, theta, alpha, num_pos, num_neg):
    num_class = len(feature_idx)
    num_img = num_pos+num_neg
    result_w = np.zeros((num_img,num_class))
    
    for i in range (0,num_class):
        feat_temp = features[feature_idx[i],:]      
        for j in range (0,num_img):
            if p[i]*feat_temp[j]<=p[i]*(theta[i]):
                result_w[j,i] = 1

    result_s = np.dot(result_w, alpha)   
    thresh = 0.5*np.sum(alpha)

    for k in range (len(result_s)):
        if (result_s[k] > thresh)or(result_s[k]==thresh) : 
            result_s[k] = 1    
        else: 
            result_s[k] = 0
    fp = np.sum(result_s[num_pos:])/float(num_neg)
    fn = (num_pos - np.sum(result_s[:num_pos]))/float(num_pos)
    return result_s, fp, fn
	
#Function to extract and save features from training and test images
name = 'C:/Users/namra/Desktop/Fall 2018/ECE 661/hw10/ECE661_2018_hw10_DB2/train/positive/*.png'      
tr_pos_f, tr_num_pos = feat_extract(name)
print ("feature train positive done")
name = 'C:/Users/namra/Desktop/Fall 2018/ECE 661/hw10/ECE661_2018_hw10_DB2/train/negative/*.png'      
tr_neg_f, tr_num_neg = feat_extract(name)
    
name = 'C:/Users/namra/Desktop/Fall 2018/ECE 661/hw10/ECE661_2018_hw10_DB2/test/positive/*.png'      
te_pos_f, te_num_pos = feat_extract(name)
print ("feature train positive done")
name = 'C:/Users/namra/Desktop/Fall 2018/ECE 661/hw10/ECE661_2018_hw10_DB2/test/negative/*.png'      
te_neg_f, te_num_neg = feat_extract(name)
    
print ("feature train negative done")
np.save("train_positive",tr_pos_f)
np.save("train_negative",tr_neg_f)
np.save("test_positive",te_pos_f)
np.save("test_negative",te_neg_f)

#Loads features of training images
tr_pos_f = np.load("train_positive.npy")
tr_neg_f = np.load("train_negative.npy")
tr_num_pos = tr_pos_f.shape[1]
tr_num_neg = tr_neg_f.shape[1]

feat_train = feat_comb(tr_pos_f, tr_neg_f)
index = np.arange (0,(tr_num_pos+tr_num_neg),1)

#Function calls to create the adaBoost classifier
P_ACC = []
FP_ACC = []
S = 15 
for i in range (0, S):
     index, pos_acc, fp_acc = cascade(feat_train,tr_num_pos,index,i)
     P_ACC.append(pos_acc[-1])
     FP_ACC.append(fp_acc[-1])
     if len(index) == tr_num_pos:
         break 

#Loads features of test images
te_pos_f = np.load("test_positive.npy")
te_neg_f = np.load("test_negative.npy")
te_num_pos = te_pos_f.shape[1]
te_num_neg = te_neg_f.shape[1]

feat_test = feat_comb(te_pos_f, te_neg_f)  
stages = np.array([0,1,2,3,4,5,6])
FN = []
FP = []
R = []

#Function calls to classify test images 
for stage in stages:
    feature_idx_temp = np.load('C:/Users/namra/Desktop/Fall 2018/ECE 661/hw10/data/feature_idx%d.npz'%(stage))
    feature_idx = feature_idx_temp['arr_0']
   
    p_temp = np.load('C:/Users/namra/Desktop/Fall 2018/ECE 661/hw10/data/polarity%d.npz'%(stage))
    p = p_temp['arr_0']
    
    theta_temp = np.load('C:/Users/namra/Desktop/Fall 2018/ECE 661/hw10/data/theta%d.npz'%(stage))
    theta = theta_temp['arr_0']
    
    alpha_temp = np.load('C:/Users/namra/Desktop/Fall 2018/ECE 661/hw10/data/alpha%d.npz'%(stage))
    alpha = alpha_temp['arr_0']

    alpha = alpha[0:num_class]

    results, fp, fn = testing(feat_test, feature_idx, p, theta, alpha, te_num_pos, te_num_neg)
    R.append(results)
    FP.append(fp)
    FN.append(fn)

#Calculate cumulative false positive rates
fp_final = np.cumprod(FP)

#Calculate cumulative false positive rates
fn_rate = FN 
for i in range (len(fn_rate)):
    fn_rate[i] = 1 - fn_rate[i]

fn_final = 1 - np.cumprod(fn_rate)

#plot false positive and false negative rates for test images
plt.clf()
plt.rcParams.update({'font.size': 22})
x = np.array([1,2,3,4,5,6,7])
blue_line= plt.plot(x, fp_final, color='red', marker='.', markersize=15, label='False Positive')
plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
red_line= plt.plot(x, fn_final, color='green', marker='.', markersize=15, label='False Negative')
plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
plt.xlabel('Number of Stages')
plt.show()

#plot false positive rates for train images
train_fp = np.array([0.452787, 0.39500000,0.390071, 0.354271, 0.345455, 0.052632, 0.000000 ])
tot = np.cumprod(train_fp)
plt.clf()
x = np.array([1,2,3,4,5,6,7])
red_line= plt.plot(x, tot, color='green', marker='.', markersize=15, label='False Positive')
plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
plt.xlabel('Number of Stages')
plt.show()
    
    
    
    
    

    
            

        
    
    
    
    
    
    
    

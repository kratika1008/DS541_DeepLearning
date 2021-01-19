import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline

X_tr = np.load("mnist_train_images.npy")
ytr = np.load("mnist_train_labels.npy")
X_val = np.load("mnist_validation_images.npy")
yval = np.load("mnist_validation_labels.npy")
X_te = np.load("mnist_test_images.npy")
yte = np.load("mnist_test_labels.npy")

epochs = [20,25,30]
minibatch = [8]
learningRate = [.0003,.0002,0.0001]
regulatization = [0.0000000001]
hidden_units = [[50],[60]] #length of sub-list define number of hidden layers

best_model = {'epoch':0,
              'minibatch':0,
              'learningRate':0.,
              'regularization':0.,
              'num_hiddenLayers':0,
              'num_hiddenUnits':[]}
def initialize_weights(in_units,n_units,out_units):
    in_Weight=[]
    in_bias=[]
    
    for i in range(len(n_units)+1):
        m=in_units
        n=out_units
        if i==0:
            m=in_units
            if len(n_units)!=0:
                n=n_units[i]
        elif i==len(n_units):
            m=n_units[i-1]
        else:
            m=n_units[i-1]
            n=n_units[i]
        
        c=n**(-0.5)
        in_Weight.append(np.random.uniform(-c/2,c/2,(m,n)))
        in_bias.append(np.random.rand(1,n))#(np.full((1,n), 0.01))
    return in_Weight,in_bias

def relu(z):
    return np.maximum(np.zeros(z.shape),z)

def softmax(z):
    yhat = np.zeros(z.shape)
    m_z = np.amax(z)
    for r in range(z.shape[0]):
        yhat[r] = np.exp(z[r]-m_z)/np.sum(np.exp(z[r]-m_z))
    return yhat

def relu_prime(z):
    return np.where(z>0,1.,0.)

def compute_crossEntropy(y,yhat,reg,W,n):
    crossEntropy = np.sum(y*np.log(yhat))
    reg_Error=0.
    for i in range(len(W)):
        reg_Error += np.sum(np.dot(W[i].T,W[i]))
    fce = -crossEntropy/n + (reg*reg_Error/2)
    return fce
    
def feed_forward(X,W,b):
    z=[]
    h=[]
    
    for i in range(len(W)):
        if i==0:
            z.append(np.dot(X,W[i]) + b[i])
        else:
            z.append(np.dot(h[i-1],W[i]) + b[i])
        if i==(len(W)-1):
            h.append(softmax(z[i]))
        else:
            h.append(relu(z[i]))
    return z,h

def back_propagation(X,y,W,b,z,h,lr,reg):
    dz_dh=[]
    n = len(z)
    for i in range(n-1,-1,-1):
        j=n-1-i
        dz_dh.append(W[i])
        if (i==n-1):
            dJ_dz = h[-1]-y
            W[i] -= (lr*np.dot(h[i-1].T,dJ_dz))+(reg*W[i])
            b[i] -= lr*np.sum(dJ_dz,axis=0, keepdims = True)
        else:
            if i==0:
                dz_dw = X
            else:
                dz_dw = h[i-1]
                
            dh_dz = relu_prime(z[i])
            dz_db = np.ones((1,h[i].shape[0]))
            dJ_dz = np.multiply(np.dot(dJ_dz[j-1],dz_dh[j-1].T),dh_dz)
            
            W[i] -= (lr*np.dot(dz_dw.T,dJ_dz))+(reg*W[i])
            b[i] -= lr*np.dot(dz_db,dJ_dz)
    
    return W,b
    
def mnist_neuralNetwork(X_tr,ytr,epoch,n_batch,lr,reg,W,b):
    z=[]
    h=[]
    
    for e in range(epoch):
        bat=0
        for i in range(int(X_tr.shape[0]/n_batch - 1)):
            start=bat*n_batch
            end=(bat+1)*(n_batch) - 1
            X_train = X_tr[start:end]
            y_train = ytr[start:end]
            bat += 1
            n=X_train.shape[0]
            z,h = feed_forward(X_train,W,b)
            W,b = back_propagation(X_train,y_train,W,b,z,h,lr,reg)
    return W,b

def findBestHyperparameters():
    optimized_weights = []
    optimized_bias = []
    fce_minimum = 1e+10
    for epoch in epochs:
        for n_batch in minibatch:
            for lr in learningRate:
                for reg in regulatization:
                    for n_units in hidden_units:
                        print('----Model----')
                        print('Epoch:',epoch,',Batch size:',n_batch,',LearningRate:',lr,',Regularization:',reg,'NeuralNet:',len(n_units)+2,'layers, Hidden layer neurons:',n_units)
#                         n_units.insert(0,X_tr.shape[1])
#                         n_units.append(ytr.shape[1])
                        W,b = initialize_weights(X_tr.shape[1],n_units,ytr.shape[1])
                        weights,biases = mnist_neuralNetwork(X_tr,ytr,epoch,n_batch,lr,reg,W,b)
                        zval,hval = feed_forward(X_val,weights,biases)
                        val_correct_pred=0
                        yhat_val=hval[-1]
                        #print('hval',hval)
                        y_pred_val = np.zeros(yhat_val.shape)

                        for r in range(len(yhat_val)):
                            a=max(yhat_val[r])
                            y_pred_val[r] = np.where(yhat_val[r]==a,1,0)
                            if (yval[r]==y_pred_val[r]).all():
                                val_correct_pred += 1

                        acc = val_correct_pred/len(yval)
                        print("Validation Accuracy:",acc*100,'%')
                        fce_val = compute_crossEntropy(yval,hval[-1],reg,weights,len(X_val))
                        print("fce_val:",fce_val)
                        if fce_val<fce_minimum:
                            fce_minimum = fce_val
                            best_model['epoch']=epoch
                            best_model['minibatch']=n_batch
                            best_model['learningRate']=lr
                            best_model['regularization']=reg
                            best_model['num_hiddenLayers']=len(n_units)
                            best_model['num_hiddenUnits']=n_units
                            optimized_weights = weights
                            optimized_bias = biases
                        
    return fce_minimum,optimized_weights,optimized_bias

fce_minimum,optimized_weights,optimized_bias = findBestHyperparameters()
print(best_model)
print(fce_minimum)

def test_Model(X_te,yte,optimized_weights,optimized_bias):
    z_te,h_te = feed_forward(X_te,optimized_weights,optimized_bias)
    yhat_te = h_te[-1]
    y_pred_test = np.zeros(yhat_te.shape)
    fce_test = -np.sum(yte*np.log(yhat_te))/len(yte)
    correct_predictions=0
    
    for r in range(len(yhat_te)):
        y_pred_test[r] = np.where(yhat_te[r]==max(yhat_te[r]),1,0)
        if (yte[r]==y_pred_test[r]).all():
            correct_predictions += 1

    accuracy = correct_predictions/len(yte)
    return fce_test,accuracy
    
fce_test,accuracy = test_Model(X_te,yte,optimized_weights,optimized_bias)
print('--Test Error:',fce_test)
print('--Accuracy:',accuracy*100,'%')
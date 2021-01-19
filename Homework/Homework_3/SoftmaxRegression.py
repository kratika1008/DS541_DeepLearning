import numpy as np
from sklearn.model_selection import train_test_split

X_tr = np.load("mnist_train_images.npy")
ytr = np.load("mnist_train_labels.npy")
X_val = np.load("mnist_validation_images.npy")
yval = np.load("mnist_validation_labels.npy")
X_te = np.load("mnist_test_images.npy")
yte = np.load("mnist_test_labels.npy")

epochs = [200,300,400,500]
minibatch = [50,100,150,200]
learningRate = [0.005,0.001,0.05,0.01]
regulatization = [0.0001,0.001,0.005,0.01]

best_model = {'epoch':0,
              'minibatch':0,
              'learningRate':0.,
              'regularization':0.}

def compute_crossEntropy(y,yhat):
    crossEntropy = 0.
    for row in range(yhat.shape[0]):
        for col in range(yhat.shape[1]):
            crossEntropy += y[row,col]*np.log(yhat[row,col])
    return crossEntropy

def softmaxRegression_SGD(X_train,y_train,epoch,n_batch,lr,reg):
    w = np.random.rand(X_train.shape[1],10)
    b = np.random.rand(1,10)
    
    for e in range(epoch):
        bat=0
        for i in range(int(X_train.shape[0]/n_batch - 1)):
            start=bat*n_batch
            end=(bat+1)*(n_batch) - 1
            X = X_train[start:end]
            y = y_train[start:end]
            bat += 1
            param = X.shape
            z = np.matmul(X,w) + b
            yhat = np.zeros(z.shape)
            for r in range(z.shape[0]):
                yhat[r] = np.exp(z[r])/np.sum(np.exp(z[r]))
            crossEntropy_train = compute_crossEntropy(y,yhat)
            reg_Error=0.
            for k in range(yhat.shape[1]):
                reg_Error += np.matmul(w.T[k],w[:,k])

            fce_train = -crossEntropy_train/(param[0]) + (reg*reg_Error/2)
            
            gradient = np.matmul(X.T,(yhat-y))/param[0] + (reg*w)
            w = w - (lr*gradient)
            b = b - (lr*(np.sum(yhat-y)/param[0]))    
    return w,b



def tune_hyperparams():  
    optimized_weights = np.random.rand(X_tr.shape[1],10)
    optimized_bias = np.zeros((1,10))
    fce_minimum = 100000000
    for epoch in epochs:
        for n_batch in minibatch:
            for lr in learningRate:
                for reg in regulatization:
                    weights,bias = softmaxRegression_SGD(X_tr,ytr,epoch,n_batch,lr,reg)
                    z_val = np.matmul(X_val,weights) + bias
                    yhat_val = np.zeros(z_val.shape)
                    for r in range(z_val.shape[0]):
                        yhat_val[r] = np.exp(z_val[r])/np.sum(np.exp(z_val[r]))
                    crossEntropy_val = compute_crossEntropy(yval,yhat_val)
                    reg_Error_val=0.
                    for k in range(z_val.shape[1]):
                        reg_Error_val += np.matmul(weights.T[k],weights[:,k])

                    fce_val = -crossEntropy_val/(X_val.shape[0]) + (reg*reg_Error_val/2)
                    if fce_val<fce_minimum:
                        fce_minimum = fce_val
                        best_model['epoch']=epoch
                        best_model['minibatch']=n_batch
                        best_model['learningRate']=lr
                        best_model['regularization']=reg
                        optimized_weights = np.copy(weights)
                        optimized_bias = np.copy(bias)
    return fce_minimum,optimized_weights,optimized_bias

fce_minimum,optimized_weights,optimized_bias = tune_hyperparams()
print(best_model)
print(fce_minimum)

def test_Model(X_te,yte,optimized_weights,optimized_bias):
    z_test = np.matmul(X_te,optimized_weights) + optimized_bias
    yhat_test = np.zeros(z_test.shape)
    y_pred_test = np.zeros(z_test.shape)
    crossEntropy_test=0.
    correct_predictions=0
    for r in range(len(z_test)):
        yhat_test[r] = np.exp(z_test[r])/np.sum(np.exp(z_test[r]))
        a = max(yhat_test[r])
        y_pred_test[r] = np.where(yhat_test[r]==a,1,0)

    for row in range(yhat_test.shape[0]):
        for col in range(yhat_test.shape[1]):
            crossEntropy_test += yte[row,col]*np.log(yhat_test[row,col])
        if (yte[row]==y_pred_test[row]).all():
            correct_predictions += 1

    fce_test = -crossEntropy_test/(len(yte))
    accuracy = correct_predictions/len(yte)
    print(yte[0:3])
    print(yhat_test[0:3])
    print(y_pred_test[0:3])
    return fce_test,accuracy
    
fce_test,accuracy = test_Model(X_te,yte,optimized_weights,optimized_bias)
print('Test Error:',fce_test)
print('Accuracy:',accuracy*100,'%')
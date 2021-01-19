import numpy as np
from sklearn.model_selection import train_test_split

X_tr = np.reshape(np.load("../age_regression_Xtr.npy"), (-1, 48*48))
ytr = np.reshape(np.load("../age_regression_ytr.npy"), (-1, 1))
X_te = np.reshape(np.load("../age_regression_Xte.npy"), (-1, 48*48))
yte = np.reshape(np.load("../age_regression_yte.npy"), (-1, 1))

X_train, X_val, y_train, y_val = train_test_split(X_tr, ytr, test_size=0.2)

epochs = [5,10,15,20]
minibatch = [50,100,150,200]
learningRate = [0.1,0.002,0.001,0.0001]
regulatization = [0.01,0.005,0.001,0.05]

best_model = {'epoch':0,
              'minibatch':0,
              'learningRate':0,
              'regularization':0}


def linearRegression_SGD(X_train,y_train,epoch,n_batch,lr,reg):
    param = X_train.shape
    w = np.random.rand(param[1],1)
    b = np.zeros((1,1))
    for _ in range(epoch):
        bat=0
        for i in range(int(param[0]/n_batch - 1)):
            start=bat*n_batch
            end=(bat+1)*(n_batch) - 1
            X = X_train[start:end]
            y = y_train[start:end]
            bat += 1
            yhat = np.matmul(X,w) + b
            squaredError_train = np.sum(np.square(yhat-y))
            fmse_train = squaredError_train/(2*param[0]) + ((reg * np.matmul(w.T,w))/2)
            gradient = np.matmul(X.T,(yhat-y))/param[0] + (reg*w)
            w = w - (lr*gradient)
            b = b - (lr*(np.sum(yhat-y)/param[0]))
    return w,b
            
def tune_hyperparams():  
    optimized_weights = np.random.rand(X_train.shape[1],1)
    optimized_bias = np.zeros((1,1))
    fmse_minimum = 1e8
    for epoch in epochs:
        for n_batch in minibatch:
            for lr in learningRate:
                for reg in regulatization:
                    weights,bias = linearRegression_SGD(X_train,y_train,epoch,n_batch,lr,reg)
                    yhat_val = np.matmul(X_val,weights)+bias
                    squaredError_val = np.sum(np.square(yhat_val-y_val))
                    fmse_val = squaredError_val/(2*len(y_val)) + ((reg * np.matmul(weights.T,weights))/2)
                    if fmse_val<fmse_minimum:
                        fmse_minimum = fmse_val
                        best_model['epoch']=epoch
                        best_model['minibatch']=n_batch
                        best_model['learningRate']=lr
                        best_model['regularization']=reg
                        optimized_weights = np.copy(weights)
                        optimized_bias = np.copy(bias)
    return fmse_minimum,optimized_weights,optimized_bias


fmse_minimum,optimized_weights,optimized_bias = tune_hyperparams()
print(best_model)
print(fmse_minimum)

yhat_test = np.matmul(X_te,optimized_weights)+optimized_bias
squaredError_test = np.sum(np.square(yhat_test-yte))
fmse_test = squaredError_test/(2*len(yte))
print('Test Error:',fmse_test)

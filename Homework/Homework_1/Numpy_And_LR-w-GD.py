import numpy as np

#Part 1

def problem_a (A, B):
  return (A + B)

def problem_b (A, B, C):
  X = np.dot(A,B)
  return (X - C)

def problem_c (A, B, C):
  X = A*B
  Y = C.T
  return (X + Y)

def problem_d (x, y):
  return (np.dot(x.T,y))

def problem_e (A):
  B = np.zeros((A.shape))
  return B

def problem_f (A, x):
  return (np.linalg.solve(A, x))

def problem_g (A, x):
  inv_A = np.linalg.inv(A)
  return (np.dot(inv_A.T,x.T).T)

def problem_h (A, alpha):
  n = A.shape[0]
  I = np.eye(n)
  return (A + (alpha*I))

def problem_i (A, i, j):
  return (A[i-1,j-1])

def problem_j (A, i):
  return (np.sum(A[i-1,::2]))

def problem_k (A, c, d):
  return (np.mean(A[np.nonzero(np.logical_and(A>=c,A<=d))]))

def problem_l (A, k):
  val, vec = np.linalg.eig(A)
  sortK_val = np.argsort(val)[-k:][::-1]
  sortK_vec = vec[:,sortK_val]
  return sortK_vec

def problem_m (x, k, m, s):
  p = x.shape
  z = np.ones(p)
  I = np.eye(p[0])
  mu = x + (m*z)
  sigma = np.sqrt(s*I)
  return (np.dot(sigma, np.random.randn(p[0],k)) + mu)

def problem_n (A):
  rows = A.shape[0]
  per_rows = np.random.permutation(rows)
  return (A[per_rows])

#Part 2

def linear_regression (X_tr, y_tr):
  w = np.dot(np.linalg.inv(np.dot(X_tr.T,X_tr)),(np.dot(X_tr.T,ytr)))
  return w
def train_age_regressor ():
  #loading and reshaping training and testing datasets.
  X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
  ytr = np.load("age_regression_ytr.npy")
  X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
  yte = np.load("age_regression_yte.npy")
  #calculating w by calling linear_regression function and passing training data and labels
  w = linear_regression(X_tr, ytr)
  tr_error = 0.5*np.mean((np.dot(X_tr,w) - ytr)**2)
  te_error = 0.5*np.mean((np.dot(X_te,w) - yte)**2)
  #printing errors for both
  print("Training error ",tr_error)
  print("Testing error ",te_error)
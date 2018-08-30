import sys
import dlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from math import pi


# Make up a dataset of 2D vectors and plot it on the screen
np.random.seed(5)
pos = np.random.rand(300)*pi   
neg = -np.random.rand(50)*pi   
pos = np.stack((pos-pi/4, np.sin(pos)-0.2),axis=1) + np.random.randn(len(pos),2)*0.2
neg = np.stack((neg+pi/4, np.sin(neg)+0.2),axis=1) + np.random.randn(len(neg),2)*0.2
plt.scatter(pos[:,0], pos[:,1])
plt.scatter(neg[:,0], neg[:,1])
plt.show()


# Train the classifier
x = np.concatenate((pos,neg))
y = np.concatenate([np.repeat(1,pos.shape[0]), np.repeat(-1,neg.shape[0])])
df = dlib.auto_train_rbf_classifier(x,y,max_runtime_seconds=30)

print(dlib.test_binary_decision_function(df, x, y))


def plot_df(df):
    " plot the areas df classifies as + and -"
    n = 256 
    plt.figure()
    x = np.linspace(-4., 4., n) 
    y = np.linspace(-2., 2., n) 
    X, Y = np.meshgrid(x, y) 
    Z = X * np.sinc(X ** 2 + Y ** 2) 

    tmp = np.stack([X.reshape(n*n,1), Y.reshape(n*n,1)],axis=1).squeeze()
    plt.pcolormesh(X, Y, df.batch_predict(tmp).reshape(n,n)>0, cmap = cm.gray)

    plt.scatter(pos[:,0], pos[:,1])
    plt.scatter(neg[:,0], neg[:,1])



plot_df(df)

print("number of basis vectors: ", len(df.basis_vectors))

print(dlib.test_binary_decision_function(df, x,y))

# To make this fun faster we can also reduce the number of basis vectors.
df_compact = dlib.reduce(df,x, num_basis_vectors=10)

print("number of basis vectors in df_compact: ", len(df_compact.basis_vectors))
print(dlib.test_binary_decision_function(df_compact, x,y))

plot_df(df_compact)

sys.stdout.flush()
plt.show()

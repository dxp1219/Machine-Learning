# -*- coding: utf-8 -*
from _operator import inv
import numpy as np
import matplotlib.pyplot as plt 

# ��CSV�ļ�����Ϊһ��numpy����
data_file = open("3.0��.csv")
dataset = np.loadtxt(data_file, delimiter=",")
X = dataset[:,1:3]
y = dataset[:,3]
f1 = plt.figure(1)       
plt.title("3.0��")  
plt.xlabel("�ܶ�")  
plt.ylabel("������")  
plt.scatter(X[y == 0,0], X[y == 0,1], marker = "o", color = "k", s=100, label = "bad")
plt.scatter(X[y == 1,0], X[y == 1,1], marker = "o", color = "b", s=100, label = "good")
plt.legend(loc = "upper right")  
# plt.show()
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
import matplotlib.pyplot as plt
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=0)
lda_model = LinearDiscriminantAnalysis(solver = "lsqr", shrinkage = None).fit(X_train, y_train)
y_pred = lda_model.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
f2 = plt.figure(2) 
h = 0.001
x0_min, x0_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
x1_min, x1_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1
x0, x1 = np.meshgrid(np.arange(-1, 1, h),
                     np.arange(-1, 1, h))
x0, x1 = np.meshgrid(np.arange(x0_min, x0_max,h),
                     np.arange(x1_min, x1_max, h))
z = lda_model.predict(np.c_[x0.ravel(), x1.ravel()])
z = z.reshape(x0.shape)
plt.contourf(x0, x1, z)
plt.title("3.0��")  
plt.xlabel("�ܶ�")  
plt.ylabel("������")  
plt.scatter(X[y == 0,0], X[y == 0,1], marker = "o", color = "k", s=100, label = "bad")
plt.scatter(X[y == 1,0], X[y == 1,1], marker = "o", color = "b", s=100, label = "good")
# plt.show()
u = []  
for i in range(2):
    u.append(np.mean(X[y==i], axis=0))

m,n = np.shape(X)
Sw = np.zeros((n,n))
for i in range(m):
    x_tmp = X[i].reshape(n,1)
    if y[i] == 0: u_tmp = u[0].reshape(n,1)
    if y[i] == 1: u_tmp = u[1].reshape(n,1)
    Sw += np.dot( x_tmp - u_tmp, (x_tmp - u_tmp).T )

Sw = np.mat(Sw)
U, sigma, V= np.linalg.svd(Sw) 

Sw_inv = V.T * np.linalg.inv(np.diag(sigma)) * U.T

w = np.dot( Sw_inv, (u[0] - u[1]).reshape(n,1) )

print(w)

f3 = plt.figure(3)
plt.xlim( -0.2, 1 )
plt.ylim( -0.5, 0.7 )

p0_x0 = -X[:, 0].max()
p0_x1 = ( w[1,0] / w[0,0] ) * p0_x0
p1_x0 = X[:, 0].max()
p1_x1 = ( w[1,0] / w[0,0] ) * p1_x0

plt.title("3.0��--LDA")  
plt.xlabel("�ܶ�")  
plt.ylabel("������")  
plt.scatter(X[y == 0,0], X[y == 0,1], marker = "o", color = "k", s=10, label = "bad")
plt.scatter(X[y == 1,0], X[y == 1,1], marker = "o", color = "b", s=10, label = "good")
plt.legend(loc = "upper right")  

plt.plot([p0_x0, p1_x0], [p0_x1, p1_x1])

from self_def import GetProjectivePoint_2D 

m,n = np.shape(X)
for i in range(m):
    x_p = GetProjectivePoint_2D( [X[i,0], X[i,1]], [w[1,0] / w[0,0] , 0] ) 
    if y[i] == 0: 
        plt.plot(x_p[0], x_p[1], 'ko', markersize = 5)
    if y[i] == 1: 
        plt.plot(x_p[0], x_p[1], 'go', markersize = 5)   
    plt.plot([ x_p[0], X[i,0]], [x_p[1], X[i,1] ], 'c--', linewidth = 0.3)
    
# plt.show()

X = np.delete(X, 14, 0)
y = np.delete(y, 14, 0)

u = []  
for i in range(2): # two class
    u.append(np.mean(X[y==i], axis=0))

m,n = np.shape(X)
Sw = np.zeros((n,n))
for i in range(m):
    x_tmp = X[i].reshape(n,1)
    if y[i] == 0: u_tmp = u[0].reshape(n,1)
    if y[i] == 1: u_tmp = u[1].reshape(n,1)
    Sw += np.dot( x_tmp - u_tmp, (x_tmp - u_tmp).T )

Sw = np.mat(Sw)
U, sigma, V= np.linalg.svd(Sw) 

Sw_inv = V.T * np.linalg.inv(np.diag(sigma)) * U.T

w = np.dot( Sw_inv, (u[0] - u[1]).reshape(n,1) )

print(w)

f4 = plt.figure(4)
plt.xlim( -0.2, 1 )
plt.ylim( -0.5, 0.7 )

p0_x0 = -X[:, 0].max()
p0_x1 = ( w[1,0] / w[0,0] ) * p0_x0
p1_x0 = X[:, 0].max()
p1_x1 = ( w[1,0] / w[0,0] ) * p1_x0

plt.title("3.0��--LDA")  
plt.xlabel("�ܶ�")  
plt.ylabel("������")
plt.scatter(X[y == 0,0], X[y == 0,1], marker = "o", color = "k", s=10, label = "bad")
plt.scatter(X[y == 1,0], X[y == 1,1], marker = "o", color = "b", s=10, label = "good")
plt.legend(loc = "upper right")  

plt.plot([p0_x0, p1_x0], [p0_x1, p1_x1])

from self_def import GetProjectivePoint_2D 

m,n = np.shape(X)
for i in range(m):
    x_p = GetProjectivePoint_2D( [X[i,0], X[i,1]], [w[1,0] / w[0,0] , 0] ) 
    if y[i] == 0: 
        plt.plot(x_p[0], x_p[1], "ko", markersize = 5)
    if y[i] == 1: 
        plt.plot(x_p[0], x_p[1], "go", markersize = 5)   
    plt.plot([ x_p[0], X[i,0]], [x_p[1], X[i,1] ], "c--", linewidth = 0.3)

plt.show()
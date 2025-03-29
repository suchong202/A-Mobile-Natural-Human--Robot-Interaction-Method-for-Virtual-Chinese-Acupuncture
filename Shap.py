import openpyxl
from openpyxl import load_workbook
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import Turn1
import Turn2
import random
import Gate
import Attention
import LMF

def advance_step(S:[],i):
    W=S[:]
    W[i] = W[i] +1.0
    return  W

#Take a step back
def backward_step(S:[],i):
    W = S[:]
    W[i] = W[i] -1.0
    return  W

def select(E):

    n=len(E)

    random_number = random.randint(0, n-1)

    return E[random_number]

def bian(X, P):
    new_x = []
    for i in range(0, len(X)):
        x = []
        for j in range(0, len(P)):
            x.append(X[i][j] * P[j])
        new_x.append(x)
    return new_x

model_path = './Output/model.pkl'
model= pickle.load(open(model_path, 'rb'))

def Re(E, y_test):
    y_pred = model.predict(E)

    #print(y_test)
    #print(y_pred)

    accuracy = accuracy_score(y_test, y_pred)


    return accuracy




def getE (path1,path2,P):
    A1,A2=Turn1.shap_data(path1)
    A3,A4=Turn2.shap_data(path2)

    P1 = P[0:3]
    P2 = P[3:9]
    P3 = P[9:13]
    P4 = P[13:18]

    M1=[]
    M2=[]
    M3=[]
    M4=[]

    for j in range(0,200):
        M1.append(select(A1))
        M2.append(select(A2))
        M3.append(select(A3))
        M4.append(select(A4))

    M1 = bian(M1, P1)
    M2 = bian(M2, P2)
    M3 = bian(M3, P3)
    M4 = bian(M4, P4)


    P1 = Gate.gate(M1, M2)
    P2 = Gate.gate(M2, M1)
    P3 = Gate.gate(M3, M4)
    P4 = Gate.gate(M4, M3)

    C = np.hstack((P1, P2))
    D = np.hstack((P3, P4))

    C = Attention.attention(C, C)
    D = Attention.attention(D, D)
    #print(C,D)

    E=LMF.Lmf(C,D)

    return E



k=1
path1 = './Data/Cot/1/01.xlsx'
path2 = './Data/Video/1/03.avi'

P=[]
for i in range(0,18):
    P.append(1)
print(P)


Y = []
for i in range(0, 200):
    Y.append(k)


E = getE(path1, path2, P)
r = Re(E,Y)

for j in range(0,10):
    for i in range(0,len(P)):


        P1 = advance_step(P, i)
        P2 = backward_step(P,i)

        E1 = getE(path1, path2, P1)
        r1 = Re(E1, Y)

        E2 = getE(path1, path2, P2)
        r2 = Re(E2, Y)

        if r1>=r and r1>=r2:
            r=r1
            P=P1

        if r2>=r and r2>=r1:
            r=r2
            P=P2

        print(f"Accuracy: {r:.3f}")
        print(f"P: {P}")








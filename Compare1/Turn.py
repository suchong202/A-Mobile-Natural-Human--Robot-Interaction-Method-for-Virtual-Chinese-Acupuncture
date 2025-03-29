import openpyxl
import os
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import openpyxl
from openpyxl import load_workbook
import Gate
import Alexnets
import CNN
import Densnet
import Efficientnet
import LSTM
import Mobilenets
import Resnets50
import Transformer
import Vggs16
import Inceptionv2

path='Pic/Pic2/'
savepath='Excel2/Video8/'

for i in range(1,5):

    path1=savepath+str(i)+'.xlsx'

    wb = openpyxl.Workbook()
    sheet = wb.active

    name = ['Num']
    for j in range(1, 10):
        name.append(str(j))
    name.append('Type')

    sheet.append(name)

    print(path1)

    path2=path+str(i)

    print(path2)

    file = os.listdir(path2)
    n=0


    for f in file:
        path3 = os.path.join(path2, f)
        print(path3)
        A = []
        A.append(n)
        A=A+Inceptionv2.get(path3)
        A.append(int(i))
        sheet.append(A)
        n=n+1

    wb.save(path1)


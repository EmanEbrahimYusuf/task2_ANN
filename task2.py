from sklearn.utils import shuffle
import pandas as pd
import numpy as np


from classtask1 import TaskTwo
from classtask1 import confusion_matrix



def task(columsList,lable1,lable2,L,epochs,bias,th):

  data =pd.read_csv('penguins.csv')
  data['gender'].fillna('male',inplace=True)
  

  def genderlable(x):
    if x=='male':
      return 1
    else :return 0
  data['gender']=data['gender'].apply(genderlable)
  model=TaskTwo(data,columsList,lable1,lable2)
  #model=TaskOne(data,['bill_depth_mm', 'flipper_length_mm'],'Adelie','Chinstrap')
  model.fitAdaline(L,epochs,bias,th)
  print(model.score())
  print(model.confusionMatrix())
  model.draw()




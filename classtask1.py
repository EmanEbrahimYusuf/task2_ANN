from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error
import matplotlib.pylab as plt
import seaborn as sns
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

class TaskTwo:
 
    def __init__(self,data,columsList,lable1,lable2) -> None:
        self.weights=np.random.rand(len(columsList))
        self.bios=0
        self.columsList=columsList
        self.lable1=lable1
        self.lable2=lable2
        self.data=data
        self.columsList2=columsList
        self.trainData,self.testData=self.get_data()
        sc = MinMaxScaler()
        X=sc.fit_transform(self.trainData.drop('species',axis=1))
        self.sc_train = pd.DataFrame(X,columns=(self.trainData.drop('species',axis=1)).columns)
        self.sc_train['species'] = self.trainData['species'].values
        X2=sc.fit_transform(self.testData.drop('species',axis=1))
        self.sc_test = pd.DataFrame(X2,columns=(self.testData.drop('species',axis=1)).columns)
        self.sc_test['species'] =self.testData['species'].values
        print(len(self.trainData),len(self.testData))
 
    def get_data(self):
        d1=self.data[self.data['species']==self.lable1]
        d1trian,d1test=d1[:30],d1[30:50]
        d2=self.data[self.data['species']==self.lable2]
        d2trian,d2test=d2[:30],d2[30:50]
        datatrain=shuffle(pd.concat([d1trian,d2trian]))
        datatest=shuffle(pd.concat([d1test,d2test]))
 
        return datatrain,datatest
 
    def signum(self,net):
        if net>0:
            return 1
        elif net<0:
            return -1
        elif net == 0:
            return 0 
 
    def lable(self,x):
            if x==self.lable1:
                return 1
            else: return -1
 
    def fit(self,learnRate,epochs,biosed=True):
 
        self.trainData['species']=self.trainData['species'].apply(self.lable)
        self.trainData['species']=self.trainData['species'].astype('int64')
        for i in range(epochs):
            for X,Y in zip(self.trainData[self.columsList].values,self.trainData['species'].values):
                net=np.dot(self.weights.T,np.array(X))+self.bios
                Q=self.signum(net)
                loss=Y-Q
                self.weights=self.weights+learnRate*loss*X
                if biosed:
                    self.bios=self.bios+learnRate*loss
        return self.weights,self.bios
    
    def calc_error(self,X,Y):
        errors = []
        for x_i,y_i in zip(X,Y):
            net=np.dot(self.weights.T,np.array(x_i))+self.bios
            error=y_i - net
            errors.append(error)
        return (np.square(np.array(errors))).sum()/(2*len(X))
             

    def fitAdaline(self,learning_rate,epochs,biosed=True,threshold=0.01):
        
        self.sc_train['species']=self.sc_train['species'].apply(self.lable)
        self.sc_train['species']=self.sc_train['species'].astype('int64')
        X = self.sc_train[self.columsList].values
        Y = self.sc_train['species'].values
        # sc = StandardScaler()
        # sc_x=sc.fit_transform(X)
        for i in range(epochs):
            for x_i,y_i in zip(X,Y):
                net=np.dot(self.weights,x_i)+self.bios
                loss=y_i - net
                self.weights=self.weights+learning_rate*loss*x_i    
                if biosed:               
                    self.bios=self.bios+learning_rate*loss  
            
            
                
            MSE = self.calc_error(X,Y)
            if(i%10==0 ):
                print("epochs number: {0}".format(i))
                print("weights: {0} \nbias: {1}".format(self.weights,self.bios))
                print("Error : {0}".format(MSE))
            if(MSE <= threshold): break
        
        return self.weights,self.bios
 
 
    def predict(self):
        ydash=[]
        def lable(x):
            if x==self.lable1:
                return 1
            else: return -1
        self.sc_test['species']=self.sc_test['species'].apply(lable)
        self.sc_test['species']=self.sc_test['species'].astype('int64')
        
        for X in self.sc_test[self.columsList].values:
                net=np.dot(self.weights.T,np.array(X))+self.bios
                ydash.append(self.signum(net))
 
        return np.array(ydash)
 
    def score(self):
        self.pred=self.predict()
        sumCorrectItems=0
        for i,j in zip(self.pred,self.sc_test['species']):
            if i==j:
                sumCorrectItems+=1
        return sumCorrectItems/len(self.sc_test)
 
    def draw(self):
        np.random.seed(19680801)
        df1=self.sc_test[self.sc_test["species"]==1]
        df2=self.sc_test[self.sc_test["species"]==-1]
        plt.scatter(df1[self.columsList[0]],df1[self.columsList[1]],c='red')
        plt.scatter(df2[self.columsList[0]],df2[self.columsList[1]],c='blue')
        x1_1=int(min(self.sc_test[self.columsList[0]]))
        x1_2=(-self.bios-(x1_1*self.weights[0]))/self.weights[1]
 
        x2_1=int(max(self.sc_test[self.columsList[0]]))
        x2_2=(-self.bios-(x2_1*self.weights[0]))/self.weights[1]
 
        plt.plot([x1_1,x2_1],[x1_2,x2_2])
 
        plt.xlabel(self.columsList[0])
        plt.ylabel(self.columsList[1])
 
        # naming the title of the plot
        plt.title("Plot between {0} & {1}".format(self.columsList[0],self.columsList[1]))
        plt.show()
 
 
    def confusionMatrix(self):
        TP=0
        FP=0
        FN=0
        TN=0
 
        for i in range(len(self.testData)):
            if(self.pred[i]==1 and self.sc_test['species'].values[i]==1):
                TP+=1
            elif(self.pred[i]==-1 and self.sc_test['species'].values[i]==-1):
                TN+=1
            elif(self.pred[i]==1 and self.sc_test['species'].values[i]==-1):
                FP+=1
            elif(self.pred[i]==-1 and self.sc_test['species'].values[i]==1):
                FN+=1
 
        return np.array([[TP,FP],[FN,TN]])
 
 
    def line(self):
        x1_1=random.randint(1,20)
        x1_2=(-self.bios-(x1_1*self.weights[0]))/self.weights[1]
 
        x2_1=random.randint(5,25)
        x2_2=(-self.bios-(x2_1*self.weights[0]))/self.weights[1]
        #plt.plot([x1_1,x2_1],[x1_2,x2_2])
        #plt.show()
        return [(x1_1,x1_2),(x2_1,x2_2)]



#task(['flipper_length_mm', 'bill_depth_mm'],'Adelie','Chinstrap',0.01,4000,True)



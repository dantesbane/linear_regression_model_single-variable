#this code wasn't working at the start because i took the wrong value of alpha, note to self always modify the value of alpha first
#before going to stackoverflow
#reduce value of alpha if you get overflow in double_scalar error


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd                      #will use this for data cleaning 

dataframe=pd.read_csv(r"datasets\train.csv")    #reading the csv file use raw string to prevent utf-8 encoding from kicking in 
print(dataframe.isna().any())                   #check for null values in the dataframe 
print(dataframe.isna().sum())                   #check total number of null values in each column in the dataframe 
dataframe.dropna(inplace=True)                  #drop the rows with null values in the dataframe 
print(dataframe.isna().sum())                   #check if any null values are still there in the dataframe 

x,y=[dataframe["x"],dataframe["y"]]   
x=np.array(x);y=np.array(y)                     #convert the dataframe column into an array 
#print(type(x[0]))


def deri(x:list,y:list,w:float,b:float):     #function to find the derivative of j(w,b) wrt w and b 
    m=len(x);dw,db=0.0,0.0
    for i in range (m):
        fwb=w*x[i]+b
        #print(fwb)
        dw=dw+(fwb-y[i])*x[i]
        db=db+(fwb-y[i])
        
    dw=dw/m;db=db/m
    return dw,db

def gradient(x,y,w,b,a,iters):  #gradient descent algorithm implemented 
    for i in range (iters):
        dw,db=deri(x,y,w,b)
        if dw==0:
            return w,b
        w=w-a*(dw);b=b-a*(db)
    return w,b

def predicted(x,w,b):      #finding the value of predicted value according to the training data 
    m=x.shape[0];y=[]
    for i in range(m):
        y.append(w*x[i]+b)
    return y

def graph(x,y,w,b,a,iters):
    plt.scatter(x,y,c="r")
    w,b=gradient(x,y,w,b,a,iters)
    pred=predicted(x,w,b)
    plt.plot(x,pred,c="b")
    plt.show()

graph(x,y,0.0,0.0,0.00055,1000)    #any value below 0.00055 for alpha will work for this algorithm.
                                    #if you use 0.0006 then the algorithm will become divergent and you will get            
                                    #nan values and overflow for double scalar error 
import numpy as np
import random
import pylab
class Perce_cell:
    def data_build(self,size=[0,0],peace=False,sdv0=[-5,0],sdv1=[5,0]):
        np.random.seed(0)
        if(peace!=False):
            train_data=np.random.normal(size=size)
            train_label=np.sign(np.random.normal(size=size[0])*random.random())
            test_data=np.random.normal(size=[100,size[1]])
            test_label=np.sign(np.random.normal(size=100)*random.random())
            return train_data,train_label
        train_datahalf=np.random.normal(loc=sdv0[0],scale=1,size=[100,1])
        train_data=np.hstack([train_datahalf,np.random.normal(loc=sdv0[1],scale=1,size=[100,1])])
        train_datahalf=np.random.normal(loc=sdv1[0],scale=1,size=[100,1])
        train_datahalf=np.hstack([train_datahalf,np.random.normal(loc=sdv1[1],scale=1,size=[100,1])])
        return np.vstack([train_data,train_datahalf]),np.hstack([np.ones(100),np.ones(100)*-1])
    def __showpic(self,train_data,w,title):
        line_x=[-w[1],w[1]]
        line_y=[-w[0],w[0]]
        pylab.figure()
        pylab.title(title)
        pylab.scatter(x=train_data[:,0],y=train_data[:,1],c=train_label)
        pylab.plot(line_x,line_y)
        pylab.show()
    def train(self,train_data,train_label,step=None,title=None):
        w=np.random.random(size=train_data.shape[1])
        num=0
        step_max=step
        if step==None:
            step_max=10000
        step=0
        while(1):
            local=step%(train_data.shape[0])
            if(step!=0 and local==0):
                if num==0:
                    print('train:%d'%step)
                    break
                num=0;
            if np.sign(w.dot(train_data[local].reshape(train_data.shape[1],1)))[0] !=train_label[local]:
                num=num+1
                w=w+train_label[local]*train_data[local]
            if(step>step_max):
                print('find not')
                break
            step=step+1
        if title!=None:
            self.__showpic(train_data,w,title)
        return w
cell=Perce_cell()
print("*********************************************homework2***********************************************")
train_data,train_label=cell.data_build(peace=False,size=[100,10],sdv0=[-5,1],sdv1=[5,1])
w=cell.train(train_data=train_data,train_label=train_label,step=10000,title='X:data sheet--homework2')
print(w)
train_data,train_label=cell.data_build(peace=False,size=[100,10],sdv0=[-5,1],sdv1=[5,1])
w=cell.train(train_data=train_data,train_label=train_label,step=10000,title='X\':data sheet--homework2')
print(w)
print("*********************************************homework3***********************************************")
train_data,train_label=cell.data_build(peace=False,size=[100,10],sdv0=[-2,1],sdv1=[2,1])
w=cell.train(train_data=train_data,train_label=train_label,step=10000,title='X:data sheet--homework3')
print(w)
train_data,train_label=cell.data_build(peace=False,size=[100,10],sdv0=[-2,1],sdv1=[2,1])
w=cell.train(train_data=train_data,train_label=train_label,step=10000,title='X\':data sheet--homework3')
print(w)
print("*********************************************homework4***********************************************")
train_data,train_label=cell.data_build(peace=False,size=[100,10],sdv0=[-1,1],sdv1=[1,1])
w=cell.train(train_data=train_data,train_label=train_label,step=10000,title='X:data sheet--homework4')
print(w)
train_data,train_label=cell.data_build(peace=False,size=[100,10],sdv0=[-1,1],sdv1=[1,1])
w=cell.train(train_data=train_data,train_label=train_label,step=10000,title='X\':data sheet--homework4')
print(w)
print('*****************************************************homework5*************************************************')
print('可以看出，随着数据及出现了交互点，感知器就不能用了，这个仅仅适用于数据线性能分')
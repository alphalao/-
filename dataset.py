import numpy as np
import random
import pylab
def data_build(size=[100,2],sdv=None,label=("random",[-1,1,5]),dim=0):
    #size:数据尺寸；sdv:协方差；dim:数据偏移；
    #库说明：数据维度由size控制，sdv要与size一致
    def data_struct(size,sdv):
        datahalf=np.random.normal(loc=sdv[0],scale=1,size=[size[0],1])
        for b in range(1,size[-1]):
            datahalf=np.hstack([datahalf,np.random.normal(loc=sdv[b],scale=1,size=[size[0],1])])
        return datahalf
    if sdv==None:
        sdv=[[0]*size[-1]]*(len(size)-1)
    base_data=np.ones([sum(size)-size[-1],dim])
    datalabel=np.array([[label[1][0]]]*int((sum(size)-size[-1])/len(label[1])))
    data=data_struct([size[0],size[-1]],sdv[0])
    for b in range(1,len(size)-1):
        data=np.vstack([data,data_struct([size[b],size[-1]],sdv[b])])
    for b in range(1,len(label[1])-1):
        datalabel=np.vstack([datalabel,np.array([[label[1][b]]]*int((sum(size)-size[-1])/len(label[1])))])
    datalabel=np.vstack([datalabel,np.array([[label[1][-1]]]*(sum(size)-size[-1]-len(datalabel)))])
    if(label[0]=="random"):
        np.random.shuffle(datalabel)
    return np.hstack([base_data,data]),datalabel
def showpic(train_data,w,title,train_label,px):
    line_x=[-w[1]*px,w[1]*px]
    line_y=[-w[0]*px,w[0]*px]
    pylab.figure()
    pylab.title(title)
    pylab.scatter(x=train_data[:,0],y=train_data[:,1],c=train_label.T[0])
    pylab.plot(line_x,line_y)
    pylab.show()
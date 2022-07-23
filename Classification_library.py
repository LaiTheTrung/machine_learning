import numpy as np
class BinaryClassification():
    def __init__(self,Xs,Ys,lr,rl=0):
        self.Xs=self.transform(Xs)
        self.Ys=Ys
        self.lr=lr
        self.rl=rl
        self.n,self.m=self.Xs.shape
        self.theta=np.random.uniform(low=0.1, high = 0.5, size=(self.n,1))

    def transform(self,Xs,eval=True):
      
        Xs = self.normalize(Xs,eval)
        n,m = Xs.shape
        Xs_sqrt,Xs_power2=self.polynomial(Xs)     
        Xs = np.concatenate((np.ones([1,m]),Xs,Xs_sqrt,Xs_power2))
        return Xs

    def normalize(self,Xs,nor=True):
        n,m = Xs.shape
        if nor==True:
            self.mean_value=np.mean(Xs, axis=1).reshape(n,1)
            max_value = np.amax(Xs,axis=1).reshape(n,1)
            min_value = np.amin(Xs,axis=1).reshape(n,1)
            self.rangeMtx = max_value-min_value
        return (Xs-self.mean_value)/self.rangeMtx +1

    def polynomial(self,Xs):
        Xs_sqrt= np.sqrt(Xs)
        Xs_power2=np.float_power(Xs,2)
        return Xs_sqrt,Xs_power2
    
    def hypothesis(self,Xs):
        z= np.matmul(self.theta.transpose(),Xs)  #1,m
        hx= self.sigmoid(z)
        return hx #1,m
    
    def sigmoid(self,z):
        g = 1/(1+np.e**(-z))
        return g

    def CostFunc(self):
        hx = self.hypothesis(self.Xs)
        cost = -self.Ys*np.log(hx)-(1-self.Ys)*np.log(1-hx)
        regularization_decrease=self.regularization(self.theta)
        J = np.mean(cost)-regularization_decrease     
        return J
    
    def regularization(self,theta):
        return (self.rl/2)*np.mean(theta**2)

    def reAssign(self):
        hx = self.hypothesis(self.Xs)
        regularization_theta0=np.array([0]).reshape(1,1)
        regularization_reduce= np.concatenate(np.array([0]),(self.rl/self.m)*self.theta[1:])
        reduce_temp = self.lr*np.mean((hx-self.Ys)*self.Xs, axis=1).reshape(self.n,1)+regularization_reduce
        self.theta -= reduce_temp
    
    
    def predict(self,Xstest):
        Xstest=self.transform(Xstest,eval=False)
        hx=self.hypothesis(Xstest)
        prob=hx
        threshold=0.5
        pred= prob/threshold
        pred=pred.astype(int)
        return prob, pred
    
    def evaluate(self,Xstest,Ystest):
        prob,pred=self.predict(Xstest)
        error_case = np.sum(np.absolute(pred-Ystest))
        pred_len=len(pred.flatten())
        accuracy = (pred_len-error_case)/pred_len
        return accuracy

########################################################################################     
# 
#   
class OneVsAllClassification():
    def __init__(self,Xs,Ys,lr,rl=0):
        self.Xs=self.transform(Xs)
        self.Ys=Ys
        self.lr=lr
        self.rl=rl
        self.n,self.m=self.Xs.shape
        self.values = np.unique(self.Ys)
        self.list_theta=[]
        self.list_Y=[]
        for value in self.values:
            Y=self.Ys.copy()
            sample_theta=np.random.uniform(low=0.1, high = 0.5, size=(self.n,1))
            sample_Y=self.convert_Y(Y,value)
            self.list_theta.append(sample_theta)
            self.list_Y.append(sample_Y)

    def convert_Y(self,Ys,i):
        Ys[Ys==i]=0
        Ys[Ys!=0]=1
        return Ys

    def transform(self,Xs,eval=True):     
        Xs = self.normalize(Xs,eval)
        n,m = Xs.shape
        Xs_sqrt,Xs_power2=self.polynomial(Xs)     
        Xs = np.concatenate((np.ones([1,m]),Xs))
        return Xs

    def normalize(self,Xs,nor=True):
        n,m = Xs.shape
        if nor==True:
            self.mean_value=np.mean(Xs, axis=1).reshape(n,1)
            max_value = np.amax(Xs,axis=1).reshape(n,1)
            min_value = np.amin(Xs,axis=1).reshape(n,1)
            self.rangeMtx = max_value-min_value
        return (Xs-self.mean_value)/self.rangeMtx +1

    def polynomial(self,Xs):
        Xs_sqrt= np.sqrt(Xs)
        Xs_power2=np.float_power(Xs,2)
        return Xs_sqrt,Xs_power2
    
    def hypothesis(self,Xs,cur_theta):
        z= np.matmul(cur_theta.transpose(),Xs)  #1,m
        hx= self.sigmoid(z)
        return hx #1,m
    
    def sigmoid(self,z):
        g = 1/(1+np.e**(-z))
        return g

    def CostFunc(self,Ys,cur_theta):
        hx = self.hypothesis(self.Xs,cur_theta)
        cost = -Ys*np.log(hx)-(1-Ys)*np.log(1-hx)
        regularization_decrease = self.regularization(cur_theta)
        J = np.mean(cost)-regularization_decrease    
        return J

    def regularization(self,theta):
        return (self.rl/2)*np.mean(theta**2)  

    

    def TotalCostFunc(self):
        Total_Cost=[]
        for i in range(len(self.values)):
            loss = self.CostFunc(self.list_Y[i],self.list_theta[i])
            Total_Cost.append(loss)
        return Total_Cost


    def reAssign(self,Ys,cur_theta):
        hx = self.hypothesis(self.Xs,cur_theta)
        reduce_temp = self.lr*np.mean((hx-Ys)*self.Xs, axis=1).reshape(self.n,1)+(self.rl/self.m)*cur_theta
        cur_theta -= reduce_temp
        return cur_theta
    
    def TotalReAssign(self):       
        for i in range(len(self.values)):
            cur_theta = self.list_theta[i].copy()
            self.list_theta[i] = self.reAssign(self.list_Y[i],cur_theta)
    
    def predict(self,Xstest,cur_theta):
        Xstest=self.transform(Xstest,eval=False)
        hx=self.hypothesis(Xstest,cur_theta)
        prob=hx
        threshold=0.5
        pred= prob/threshold
        pred=pred.astype(int)
        return prob, pred
    
    def TotalPredict(self,Xstest):
        dt_prob=[]
        dt_pred=[]
        for i in range(len(self.values)):
            cur_theta = self.list_theta[i].copy()
            prob,pred=self.predict(Xstest,cur_theta)
            dt_prob.append(prob)
            dt_pred.append(pred)
        return dt_prob,dt_pred
    
    def evaluate(self,Xstest,Ystest):
        accuracies=[]
        dt_prob,dt_pred=self.TotalPredict(Xstest)
        pred_len=len(dt_pred[0].flatten())
       
        for i,value in enumerate(self.values):
            Y=Ystest.copy()
            sample_Y = self.convert_Y(Y,value)
            error_case = np.sum(np.absolute(dt_pred[i]-sample_Y))
            accuracy = (pred_len-error_case)/pred_len
            accuracies.append(accuracy)        
        return accuracies


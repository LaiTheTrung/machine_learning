
import numpy as np

class NeuralNetwork():
    def __init__(self,Xs,Ys,lr,rl=0,num_layers=4,num_unit_per_layer=5):
        self.Xs = self.transform(Xs,eval=True)
        self.n,self.m=self.Xs.shape

        self.Ys=self.convert_Y(Ys)
        self.lr=lr
        self.rl=rl
        self.num_layers=num_layers
        self.num_unit_per_layer=num_unit_per_layer
        self.K=len(np.unique(Ys))
        
        
        self.unit_in_layer = [self.n]+[self.num_unit_per_layer+1]*(self.num_layers-2)+[self.K] # output là list của số unit trên mỗi lớp (bao gồm bias unit)
        # tính số lượng unit/layer = [Xs]+ [số unit trên mỗi hidden layer]*(số hidden layer)+ [số unit của ouput layer]
        self.list_theta = []
       
        for l in range(self.num_layers-1): # randome inital theta
            sample_theta=np.random.uniform(low=-0.3, high = 0.5, size=(self.unit_in_layer[l+1],self.unit_in_layer[l])) # numer of theta between layer l and l+1 ((sl+1)xsl)
            self.list_theta.append(sample_theta)
        
        self.list_a = self.ForwardPropagation(self.Xs,self.list_theta)
            

    def convert_Y(self,Ys): # tạo ra 1 list các vector y Ví dụ:[0,1,0,0],[1,0,0,0]... 
        values = np.unique(Ys,axis=1)
        k = len(np.unique(Ys))
        new_Y = []
        n,m=Ys.shape
         #ví dụ với K=4 : [0,0,0,0]
        for i in values[0]:
            Y=Ys.copy()
            if i!=0:
                Y[Y!=i]=0
                Y[Y!=0]=1
            else: 
                Y+=1
                Y[Y!=1]=0
            new_Y.append(Y)
        new_Y = np.array(new_Y).reshape(k,m)
        return new_Y


    def transform(self,Xs,eval=True): 
        if eval ==True:
            Xs=self.normalize(Xs,eval) # hàm normalize return mean_value, rangeMtx, scaled_Xs
        n,m = Xs.shape
        Xs = np.concatenate((np.ones([1,m]),Xs))
        return Xs


    def normalize(self,Xs,nor=True):
        n,m = Xs.shape
        if nor == True:
            self.mean_value = np.mean(Xs,axis=1).reshape(n,1)
            max_value = np.amax(Xs,axis=1).reshape(n,1)
            min_value =np.amin(Xs,axis=1).reshape(n,1)
            self.rangeMtx = max_value-min_value
            scaled_Xs=(Xs-self.mean_value)/self.rangeMtx

        else: # Trong trường hợp predict thì chạy hàm else này để trả normalize các giá trị X_test
            scaled_Xs=(Xs-self.mean_value)/self.rangeMtx
        
        return   scaled_Xs
       
       

    def hypothesis(self,a,cur_theta):
        z= np.matmul(cur_theta,a)  #(sl+1,sl)x(sl,m) = sl+1xm
        hx= self.sigmoid(z)
        return hx #sl+1xm
    
    def sigmoid(self,z):
        g = 1/(1+np.e**(-z))
        return g

    def ForwardPropagation(self,Xs,list_theta):
        list_a= []
        a1 = Xs
        list_a.append(a1)

        for l in range(self.num_layers-1):
            cur_theta = list_theta[l]
            al=self.hypothesis(list_a[-1],cur_theta)
            list_a.append(al)
        return list_a
        
    def regularization(self,theta): #theta l (sl+1xsl)
        return (self.rl/2)*np.mean(theta[:,1:]**2)  

    def CostFunc(self):
        hx_ouput = self.list_a[-1] # k,m
        Cost = - np.mean( self.Ys*np.log(hx_ouput)+(1-self.Ys)*np.log(1-hx_ouput))
        regularization_cost=0
        for cur_theta in self.list_theta:
            regularization_cost += self.regularization(cur_theta)
        TotalLoss = Cost + regularization_cost
        return TotalLoss

    def sigmoid_derivative(self,a):
        return a*(1-a)
    
   
    def BackPropagation(self):
        delta=[]
        self.DELTA=[]

        self.list_a = self.ForwardPropagation(self.Xs,self.list_theta)
        delta_L= self.list_a[-1]-self.Ys
        delta.append(delta_L)
        
        for l in range(self.num_layers-2,0,-1):
     
            delta_l =  np.matmul(self.list_theta[l].T,delta[-1])*self.sigmoid_derivative(self.list_a[l])           
            delta.append(delta_l)                 # delta_l (sl,1)
        delta.reverse()                           # Đảo lại thứ tự của list delta, chạy từ delta2, delta3, . . deltaL
        for indx,delta in enumerate(delta):
            DELTA_l =  self.list_a[indx+1]*delta   # (sl,m).*(sl,1) = (sl,m)
            self.DELTA.append(DELTA_l)   
               
             
    def ReAssign(self):
        for indx,theta in enumerate(self.list_theta): #theta (sl+1)xsl có tổng cộng sl+1 theta0
            n,m=theta.shape             # n= sl+1, m=sl

            regularization_theta0=np.zeros([n,1]).reshape(n,1)
            regularization_thetaj=(self.rl/self.m)*theta[:,1:]      
            regularization_reduce= np.concatenate([regularization_theta0,regularization_thetaj],axis=1)

            reduce_temp = self.lr*np.mean(self.DELTA[indx],axis=1).reshape(n,1) + self.rl*regularization_reduce
            theta -= reduce_temp
            self.list_theta[indx] = theta 


    def predict(self,Xstest,cur_theta):
        Xstest=self.transform(Xstest,eval=False)
        a= self.ForwardPropagation(Xstest,cur_theta)
        output=a[-1]
        prob=output
        threshold=0.5
        pred= prob/threshold
        pred=pred.astype(int)
        return prob, pred # exmple return với K=4: [0.6,0.7,0.3,0.1], [1,1,0,0]
    

    def TotalPredict(self,Xstest):
    
        cur_theta = self.list_theta.copy()
        dt_prob,dt_pred=self.predict(Xstest,cur_theta)

        return dt_prob,dt_pred
    
    def evaluate(self,Xstest,Ystest):
        dt_prob, dt_pred=self.TotalPredict(Xstest)
        pred_len=len(dt_pred[0].flatten()) #=m
        Y=Ystest.copy()  # 1,m
        Y = self.convert_Y(Y) # k,m
        error_case = np.sum(np.absolute(dt_pred-Y),axis=1) 
        accuracy = (pred_len-error_case)/pred_len
                
        return accuracy
        




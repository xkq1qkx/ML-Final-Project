import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from tqdm import tqdm
# from sympy.polys.matrices import DM
# from sympy.polys.domains import ZZ, QQ
import argparse  
import json
import time
def main():
    pass

def z_generator(n):
    return np.random.uniform(0, 1, n)

def B_generator(n,m):
    return np.random.normal(0, 1, (n, m))

def dot_product(vec1, vec2):
    return sum(x1 * x2 for x1, x2 in zip(vec1, vec2))

def check_orthogonality(vec1, vec2):
    if dot_product(vec1, vec2) == 0:
        return True
    else:
        return False

def gramschmidt(V):
    U = []
    for v in V:
        temp = v
        for u in U:
            if np.any(np.isinf(u)):
                print('lll')
            if dot_product(u, u) == 0:
                print('lll')
            #temp = temp - projection(u, v)
            temp = temp - (np.dot(u, v) / np.dot(u, u)) * u
        if np.any(temp):
            U.append(temp)
            
    return U
def coef(i, j): 
    return (dot_product(i, j) / dot_product(j, j)) if dot_product(j,j) != 0 else 0
def RED(basis):
    n = len(basis)
    k = 1
    o_basis = gramschmidt(basis)

    while k < n:
        for j in range(k - 1, -1, -1):
            mu1 = coef(basis[k],o_basis[j])
            if (abs(mu1) > 1/2):
                basis[k] -= (round(mu1) * basis[j])
                o_basis = gramschmidt(basis)

        mu2 = coef(basis[k], o_basis[k-1])
        if (dot_product(o_basis[k], o_basis[k]) >= ( 0.7 - mu2**2 ) * dot_product(o_basis[k - 1], o_basis[k - 1]) ):
            k += 1
        else:
            t=basis[k].copy()
            basis[k] = basis[k-1]
            basis[k - 1]=t
            o_basis = gramschmidt(basis)
            k = max(k - 1, 1)
    return basis

def ORTH(B):
    A = B@B.T
    L = np.linalg.cholesky(A)

    return L

def CLP0(B,r):
    n=r.shape[0]
    c=1e10
    i=n
    d=(n-1)*np.ones((n),dtype=np.int32)
    lmda=np.zeros((n+1),dtype=np.float32)
    F=np.zeros((n,n),dtype=np.float32)
    F[n-1,:]=r.copy()
    u=np.zeros((n),dtype=np.int32)
    u_ans=np.zeros((n),dtype=np.int32)
    p=np.zeros((n),dtype=np.float32)
    det=np.zeros((n),dtype=np.float32)
    while True:
        while(lmda[i]<c):
            if i==0:
                u_ans=u.copy()
                c=lmda[0]
            elif i==n:
                i=i-1
                p[i]=F[i,i]/B[i,i]
                u[i]=round(p[i])
                y=(p[i]-u[i])*B[i,i]
                det[i]=np.sign(y)
                lmda[i]=lmda[i+1]+y*y
            else:
                i=i-1
                for j in range(i,d[i]):
                    F[j-1,i]=F[j,i]-u[j-1]*B[j,i]
                p[i]=F[i,i]/B[i,i]
                u[i]=round(p[i])
                y=(p[i]-u[i])*B[i,i]
                det[i]=np.sign(y)
                lmda[i]=lmda[i+1]+y*y
        m=i
        while(lmda[i]>=c):
            if i==n-1:
                return u_ans
            else:
                i=i+1
                u[i]+=det[i]
                det[i]=-det[i]-np.sign(det[i])
                y=(p[i]-u[i])*B[i,i]
                lmda[i]=lmda[i+1]+y*y
        for j in range(m,i):
            d[j]=i
        for j in range(1,m):
            if d[j]<i:
                d[j]=i
            else:
                break
   
def CLP(B,r):
    n=r.shape[0]
    c=1e12
    i=n+1
    d=n*np.ones((n+1),dtype=np.int32)
    lmda=np.zeros((n+2),dtype=np.float32)
    F=np.zeros((n+1,n+1),dtype=np.float32)
    F[n,1:]=r.copy()
    u=np.zeros((n+1),dtype=np.int32)
    u_ans=np.zeros((n+1),dtype=np.int32)
    det=np.zeros((n+1),dtype=np.float32)
    p=np.zeros((n+1),dtype=np.float32)
    start_time=time.time()
    while True:
        cur_time=time.time()
        if cur_time-start_time>1.0:
            print('time out')
            return CLP_estimate(B,r)
        while(lmda[i]<c):
            if i!=1:
                i=i-1
                for j in range(d[i],i,-1):
                    F[j-1,i]=F[j,i]-u[j]*B[j-1,i-1]
                p[i]=F[i,i]/B[i-1,i-1]
                u[i]=round(p[i])
                y=(p[i]-u[i])*B[i-1,i-1]
                det[i]=np.sign(y)
                lmda[i]=lmda[i+1]+y**2
            else:
                u_ans=u.copy()
                c=lmda[1].copy()
        m=i
        while(lmda[i]>=c):
            if i==n:
                return u_ans[1:]
            else:
                i=i+1
                u[i]+=det[i]
                det[i]=-det[i]-np.sign(det[i])
                y=(p[i]-u[i])*B[i-1,i-1]
                lmda[i]=lmda[i+1]+y**2
        for j in range(m,i):
            d[j]=i
        for j in range(1,m):
            if d[j]<i:
                d[j]=i
            else:
                break
                
def CLP_estimate(B,r):
    B=RED(B)
    B=B.T
    n=r.shape[0]
    c=np.linalg.solve(B,r)
    p=np.zeros_like(c)
    for i in range(n-1,-1,-1):
        p[i]=round(c[i])
        c-=p[i]*B[:,i]
    return p

def dp_clp(R,y,radius,depth,solution,best_dist,closest_point):
    if depth==-1:
        dist=np.linalg.norm(y-R@solution)**2
        if dist<best_dist:
            best_dist=dist
            closest_point=solution.copy()
        return best_dist,closest_point
    for z in range(int(np.floor(y[depth]/R[depth,depth]-radius)),int(np.ceil(y[depth]/R[depth,depth]+radius))+1):
        solution[depth]=z
        if depth>0:
            best_dist,closest_point=dp_clp(R,y,radius,depth-1,solution,best_dist,closest_point)
        else:
            dist=np.linalg.norm(y-R@solution)**2
            if dist<best_dist:
                best_dist=dist
                closest_point=solution.copy()
    return best_dist,closest_point
                
def CLP_exact(B,r,radius):
    Q,R=np.linalg.qr(B.T)
    y=Q.T@r
    n=r.shape[0]
    solution=np.zeros_like(r)
    best_dist=float('inf')
    closest_point=None
    best_dist,closest_point=dp_clp(R,y,radius,n-1,solution,best_dist,closest_point)
    return closest_point
    
def test_CLP():
    radius=5
    B=np.array([[1,0.0],[2.0,1.0]])
    r=z_generator(2)
    print(r)
    print(r@B)
    ans1=CLP(B,r@B)
    ans2=CLP_exact(B,r@B,radius)
    print(ans1,ans1@B,np.linalg.norm((ans1-r)@B))
    print(ans2,ans2@B,np.linalg.norm((ans2-r)@B))
     
def plot_NBr(B,n):
    float_list=np.arange(0, 5.001, 0.001)
    area=[-2,1,0,1,2]
    combination=list(itertools.product(area,repeat=n))
    norm_list=[]
    for u in combination:
        ub=np.dot(u,B)
        norm=dot_product(ub,ub)
        norm_list.append(norm)
    norm_list.sort()
    counts = [sum(1 for x in norm_list if x < r) for r in float_list]

# 绘制图形
    plt.plot(float_list, counts)
#     xticks = np.arange(0, len(float_list), 1000)

# # 生成要显示的刻度标签
#     xtick_labels = float_list[xticks]

# # 设置横坐标的刻度和标签
#     plt.xticks(xticks, xtick_labels)
    plt.title('Number of values less than r')
    plt.xlabel('r')
    plt.ylabel('Count')

# 显示图形
    plt.show()
                
def vis_voronoi():
    B=np.array([[1.0,0.0],[0.1,0.1]])
    area=[-3,-2,1,0,1,2,3]
    combination=list(itertools.product(area,repeat=2))
    x_vals = np.linspace(-4, 4, 1000)
    y_vals = np.linspace(-0.4, 0.4, 100)
    x, y = np.meshgrid(x_vals, y_vals)
    logical_result=np.zeros((1000,100))
    for xi in range(1000):
        for yi in range(100):
            z=np.array([x[yi,xi],y[yi,xi]])
            u=CLP(z,B,2)
            if np.linalg.norm(u)<0.0001:
                logical_result[xi,yi]=255
            u_r=np.array([u[1],u[0]])
            if np.linalg.norm(np.dot(u_r,B)-z)<0.1:
                logical_result[xi,yi]=125
    plt.imshow(logical_result, cmap='gray', interpolation='nearest')

    # 设置图表标题和标签
    plt.title("Array Visualization")
    plt.xlabel("Column")
    plt.ylabel("Row")

    # 显示图表
    plt.show()
    
def test_LLL():
    B=B_generator(2,2)
    print(B)
    print(RED(B))
    B=B*1e10
    y = DM(B,ZZ)
    t=y.lll_transform()
    r=np.array([t[0].rep[0],t[0].rep[1]])
    r=r/1e10
    print(r)

def origin_method(args):
    B_process=[]
    n=args['n']
    iter_num=args['iter_num']
    reduction_interval=args['reduction_interval']
    miu0=args['miu0']
    ratio=args['ratio']
    B=ORTH(RED(B_generator(n,n)))
    #B=np.array([[1.0,0],[0,1.0]])#二维最优B，调试用
    B = np.array([[2.0,1.0,1.0],[1.0,2.0,1.0],[1.0,1.0,2.0]])#二维最优B，调试用
    V=1.0
    for i in range(n):
        V*=B[i,i]
    B=math.pow(V,-1.0/n)*B
   
    for t in tqdm(range(iter_num),desc='iter',total=iter_num):
        if (t*args['batch_size'])%100000==0:
            B_process.append(B)
        miu=miu0*math.pow(ratio,-t/(iter_num-1))
        z=z_generator(n)
        y=z-CLP(B,z@B)
        e=y@B
        for i in range(n):
            for j in range(i):
                B[i,j]=B[i,j]-miu*y[i]*e[j]
            # if B[i,i]<=0:
            #     return print('error')
            # 原文要求在每次更新后检查B对角元是否为正，否则直接失败终止，但实际上在ORTH操作后，B对角元一定为正，
            # 加上这个检查后大多初始化都将失败
            B[i,i]=B[i,i]-miu*(y[i]*e[i]+np.dot(e,e)/(B[i,i]*n))
        if t%reduction_interval==(reduction_interval-1):
            B=ORTH(RED(B))
            V=1.0
            for i in range(n):
                V*=B[i,i]
            B=math.pow(V,-1.0/n)*B
        if np.any(np.isinf(B)):
            print('inf')
            return False
    return B,B_process
        
class SGD:
    def __init__(self,args):
        pass
    def update(self,B,gradient):
        B-=0.001*gradient
        return B

class Adam:
    def __init__(self,args):
        self.iter_num=0
        self.lr=args['lr']
        self.beta1=args['beta1']
        self.beta2=args['beta2']
        self.m=np.zeros((args['n'],args['n']),dtype=np.float32)
        self.v=np.zeros((args['n'],args['n']),dtype=np.float32)
    def update(self,B,gradient):
        self.iter_num+=1
        self.m=self.beta1*self.m+(1-self.beta1)*gradient
        self.v=self.beta2*self.v+(1-self.beta2)*np.square(gradient)
        m_hat=self.m/(1-math.pow(self.beta1,self.iter_num))
        v_hat=self.v/(1-math.pow(self.beta2,self.iter_num))
        B-=self.lr*m_hat/(np.sqrt(v_hat)+1e-8)
        return B

class AdamW:
    def __init__(self,args):
        self.iter_num=0
        self.lr=args['lr']
        self.beta1=args['beta1']
        self.beta2=args['beta2']
        self.weight_decay=args['weight_decay']
        self.m=np.zeros((args['n'],args['n']),dtype=np.float32)
        self.v=np.zeros((args['n'],args['n']),dtype=np.float32)
        self.pre_B=np.zeros((args['n'],args['n']),dtype=np.float32)
    def update(self,B,gradient):
        B_copy=B.copy()
        self.iter_num+=1
        self.m=self.beta1*self.m+(1-self.beta1)*gradient
        self.v=self.beta2*self.v+(1-self.beta2)*np.square(gradient)
        m_hat=self.m/(1-math.pow(self.beta1,self.iter_num))
        v_hat=self.v/(1-math.pow(self.beta2,self.iter_num))
        B-=self.lr*m_hat/(np.sqrt(v_hat)+1e-8)
        B-=self.lr*self.weight_decay*self.pre_B
        self.pre_B=B_copy
        return B


    

def improved_method(args):
    B_process=[]
    n=args['n']
    iter_num=args['iter_num']
    reduction_interval=args['reduction_interval']
    miu0=args['miu0']
    ratio=args['ratio']
    
    if args['use_reduction']:
        B=ORTH(RED(B_generator(n,n)))
    else:
        B=ORTH(B_generator(n,n))
    '''
    B=np.array([
        [1.0,0,0,0,0,0,0,0],
        [1.0,1.0,0,0,0,0,0,0],
        [1.0,1.0,1.0,0,0,0,0,0],
        [1.0,1.0,1.0,1.0,0,0,0,0],
        [1.0,1.0,1.0,1.0,1.0,0,0,0],
        [1.0,1.0,1.0,1.0,1.0,1.0,0,0],
        [1.0,1.0,1.0,1.0,1.0,1.0,1.0,0],
        [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    ])
    '''
    #B = np.array([[2.0,1.0,1.0],[1.0,2.0,1.0],[1.0,1.0,2.0]])#二维最优B，调试用
    B = np.array([[1.0,0],[0,1.0]])
    V=1.0
    for i in range(n):
        V*=B[i,i]
    B=math.pow(V,-1.0/n)*B
    gradient=np.zeros((n,n),dtype=np.float32)
    if args['optimization_method']=='SGD':
        optimizer = SGD('optimization_args')
    elif args['optimization_method']=='Adam':
        optimizer=Adam(args['optimization_args'])
    elif args['optimization_method']=='AdamW':
        optimizer=AdamW(args['optimization_args'])
    for t in tqdm(range(iter_num),desc='iter',total=iter_num):
        if (t*args['batch_size'])%10000==0:
            B_process.append(B)
        miu=miu0*math.pow(ratio,-t/(iter_num-1))
        if args['use_batch']:#计算batch_size个z，梯度求和/平均，然后更新B
            gradient=np.zeros((n,n),dtype=np.float32)
            for t in range(args['batch_size']):
                z=z_generator(n)
                y=z-CLP(B,z@B)
                e=y@B
                for i in range(n):
                    for j in range(i):
                        gradient[i,j]+=miu*y[i]*e[j]
                    gradient[i,i]+=miu*(y[i]*e[i]+np.dot(e,e)/(B[i,i]*n))
            if args['batch_mode']=='mean':
                gradient/=args['batch_size']
        else:
            z=z_generator(n)
            y=z-CLP(B,z@B)
            e=y@B
            for i in range(n):
                for j in range(i):
                    gradient[i,j]=miu*y[i]*e[j]
                gradient[i,i]=miu*(y[i]*e[i]+np.dot(e,e)/(B[i,i]*n))
        B=optimizer.update(B,gradient)
        if t%reduction_interval==(reduction_interval-1):
            if args['use_reduction']:
                B=ORTH(RED(B))
            else:
                B=ORTH(B)
            V=1.0
            for i in range(n):
                V*=B[i,i]
            B=math.pow(V,-1.0/n)*B
        if np.any(np.isinf(B)):
            print('inf')
            return False
    return B,B_process
if __name__ == '__main__':
    origin_method_args={
        'n':3, # 生成矩阵的维度
        'iter_num':1000000, # 迭代次数
        'reduction_interval':100, # 降维间隔，对B简化，并重新计算V
        'miu0':0.0005, # 初始步长
        'ratio':200, 
        'use_batch':True,
        'batch_size':100,
        'batch_mode':'mean',
        'optimization_method':'SGD',
        'use_reduction':True# 步长衰减比率,指第一次迭代和最后一次迭代的步长之间的比例，以指数衰减
        # 以上5个参数是原论文方法里面的参数，可以借鉴原论文中的参数table来设置，也可视情况自行调整
    }
    n=2
    improve_method_args={
        'n':n, # 生成矩阵的维度
        'iter_num':100000, # 迭代次数
        'reduction_interval':100, # 降维间隔，对B简化，并重新计算V
        'miu0':0.0005, # 初始步长
        'ratio':100, # 步长衰减比率,指第一次迭代和最后一次迭代的步长之间的比例，以指数衰减
        # 以上5个参数是原论文方法里面的参数，可以借鉴原论文中的参数table来设置，也可视情况自行调整
        'use_reduction':True, # 是否使用RED函数，RED函数可以简化B矩阵，但是对优化的效果影响未知
        'use_batch':True, # 是否使用批量计算，原论文每次生成一个z，就更新一次B，这里可以改为批量生成z，然后更新一次B
        'batch_size':500, # 批量计算的大小，即每次生成z数量，只有use_batch为True时有效
        'batch_mode':'mean', # 批量计算的方式，mean表示梯度求平均，sum表示梯度求和，只有use_batch为True时有效,默认为sum
        'optimization_method':'Adam', # 优化方法，原论文中只有SGD实现，我添加了AdamW,Adam
        'optimization_args':{ # 优化方法的参数
            'n': n, # 生成矩阵的维度
            'lr': 0.0001, # 优化器学习率,由于在计算梯度时已经使用了原来的算法的学习率，这里的学习率是单独的
            'beta1': 0.9, # Adam(W)的参数
            'beta2': 0.99, # Adam(W)的参数
            'weight_decay': 0.01, # AdamW的参数
            
        }
    }
    #origin_method(origin_method_args)
    #best_B,B_process=improved_method(improve_method_args)
    best_B,B_process=improved_method(improve_method_args)
    with open('B_process_n10.json', 'w') as f:
        json.dump([b.tolist() for b in B_process], f)
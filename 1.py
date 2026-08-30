import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
#划分网格6*6
nx=600
ny=600

#网格长度
lx=500
ly=500
## Current solver assumes dx == dy
#网格间距
dx=lx/(nx-1)
dy=ly/(ny-1)

#离散点参数
x=np.linspace(0,lx,nx)
y=np.linspace(0,ly,ny)

#各节点温度
T=np.zeros((ny,nx))

# 内部节点编号（左下到右上，0开始）
def idx(i,j,nx):
    return (nx-2)*(j-1)+(i-1)# j表示行，i表示列
# 填充内部网格
N=(nx-2)*(ny-2)
A=lil_matrix((N,N))#创建空内部矩阵
b=np.zeros(N)

for row in range(1,ny-1):
    for col in range(1,nx-1):
        p=idx(col,row,nx)
        A[p,p]=4
        if col>=2:
            A[p,p-1]=-1
        if col<nx-2:
            A[p,p+1]=-1
        if row > 1:
            A[p, p - (nx - 2)] = -1
        if row < ny - 2:
            A[p, p + (nx - 2)] = -1
        b[p] = dx ** 2
##print(A.toarray())
A=A.tocsr()
#内部离散点温度
T_inner=spsolve(A,b)
#print(T_inner)
#回填矩阵
for row in range(1, ny - 1):
    for col in range(1, nx - 1):
        p = idx(col, row, nx)
        T[row, col] = T_inner[p]
#print(T)
X,Y=np.meshgrid(x,y)#确定（x,y)处的温度是多少

plt.contourf(X, Y, T)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('result/temperature_field.png', dpi=300, bbox_inches='tight')
plt.show()

T_max=np.amax(T)
T_max_pos=np.argwhere(np.isclose(T, T_max))#找最高温所在的行和列,inclose消除精度误差
rows=T_max_pos[:,0]#所有行的一个个数，也就是行数
cols=T_max_pos[:,1]#所有行的二个数，也就是列数
print('最高温度：',T_max)
for i in range(len(rows)):
        print(f'({x[rows[i]]:.2f},{y[cols[i]]:.2f})')
#np.set_printoptions(precision=16)
#print(T)
#残差
r=A@T_inner-b
print('残差：',r)
print('最大残差：',np.max(abs(r)))
print('网格：', nx, 'x', ny)
print('dx =', dx)
print('未知节点数 N =', N)
print('最高温度 =', T_max)

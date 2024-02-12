#Algoritmo de Gradiente Descendente para Regressao Linear

print("Aluna: Thays da Cruz Franco")

#Imports

import numpy as np
import matplotlib.pyplot as plt

#Conjunto de dados de treinamento

def f_true ( x ) :
    return 2 + 0.8 * x

xs = np . linspace ( -3 , 3 , 100)
ys = np . array ( [ f_true ( x ) + np . random . randn () *0.5 for x in xs ] )



#Funcoes do metodo 

#hipotese

def h(x, theta):
    
    theta_0, theta_1 = theta
    return theta_0 + (theta_1 * x)

#custo

def J ( theta , xs , ys ):
    
    n = len(xs)
    erro = 0 
    
    for i in range(n):
      
        erro += (h(xs[i], theta) -  ys[i] )**2
    
    return erro/(2.0*n)

#gradiente

def gradient (j , theta , xs , ys, alpha ) :
    
    s_0, s_1, epoca = 0,0,0
    theta_0, theta_1 = theta
    n = len(xs)
    a = alpha/n
    custo = []
    epc = []
    
    while epoca < j :   
        
        for i in range(n):
            
            s_0 += (h(xs[i], theta) - ys[i])
            s_1 += (h(xs[i], theta) - ys[i])*xs[i]
        
        temp_0 = theta_0 - a * s_0
        temp_1 = theta_1 - a * s_1
        theta = temp_0, temp_1
        
        custo.append(float(J(theta, xs, ys)))
        epc.append(epoca)
            
        epoca += 1
        
    return [theta, custo, epc]

  
''' plota no mesmo grafico : - o modelo / hipotese ( reta )
- a reta original ( true function )
- e os dados com ruido (xs , ys)
'''
def print_modelo ( xs , ys, j, theta, alpha):
    
    x = []
    hip = []
    theta = gradient(j, theta, xs, ys, alpha)[0]
    
    for i in range(len(xs)):
        
        x.append(f_true(xs[i]))
        hip.append( h(xs[i], theta))
    
    plt.plot(xs, x, label = 'true function',  color = 'orange')   
    plt.plot(xs,hip, label = 'hypothesis function' ,  color = 'green')
    plt.title('function behavior');
    plt.scatter(xs, ys, s = 5,  color = 'navy', label = 'residual')
    plt.legend()
    plt.savefig("regressaolinear.png")
    plt.close()
    

print_modelo(xs,ys,600,(4,1), 0.01)
    

#alfa 0.1      

v = gradient(5000, (4,1), xs, ys, 0.1)

f = v[1]
e = v[2]
       
plt.plot(e,f)
plt.title('cost with alpha 0.1');
plt.ylabel("custo")
plt.xlabel("epoca")
plt.savefig("alpha1.png")

plt.close()
   
#alfa 0.9  

l = gradient(5000, (4,1), xs, ys, 0.9)

u = l[1]
o = l[2]
       
plt.plot(o,u)
plt.title('cost with alpha 0.9');
plt.ylabel("custo")
plt.xlabel("epoca")
plt.savefig("alpha09.png")
plt.close()
    
#alfa 0.0001

a1 = gradient(5000, (4,1), xs, ys, 0.0001)

u1 = a1[1]
o1 = a1[2]
       
plt.plot(o1,u1)
plt.title('cost with alpha 0.0001');
plt.ylabel("custo")
plt.xlabel("epoca")
plt.savefig("alpha0001.png")
plt.close()
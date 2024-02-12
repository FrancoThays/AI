#################################################
#    Universidade Federal de Santa Catarina     #
#    Willian do Nascimento Finato Benoski       #
#                Thays Franco                   #
#              Machine Learning                 #
#################################################

#%% Bibliotecas
%reset -f
%clear

import random
import numpy as np
import math
import copy
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd

#%% Inicializando as redes: definiÃ§Ã£o de parÃ¢metros



def camada(Tipo, nro_Neuronios, ordem, act_Func = 'sigmoid'):
    if act_Func != 'sigmoid' and act_Func != 'relu' and act_Func != 'tanh':
        print('Erro: funÃ§Ã£o de ativaÃ§Ã£o nÃ£o reconhecida')
        return
    if Tipo == 'Entrada':
        return ['Entrada', nro_Neuronios,ordem,'none']
    if Tipo == 'Saida':
        return ['Saida', nro_Neuronios,ordem, act_Func]
    if Tipo == 'Camada':
        return ['Camada', nro_Neuronios,ordem, act_Func]


#%% FunÃ§Ã£o para ordenar as camadas

def ordenar(camadas):
    ordem = []
    aux1 = []
    
    for i in range(len(camadas)):
        aux1.append(camadas[i][2])
        
    for i in range(len(aux1)):
        ordem.append(camadas[aux1.index(i+1)])

    return ordem
                
#%%Funções de randomização

#Xavier
def Xavier(ant_Con, pos_Con):
    return random.uniform(-1,1)*math.sqrt(6./(ant_Con+pos_Con))

#%% FunÃ§Ã£o para iniciar as redes: pesos

def init_rede (camadas):
    cont_Ent = 0
    cont_Saida = 0
    cont_Camada = 0
    
    for i in range(len(x)):
        if x[i][0] == 'Entrada':
            cont_Ent += 1
        if x[i][0] == 'Saida':
            cont_Saida += 1
        if x[i][0] == 'Camada':
            cont_Camada += 1
    
    if cont_Ent > 1:
        return print('Erro: Mais de uma camada de entrada')
    if cont_Saida > 1:
        return print('Erro: Mais de uma camada de sai­da')
          
    camadas = ordenar(camadas)
    
    rede_neural = []
    pesos = []
    
    for i in range(len(x)-1):
        for j  in range(x[i+1][1]):
            pesos.append(Xavier(x[i-1][1]+1,x[i+1][1]))
            for k in range(x[i][1]):
                pesos.append(Xavier(x[i-1][1]+1,x[i+1][1]))
            rede_neural.append(pesos)
            pesos = []
                
    return rede_neural

#%% Forward Propagation


# Ativacao do neuronio
def ativacao_camada(pesos_neuronio, entradas):
    actv = 0
    for i in range(len(pesos_neuronio)):
        if i == 0:
            actv = pesos_neuronio[i]
        else:
            actv = actv + pesos_neuronio[i]*entradas[i-1]
    return actv


#Funcoes de transferencia
def sigmoid_func(neuronio_ativado):
    return 1/(1+math.exp(-neuronio_ativado))

def tanh_func(neuronio_ativado):
    return (math.exp(2*neuronio_ativado)-1)/(math.exp(2*neuronio_ativado)+1)

def relu_func(neuronio_ativado):
    if neuronio_ativado < 0:
        return 0
    else:
        return neuronio_ativado

#Transferencia do neuronio

def foward_propagation(rede_neural,pesos,entradas):
    sinal_prop = entradas
    sinal_nprop = []
    sinal_result = []
    soma_neur = 0
    for i in range(len(x)-1):
        sinal_aux = []
        aux = []
        
        for j in range(rede_neural[i+1][1]):
            actv = ativacao_camada(pesos[j+soma_neur], sinal_prop)
            aux.append(actv)
            if rede_neural[i+1][3] == 'relu':
                neur_actv = relu_func(actv)
                sinal_aux.append(neur_actv)
            elif rede_neural[i+1][3] == 'sigmoid':
                neur_actv = sigmoid_func(actv)
                sinal_aux.append(neur_actv)
            elif rede_neural[i+1][3] == 'tanh':
                neur_actv = tanh_func(actv)
                sinal_aux.append(neur_actv)
        soma_neur = soma_neur + rede_neural[i+1][1]
        sinal_prop = sinal_aux
        sinal_result.append(sinal_prop)
        sinal_nprop.append(aux)
    return sinal_result, sinal_nprop

#%% Retropropagação do erro

#Derivadas das funções de tranferencia

def sigmoid_dev(saida_retropropagada):
    return (1/(1+math.exp(-saida_retropropagada)))*(1-(1/(1+math.exp(-saida_retropropagada))))

def tanh_dev(saida_retropropagada):
    return (1-((math.exp(2*saida_retropropagada)-1)/(math.exp(2*saida_retropropagada)+1))**2)

def relu_dev(saida_retropropagada):
    if saida_retropropagada < 0:
        return 0
    else:
        return 1
    
#Backprop

def backprop(sinal_propagado, sinal_npropagado, saida, rede_neural, pesos, entrada):
    
#Separação do bias e pesos
    
    Matriz_pesos = [] 
    Matriz_bias = []
    MBaux = []
    MPaux = []
    
    for j in range(len(pesos)):
        MBaux = []
        MPaux = []
        for k in range(len(pesos[j])):
            
            if k == 0:
                MBaux.append(pesos[j][k])
            else:
                MPaux.append(pesos[j][k])
        
        Matriz_bias.append(MBaux)
        Matriz_pesos.append(MPaux)
       
    
#Calculo dos erros de Z
    
    erros = []
    for i in reversed(range(1,len(rede_neural))):
               
        if i == (len(rede_neural)-1):
            erro = np.array(sinal_propagado[i-1]) - np.array(saida)
            erros.append(erro.tolist())      

        else:
            soma = 0
            soma2 = 0
            
            for j in range(i+2):
                soma = soma + rede_neural[j][1]
                if j < i+1: 
                    soma2 = soma2 + rede_neural[j][1]
            
            matrix_aux = []
            for j in range(soma2-len(entrada),soma-len(entrada)):
                matrix_aux.append(Matriz_pesos[j])
            
            
            sNp = []
            sNp = sinal_npropagado[i-1]
            sDp = []
            
            for j in range(len(sNp)):
                if rede_neural[i][3] == 'sigmoid':
                    sDp.append(sigmoid_dev(sNp[j]))
                elif rede_neural[i][3] == 'relu':
                    sDp.append(relu_dev(sNp[j]))
                elif rede_neural[i][3] == 'tanh':
                    sDp.append(tanh_dev(sNp[j]))    
            
                           
            erro = np.multiply(np.dot(np.transpose(np.array(matrix_aux)),erro),np.array(sDp))
            erros.append(erro.tolist())
    
    erros.reverse()  
    
    
    #Calculo dos DELTAS
    delta = []
    for L in range(len(rede_neural)-1):
        for i in range(rede_neural[L+1][1]):
            delta_aux = []
            for j in range(rede_neural[L][1]+1):
                # print('Neuronio camada atual',j,'|n neuronio camada posterior' ,i, '|n camada',L+1)
                
                if j == 0:
                    delta_aux.append(erros[L][i])
                else:
                    if L == 0:
                        delta_aux.append(entrada[j-1]*erros[L][i])
                        #print(delta_aux)
                    else:
                        delta_aux.append(sinal_propagado[L-1][j-1]*erros[L][i])
                        #print(delta_aux)
                        
            delta.append(delta_aux)
    # print(delta)
    
    
    return erros, delta
 
def Calc_med(x,pesos,entrada,saida):
    Delta_sum = []
    Delta_aux1 = []
    Delta_aux2 = []
    sinal_prop1 = []
    for exemplos in range(len(entrada)):
        
        sinal_propagado, sinal_npropagado = foward_propagation(x, pesos, entrada[exemplos])  
        erro, delta = backprop(sinal_propagado, sinal_npropagado, saida[exemplos], x, pesos,entrada[exemplos])
        sinal_prop1.append(sinal_propagado[len(sinal_propagado)-1])
        #print(delta)
        Delta_sum = []
        for i in range(len(delta)):
            Delta_aux2 = []
            for j in range(len(delta[i])):
                if exemplos == 0:
                    Delta_sum = copy.deepcopy(delta)
                else:
                    Delta_aux2.append(Delta_aux1[i][j]+delta[i][j])
            Delta_sum.append(Delta_aux2)
        Delta_aux1 = copy.deepcopy(Delta_sum)

    Delta_med = []    
    for i in range(len(delta)):
        Delta_med_aux = []
        for j in range(len(delta[i])):
             Delta_med_aux.append(Delta_sum[i][j]/len(entrada)) 
        Delta_med.append(Delta_med_aux)
                
        
    # print(Delta_sum)
    # print(Delta_med)
    return Delta_sum, Delta_med, sinal_prop1

def att_Pesos(rede, entrada, taxa_aprendizado, Media, pesos):
    Peso_att = []
    for i in range(len(pesos)):
        Peso_aux = []
        for j in range(len(pesos[i])):
            Peso_aux.append(pesos[i][j]-taxa_aprendizado*Media[i][j])
        Peso_att.append(Peso_aux)
    return Peso_att


#%%VERIFICAÇÃO NUMERICA


def J(saida, Saida_prop):
    custo = []
    custo_aux = []
    m = len(saida)
    for i in range(len(saida)):
        
        for k in range(len(saida[i])):
            custo.append(-((saida[i][k]*math.log(Saida_prop[i][k]))+((1 -saida[i][k])*math.log(1-Saida_prop[i][k])))/3)
               
    # print(sum(custo))
    return sum(custo)

def grad_Approx(pesos, epsilon, rede_neural, entradas,saida):
    theta_plus = []
    theta_minus = []
    theta_plus = copy.deepcopy(pesos)
    theta_minus = copy.deepcopy(pesos)
    grad_approx = []
    
    
    for i in range(len(pesos)):
        grad_approx_aux = []
        for j in range(len(pesos[i])):
            theta_plus = []
            theta_minus = []
            theta_plus = copy.deepcopy(pesos)
            theta_minus = copy.deepcopy(pesos)
            theta_plus[i][j] = theta_plus[i][j] + epsilon
            theta_minus[i][j] = theta_minus[i][j] - epsilon
            
            
            # print(theta_plus)
            # print(theta_minus)
            sinal_ms = []
            sinal_ps = []

            
            for exemplo in range(len(entradas)):
                sinal_result_p, npro_p = foward_propagation(rede_neural,theta_plus,entradas[exemplo])
                sinal_ps.append(sinal_result_p[len(sinal_result_p)-1])
                
                sinal_result_m, npro_m = foward_propagation(rede_neural,theta_minus,entradas[exemplo])
                sinal_ms.append(sinal_result_m[len(sinal_result_m)-1])
            
            
            grad_approx_aux.append((J(saida,sinal_ps)-J(saida,sinal_ms))/(2*epsilon))
        grad_approx.append(grad_approx_aux)
        
    return theta_plus, theta_minus, grad_approx
            
    
def comp(Media, Gradiente):
    vet_m = []
    vet_g = []
    for i in range(len(Media)):
        for j in range(len(Media[i])):
            vet_m.append(Media[i][j])
            vet_g.append(Gradiente[i][j])
    
    Num = norm(np.array(vet_m) - np.array(vet_g))
    Den = norm(np.array(vet_m)) + norm(np.array(vet_g))
                                      
    Dife = (Num/Den)
    
    return  Dife
    
    
#%% Treinamento da rede neural

def acuracia(saida_prop, saida):
    Sr = []
    Sa = []
    acc = 0
    for i in range(len(saida)):
        for j in range(len(saida[i])):
            if saida_prop[i][j] < 0.5:
                Sa.append(0)
            else:
                Sa.append(1)  
            Sr.append(saida[i][j])
    
    # print('Saida real',Sr)
    # print('Saida aprendida',Sa)
    cont = 0
    for i in range(len(Sa)):
        if Sa[i] == Sr[i]:
            cont = cont + 1
    
    acc = (cont/len(Sr))*100
    
    return acc

def Treinamento(nEpocas, rede_neural, pesos, entrada, saida, taxa_aprendizado,entrada_val, saida_val):
    custo = []
    acur = []
    acur_val = []
    for Epocas in range(nEpocas):
        if Epocas == 0:
            Soma, Media, Saida_prop = Calc_med(rede_neural, pesos, entrada, saida)
            Pesos_att = att_Pesos(rede_neural, entrada, taxa_aprendizado, Media, pesos)
            Custo = J(saida, Saida_prop)
            # t1,t2,gradiente = grad_Approx(pesos, epsilon, x,entrada,saida)
            # Dife = comp(Media, gradiente)
            
            
        else:
            Soma, Media, Saida_prop = Calc_med(rede_neural, Pesos_att, entrada, saida)
            Pesos_att = att_Pesos(rede_neural, entrada, taxa_aprendizado, Media, Pesos_att)
            Custo = J(saida, Saida_prop)
            # t1,t2,gradiente = grad_Approx(pesos, epsilon, x,entrada,saida)
            # Dife = comp(Media, gradiente)
            
        custo.append(Custo)   
        acc = acuracia(Saida_prop,saida)
        acur.append(acc)
                
        if Epocas%100 == 0:
            print('Epoca = ', Epocas, '|n Taxa de aprendizado = ', taxa_aprendizado, '|n Acertos', acc,'%' )
            # print(Custo)
            # print(Saida_prop)
            
        #VALIDAÇÃO POR EPOCA  
        if Epocas == 0:
            Soma_val, Media_val, Saida_prop_val = Calc_med(rede_neural, pesos, entrada_val, saida_val)
            
            
        else:
            Soma_val, Media_val, Saida_prop_val = Calc_med(rede_neural, Pesos_att, entrada_val, saida_val)
            
             
        acc_val = acuracia(Saida_prop_val,saida_val)
        acur_val.append(acc_val)
        
    
    x1s = np.linspace(-1,1.5,50)
    x2s = np.linspace(-1,1.5,50)
    
    X, Y = np.meshgrid(x1s, x2s)
    z=np.zeros((len(x1s),len(x2s)))
    
    # xre_aux = []
    # xre = []
    
    # Z =[]
    
    for Ei in range(len(x1s)):
        for Ej in range(len(x2s)):
            # print([x1s[Ei],x2s[Ej]])    
            A, Z_aux  = foward_propagation(rede_neural,Pesos_att,[x1s[Ei],x2s[Ej]])
            # print(Z_aux[len(rede_neural)-2])
            z[Ei,Ej] = Z_aux[len(rede_neural)-2][0]
    # print(z)
    
    # z[i,j] = cachet["Z"+str(int(len(params)/2 + 1))] # saida do modelo antes de aplicar a função sigmoide 

    df=pd.read_csv('classification2.txt', header=None)

    plt.contourf(X,Y,z,0)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc=0)

    X=df.iloc[:,:-1].values
    y=df.iloc[:,-1].values
    pos , neg = (y==1).reshape(118,1) , (y==0).reshape(118,1) # 118
    plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+")
    plt.scatter(X[neg[:,0],0],X[neg[:,0],1],marker="o",s=10)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(["Accepted","Rejected"],loc=0)

    plt.show()
        
        
    return Saida_prop, Pesos_att, custo,acur, acur_val




#%% import de dados

with open ('classification2.txt', "r") as f:
    en = np.ones([118,2])
    sa = np.ones([118,1])
    i = 0
    for lines in f:
        currentline = lines.split(",")
        en[i,0] = currentline[0]
        en[i,1] = currentline[1]
        sa[i] = currentline[2]
        i = i + 1
        
    
    
entrada = np.ones([70,2])
saida = np.ones([70,1])
val_entrada = np.ones([28,2])
val_saida = np.ones([28,1])
tes_entrada = np.ones([20,2])
tes_saida = np.ones([20,1])

j=0
k=0
l=0
for i in range(len(en)):
    if (i>=0 and i<35)or(i>=59 and i<94):
        entrada[j] = en[i]
        saida[j] = sa[i]
        j=j+1
    elif (i>=35 and i<49)or(i>=94 and i<108):
        val_entrada[k] = en[i]
        val_saida[k] = sa[i]
        k=k+1
    elif (i>=49 and i<59)or(i>=108 and i<=117):
        tes_entrada[l] = en[i]
        tes_saida[l] = sa[i]
        l=l+1

entrada = entrada.tolist()
saida = saida.tolist()   
val_entrada = val_entrada.tolist()
val_saida = val_saida.tolist()        
tes_entrada = tes_entrada.tolist()
tes_saida = tes_saida.tolist()        
     

#%% TESTE

# entrada = [[0.051267,0.69956],[-0.092742,0.68494], [-0.21371,0.69225]]
# saida = [[1],[1],[1]]
x = []
x.append(camada('Entrada',2, 1))
x.append(camada('Camada',5,2,'relu'))
x.append(camada('Camada',5,3,'relu'))
x.append(camada('Camada',5,4,'relu'))
x.append(camada('Saida',1,5,'sigmoid'))

pesos = init_rede(x)
taxa_aprendizado = 0.1
epsilon = 0.001
Epocas = 3000

#%% TREINO


#Saida_Aprendida, Pesos_att, custo, vet_acur = Treinamento(Epocas, x, pesos, entrada, saida, taxa_aprendizado)

#%%TREINO COM VALIDAÇÃO

Saida_Aprendida, Pesos_att, custo, vet_acur, val_acur = Treinamento(Epocas, x, pesos, entrada, saida, taxa_aprendizado,val_entrada,val_saida)



#%%Validação

Soma, Media, Saida_result_val = Calc_med(x, Pesos_att, val_entrada, val_saida)
Acuracia_val = acuracia(Saida_result_val, val_saida)
print('Precisão da validação =', Acuracia_val)


#%% Teste

Soma, Media, Saida_result_tes = Calc_med(x, Pesos_att, tes_entrada, tes_saida)
Acuracia_tes = acuracia(Saida_result_tes, tes_saida)
print('Precisão do teste =',Acuracia_tes)


#%% Gráficos

Epocas_graf = np.linspace(0,Epocas,Epocas)

plt.figure(figsize=(6, 4))
plt.plot(Epocas_graf,custo)
plt.xlabel('Epocas')
plt.ylabel('Custo')
plt.grid(True)
plt.title('Custo x Epocas')


plt.figure(figsize=(6, 4))
plt.plot(Epocas_graf,vet_acur)
plt.xlabel('Epocas')
plt.ylabel('Acuracia')
plt.grid(True)
plt.title('Acuracia x Epoca')

plt.figure(figsize=(6, 4))
plt.plot(Epocas_graf,val_acur)
plt.xlabel('Epocas')
plt.ylabel('Acuracia')
plt.grid(True)
plt.title('Validação:Acuracia x Epoca')


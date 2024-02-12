'''
Algoritmo genético para exercício 1
Machine Learning
Thays da Cruz Franco 9-20-2022
'''

#1.a) Implemente um algoritmo genético simples

#IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import random
from bits import *

'''
#01 - Populacao inicial
n_pop = 5
pop_ini = np.random.uniform(0, np.pi,n_pop).tolist() #lista aleatória de cromossomos
g_0 = []

for i in pop_ini:
    g_0 += [get_bits(i)] #transformando valores reais em binários
'''

#02 - Cálculo da aptidão

def aptidao(x):
    y = get_float(x)  # transformando em reais para aplicar na função seno
    if y > np.pi or y < 0:
        y = 0
    else:
        y = y + (abs(np.sin(32 * y)))  # função objetivo como função de aptidão
    return y

##3.1 - Seleção método da roleta

def probabilidade(pop):
    total = sum(aptidao(x) for x in pop) #aptidao total
    p_i = [aptidao(x)/total for x in pop] #aptidao do cromossomo
    return p_i

def roleta(pop, p_i):
    return np.random.choice(pop, p = p_i) #pai escolhido

def media(pop):
    m = [aptidao(x) for x in pop]
    return np.mean(m)

#3.2. Cruzamento

def cruzamento(p1, p2, pc):
    c1,c2 = p1, p2
    if random.random() < pc:
        ponto = random.randint(1, len(p1)-2)
        c1 = p1[:ponto] + p2[ponto:]
        c2 = p2[:ponto] + p1[ponto:]
    return [c1,c2] #gera cruzamento

#3.3 - mutacao
def mutacao(cromossomo, pm):
    m = np.random.randint(0,len(cromossomo))
    c = list(cromossomo)
    if random.random() < pm:
        if c[m] == '0':
            c[m] = '1'
        else:
            c[m] = '0'
    c = ''.join(c)
    return c

#5. execucao do algoritmo
def algoritmo_genetico( n_pop, n_iter, pm, pc):
    pop_1 = []
    pop = []
    pop_ini = np.random.uniform(0, np.pi, n_pop).tolist()  # lista aleatória de cromossomos

    for i in pop_ini:
        pop += [get_bits(i)]  # transformando valores reais em binários

    for geracao in range(n_iter+1):
        p_i = probabilidade(pop) #calcula aptidao
        while len(pop_1) <  len(pop):
            p1, p2 = roleta(pop, p_i), roleta(pop,p_i)
            for j in cruzamento(p1, p2, pc):
                mutacao(j, pm)
                pop_1.append(j)
        pop.clear()
        pop = pop_1.copy() #substitui populacao
        pop_1.clear()
    return pop

#print(algoritmo_genetico(20, 40, 1, 1))

'''
#b - variando tamanho da população, taxa de mutação e taxa de cruzamento

b0 = algoritmo_genetico(20,40,1,1)
b1 = algoritmo_genetico(5,40,1,1)
b2 = algoritmo_genetico(20,40,0.2,1)
b3 = algoritmo_genetico(20,40, 1, 0.1)

print("valor da media de probabilidade original", np.mean(probabilidade(b0)))
print("valor alterando o tamanho da populacao", np.mean(probabilidade(b1)))
print("valor alterando a taxa de mutacao",np.mean(probabilidade(b2)))
print("valor alterando taxa de cruzamento",  np.mean(probabilidade(b3)))


#c)

def evolucao(n_pop, n_iter, pm, pc):
    pop_1 = []
    pop = []
    x = []
    y = []
    pop_ini = np.random.uniform(0, np.pi, n_pop).tolist()  # lista aleatória de cromossomos

    for i in pop_ini:
        pop += [get_bits(i)]  # transformando valores reais em binários

    for geracao in range(n_iter + 1):
        p_i = probabilidade(pop)  # calcula aptidao
        while len(pop_1) < len(pop):
            p1, p2 = roleta(pop, p_i), roleta(pop, p_i)
            for j in cruzamento(p1, p2, pc):
                mutacao(j, pm)
                pop_1.append(j)
        pop.clear()
        pop = pop_1.copy()  # substitui populacao
        pop_1.clear()
        x += [geracao]
        y += [media(pop)]

    return x,y

f = evolucao(100, 600, 0.5,0.5)
plt.plot(f[0],f[1])
plt.suptitle('evolucao de aptidao')
plt.xlabel('geracao')
plt.ylabel('aptidao')
plt.show()

'''






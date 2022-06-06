#Importe de las librerías 
import os
import copy
import time
import math
import random
import zipfile
import numpy as np
import pandas as pd
from time import time 
from timeit import timeit
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances_argmin_min

def grasp(listNodos, numeroClusters,k,n):
  inicio = time()
  coordenadas= np.zeros((len(listNodos),5))
  a2= []
  for i in range(len(listNodos)): 
    coordenadas[i,0]= listNodos[i,0]
    coordenadas[i,1]= listNodos[i,1]
    coordenadas[i,2]= listNodos[i,2]
    coordenadas[i,3]= listNodos[i,3]
    coordenadas[i,4]= i+1
    a2.append([listNodos[i,0],listNodos[i,1],listNodos[i,2],listNodos[i,3],i+1])

  kmeans = KMeans(n_clusters=numeroClusters)
  kmeans.fit(coordenadas)
  y_kmeans = kmeans.predict(coordenadas)
  plt.scatter(coordenadas[:,0], coordenadas[:,1], c=y_kmeans, s=50, cmap='viridis')
  centers = kmeans.cluster_centers_

  grupos = [[] for i in range(numeroClusters)]
  for i in range(len(coordenadas)):
    grupos[y_kmeans[i]].append(a2[i]) 

  dist=[[] for i in range(len(centers))]
  for i in range(len(centers)):
    for j in grupos[i]:
      dist[i].append(math.floor((math.sqrt((centers[i][0]-j[0])**2+(centers[i][1]-j[1])**2))))

  dist_aux=[[] for i in range(len(centers))]
  for i in range(len(centers)):
    for j in grupos[i]:
      dist_aux[i].append(math.floor((math.sqrt((centers[i][0]-j[0])**2+(centers[i][1]-j[1])**2))))

  asignaciones, objetivos, disTs, elegidoss= [],[],[],[]


  for w in range(n):
    a = a2.copy()
    aux= [[] for i in range(numeroClusters)]
    for i in range(len(dist)):
      if len(dist[i]) <= k:
        for j in range(len(grupos[i])):
          aux[i].append(grupos[i][j])
      else:
        for j in range(k):
          p= np.where(dist_aux[i]==np.min(dist_aux[i]))[0][0]
          aux[i].append(grupos[i][p])
          dist_aux[i][p]+= 100000

    e=[]
    for i in range(len(aux)):
      e.append(random.choice(aux[i]))
    

    elegidos= np.zeros((numeroClusters,5))
    for i in range(numeroClusters):
      for j in range(5):
        elegidos[i,j]= e[i][j]


    for i in range(len(elegidos)):
      cont=0
      j=0
      while elegidos[i][4] != a[j][4]:
        cont+=1
        j+=1
      a.pop(cont)

    '''
    Capacidad de cada p-median
    '''
    for i in range(len(elegidos)):
      elegidos[i][2]-= elegidos[i][3]
      elegidos[i][2]+= elegidos[i][3] 
  
    '''
    Hacemos la asignación a cada p-media
    '''
    ##calculo la distancia entre cada p-median y cada centros
    dists2= np.zeros((len(a),numeroClusters))
    for i in range(len(a)):
      for j in range(len(elegidos)):
        dists2[i,j]= math.floor(math.sqrt((a[i][0]-elegidos[j,0])**2+(a[i][1]-elegidos[j,1])**2))

    #matriz de asignacion
    entrega= [[] for i in range(numeroClusters)]
    for i in range(len(entrega)):
      entrega[i].append(elegidos[i][4])

    restantes=[]
    for i in range(len(a)):
      p= np.where(dists2[i]==np.min(dists2[i]))[0]
      if len(p)==1:
        if elegidos[int(p)][2] >= a[i][3]:
          elegidos[int(p)][2] -= a[i][3]
          elegidos[int(p)][3] += a[i][3]
          entrega[int(p)].append(a[i][4])
        else: 
          restantes.append(a[i])
      else:
        aux= random.randint(0,len(p)-1)
        p= p[aux]
        if elegidos[int(p)][2] >= a[i][3]:
          elegidos[int(p)][2] -= a[i][3]
          elegidos[int(p)][3] += a[i][3]
          entrega[int(p)].append(a[i][4])
        else:
          restantes.append(a[i])

    '''
    Si quedan nodos sin ser asignados, se vuelve a hacer una asignacion con esos nodos
    '''
    disRestantes2= np.zeros((len(restantes),len(elegidos)))
    for i in range(len(restantes)):
      for j in range(len(elegidos)):
        disRestantes2[i,j]= math.floor(math.sqrt(((restantes[i][0]-elegidos[j,0])**2+(restantes[i][1]-elegidos[j,1])**2)))

    pos= [[] for i in range(len(restantes))]
    for i in range(len(restantes)):
      for j in range(len(elegidos)):
        if restantes[i][3] <= elegidos[j][2]:
          pos[i].append(j)

    disRestantes= [[] for i in range(len(pos))]
    for i in range(len(pos)):
      for j in range(len(pos[i])):
        disRestantes[i].append(math.floor(math.sqrt((restantes[i][0]-elegidos[pos[i][j]][0])**2+(restantes[i][1]-elegidos[pos[i][j]][1])**2)))


    cont=0
    while len(restantes)>=1:
      p= np.where(disRestantes[cont]==np.min(disRestantes[cont]))[0][0]
      if elegidos[pos[cont][p]][2] > restantes[cont][3]:
        elegidos[pos[cont][p]][2]= elegidos[pos[cont][p]][2] - restantes[cont][3]
        elegidos[pos[cont][p]][3]= elegidos[pos[cont][p]][3] + restantes[cont][3]
        entrega[pos[cont][p]].append(restantes[cont][4])
        restantes.pop(0)
        pos.pop(0)
        disRestantes.pop(0)
      else:
        disRestantes[cont][p]+=1000000

    disT=[]
    for i in range(len(elegidos)):
      s=0
      for j in range(len(entrega[i])):
        s+= math.floor(math.sqrt((elegidos[i][0]-coordenadas[int(entrega[i][j]-1)][0])**2+(elegidos[i][1]-coordenadas[int(entrega[i][j]-1)][1])**2))
      disT.append(s)

    suma= sum(disT)
    asignaciones.append(entrega)
    objetivos.append(suma)
    disTs.append(disT)
    elegidoss.append(elegidos)

  p= np.argmin(objetivos)
  suma_f= objetivos[p]
  entrega_f= asignaciones[p]
  disT_f= disTs[p]
  elegidos_f= elegidoss[p]

  elapsed_time= time() - inicio
  
  entrega_aux= [[] for i in range(numeroClusters)]
  for i in range(len(entrega_aux)):
    entrega_aux[i].append(elegidos_f[i][4])

  for i in range(len(entrega_aux)):
    entrega_aux[i].append(len(entrega_f[i]))

  for i in range(len(entrega_aux)):
    for j in range(len(entrega_f[i])):
      entrega_aux[i].append(entrega_f[i][j])

  entrega2= [suma_f,elapsed_time*1000]
  entrega_aux.append(entrega2)

  for i in range(len(disT)):
    entrega_aux[i].append(disT_f[i])
  dfEntrega= pd.DataFrame(entrega_aux)

  arrayBin= np.zeros(len(listNodos))
  for i in range(len(elegidos_f)):
    arrayBin[int(elegidos_f[i][4]-1)]=1

  return suma_f, arrayBin, entrega_f, elegidos_f

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

#Método auxiliar que hace la asignación de los nodos a las medianas escogidas
def constructive2(listNodos,elegidos):
  inicio = time()

  coordenadas= np.zeros((len(listNodos),5))
  a= []
  for i in range(len(listNodos)): 
    coordenadas[i,0]= listNodos[i,0]
    coordenadas[i,1]= listNodos[i,1]
    coordenadas[i,2]= listNodos[i,2]
    coordenadas[i,3]= listNodos[i,3]
    coordenadas[i,4]= i + 1
    a.append([listNodos[i,0],listNodos[i,1],listNodos[i,2],listNodos[i,3],i+1])

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
  
  '''
  Hacemos la asignación a cada p-media
  '''
  ##calculo la distancia entre cada p-median y cada centros
  dists2= np.zeros((len(a),len(elegidos)))
  for i in range(len(a)):
    for j in range(len(elegidos)):
      dists2[i,j]= math.floor(math.sqrt((a[i][0]-elegidos[j][0])**2+(a[i][1]-elegidos[j][1])**2))

  #matriz de asignacion
  entrega= [[] for i in range(len(elegidos))]
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
      disRestantes2[i,j]= math.floor(math.sqrt(((restantes[i][0]-elegidos[j][0])**2+(restantes[i][1]-elegidos[j][1])**2)))

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

  fin = time()
  elapsed_time = fin- inicio 
  
  entrega_aux= [[] for i in range(len(elegidos))]
  for i in range(len(entrega_aux)):
    entrega_aux[i].append(elegidos[i][4])
    entrega_aux[i].append(len(entrega[i]))

  for i in range(len(entrega_aux)):
    for j in range(len(entrega[i])):
      entrega_aux[i].append(entrega[i][j])

  #Se calcula la distancias total de cada media asignada
  disT=[]
  for i in range(len(elegidos)):
    s=0
    for j in range(len(entrega[i])):
      s+= math.floor(math.sqrt((elegidos[i][0]-listNodos[int(entrega[i][j]-1)][0])**2+(elegidos[i][1]-listNodos[int(entrega[i][j]-1)][1])**2))
    disT.append(s)

  suma= sum(disT)

  entrega2= [suma, elapsed_time*1000]
  entrega_aux.append(entrega2)

  for i in range(len(disT)):
    entrega_aux[i].append(disT[i])
  dfEntrega= pd.DataFrame(entrega_aux)

  arrayBin= np.zeros(len(listNodos))
  for i in range(len(elegidos)):
    arrayBin[int(elegidos[i][4]-1)]=1
    
  return suma, entrega, arrayBin

#Método auxiliar que enumera las nuevas medianas elegidas
def elegir(lista,listNodos):
  elegidos=[]
  for i in range(len(lista)):
    if lista[i]==1:
      elegidos.append(listNodos[i])

  posiciones=[]
  for i in range(len(elegidos)):
    for j in range(len(listNodos)):
      if elegidos[i][0] == listNodos[j][0] and elegidos[i][1] == listNodos[j][1]:
        posiciones.append(j+1)

  elegidosf= np.zeros((len(elegidos),5))
  for i in range(len(elegidos)):
    for j in range(len(elegidos[0])):
      elegidosf[i][j]= elegidos[i][j]
  for i in range(len(elegidos)):
    elegidosf[i][4]= posiciones[i]
  
  return elegidosf
def vecindarioMediana(listaNodos,fI,vectorBin,mAsignaciones,elegidos):
  for i in range(len(vectorBin)):
    if vectorBin[i]==1:
      for j in range(len(vectorBin)):
        asignaciones2= vectorBin.copy()
        if vectorBin[j]==0:
          asignaciones2[i]=0
          asignaciones2[j]=1
          elegidos_j= elegir(asignaciones2,listaNodos)
          elegidos_j=elegidos_j.tolist()
          fN, vectorBinN, asigN= constructive2(listaNodos,elegidos_j)
          if fN<fI:
            return fN, asigN, vectorBinN, elegidos_j 
  return fI, vectorBin, mAsignaciones, elegidos

##Vecindario 2
def calculofo(listNodos,vect1,vect2,mAsignacioness,elegidos):
  mAsignaciones= copy.deepcopy(mAsignacioness)
  aux=[]
  eleg_aux=[]
  aux2=[]
  elegno_aux=[]
  for i in range(len(mAsignaciones)):
    if (mAsignaciones[i][0]!=vect1[0] and mAsignaciones[i][0]!=vect2[0]):
      aux.append(mAsignaciones[i])
  
  vecs= [vect1,vect2]
  asig_f= aux+vecs

  disT=[]
  for i in range(len(asig_f)):
    s=0
    for j in range(len(asig_f[i])):
      s+= math.floor(math.sqrt((listNodos[int(asig_f[i][0])-1][0]-listNodos[int(asig_f[i][j]-1)][0])**2+(listNodos[int(asig_f[i][0])-1][1]-listNodos[int(asig_f[i][j]-1)][1])**2))
    disT.append(s)
  
  elegidos_f= []
  for i in range(len(mAsignaciones)):
    l= listNodos[int(asig_f[i][0])-1]
    l= l.tolist()
    elegidos_f.append(l)
    elegidos_f[i].append(int(asig_f[i][0]))

  arrayBin= np.zeros(len(listNodos))
  for i in range(len(elegidos_f)):
    arrayBin[int(elegidos_f[i][4]-1)]=1
  
  suma= sum(disT)
  return suma, arrayBin, asig_f, elegidos_f

def intercamb(listNodos,fI,vectorBin,mAsignaciones,elegidos):
  for i in range(len(mAsignaciones)):
    for j in range(1, len(mAsignaciones[i])):
      for k in range(i+1, len(mAsignaciones)):
        for l in range(1, len(mAsignaciones[k])):
          elegidos[i][2] += listNodos[int(mAsignaciones[i][j]-1)][3]
          elegidos[i][3] -= listNodos[int(mAsignaciones[i][j]-1)][3]
          elegidos[k][2] += listNodos[int(mAsignaciones[k][l]-1)][3]
          elegidos[k][3] -= listNodos[int(mAsignaciones[k][l]-1)][3]
          if elegidos[i][2] >= listNodos[int(mAsignaciones[k][l]-1)][3] and elegidos[k][2]>= listNodos[int(mAsignaciones[i][j]-1)][3]:
            ele_i= mAsignaciones[i].copy()
            ele_k= mAsignaciones[k].copy()
            elegidos[i][2] -= listNodos[int(mAsignaciones[k][l]-1)][3]
            elegidos[i][3] += listNodos[int(mAsignaciones[k][l]-1)][3]
            elegidos[k][2] -= listNodos[int(mAsignaciones[i][j]-1)][3]
            elegidos[k][3] += listNodos[int(mAsignaciones[i][j]-1)][3]
            a= mAsignaciones[k][l]
            b= mAsignaciones[i][j]
            ele_i[j]= a
            ele_k[l]=b
            fN,vectorBinN,asigN,eleN= calculofo(listNodos,ele_i,ele_k,mAsignaciones,elegidos)
            if fN<fI:
              return fN,vectorBinN,asigN,eleN
          else:
            elegidos[i][2] -= listNodos[int(mAsignaciones[i][j]-1)][3]
            elegidos[i][3] += listNodos[int(mAsignaciones[i][j]-1)][3]
            elegidos[k][2] -= listNodos[int(mAsignaciones[k][l]-1)][3]
            elegidos[k][3] += listNodos[int(mAsignaciones[k][l]-1)][3]
  return fI, vectorBin, mAsignaciones, elegidos

##vecindario 3
def reasig(listNodos,nodo,mAsignacioness,asig_i, elegidos):
  mAsignaciones= copy.deepcopy(mAsignacioness)
  aux=[]
  ele_aux=[]
  ele_aux2=[]
  for i in range(len(mAsignaciones)):
    if nodo not in mAsignaciones[i]:
      aux.append(mAsignaciones[i])
      ele_aux.append(elegidos[i])
    else:
      elegidos[i][2] += listNodos[nodo-1][3]
      elegidos[i][3] -= listNodos[nodo-1][3]
      ele_aux2.append(elegidos[i])

  pos=[]
  for i in range(len(ele_aux)):
    if ele_aux[i][2] >= listNodos[nodo-1][3]:
      pos.append(i)

  disRestantes=[]
  for i in range(len(pos)):
    a=ele_aux[pos[i]][0]
    disRestantes.append(math.floor(math.sqrt((ele_aux[pos[i]][0]-listNodos[nodo-1][0])**2+(ele_aux[pos[i]][1]-listNodos[nodo-1][1])**2)))

  if len(disRestantes)>0:
    p= np.argmin(disRestantes)
    aux[p].append(nodo)
    ele_aux[p][2]-= listNodos[nodo-1][3]
    ele_aux[p][3]+= listNodos[nodo-1][3]

  asig_iaux= [asig_i]
  asig_f= aux+asig_iaux

  disT=[]
  for i in range(len(asig_f)):
    s=0
    for j in range(len(asig_f[i])):
      s+= math.floor(math.sqrt((listNodos[int(asig_f[i][0])-1][0]-listNodos[int(asig_f[i][j]-1)][0])**2+(listNodos[int(asig_f[i][0])-1][1]-listNodos[int(asig_f[i][j]-1)][1])**2))
    disT.append(s)

  elegidos_f=[]
  for i in range(len(mAsignaciones)):
    l= listNodos[int(asig_f[i][0])-1]
    l= l.tolist()
    elegidos_f.append(l)
    elegidos_f[i].append(int(asig_f[i][0]))

  arrayBin= np.zeros(len(listNodos))
  for i in range(len(elegidos_f)):
    arrayBin[int(elegidos_f[i][4]-1)]=1

  suma= sum(disT)
  return suma, arrayBin, asig_f, elegidos_f

def cambio(listNodos,fI,vectorBin, mAsignaciones,elegidos):
  for i in range(len(mAsignaciones)):
    for j in range(1,len(mAsignaciones[i])):
      asig_i= mAsignaciones[i].copy()
      nodo= asig_i[j]
      asig_i.pop(j)
      fN, vectorBinN, asigN,eleN = reasig(listNodos,nodo,mAsignaciones,asig_i,elegidos)
      if fN < fI:
        return fN, vectorBinN,asigN,eleN
  return fI, vectorBin, mAsignaciones,elegidos

def cambio(listNodos,fI,vectorBin, mAsignaciones,elegidos):
  for i in range(len(mAsignaciones)):
    for j in range(1,len(mAsignaciones[i])):
      asig_i= mAsignaciones[i].copy()
      nodo= asig_i[j]
      asig_i.pop(j)
      fN, vectorBinN, asigN,eleN = reasig(listNodos,nodo,mAsignaciones,asig_i,elegidos)
      if fN < fI:
        return fN, vectorBinN,asigN,eleN
  return fI, vectorBin, mAsignaciones,elegidos

# Método que mezcla los tres vecindarios
def vecindarios(j,listNodos,fI,vectoriBin,mAsignaciones,elegidos):
  if j==1:
    fN, vectorBinN, mAsigN, elegidosN= vecindarioMediana(listNodos,fI,vectoriBin,mAsignaciones,elegidos)
    return fN, vectorBinN, mAsigN, elegidosN
  elif j==2: 
    fN, vectorBinN, mAsigN, elegidosN= recocido_simulado(listNodos,fI,vectoriBin,mAsignaciones,elegidos, 15,0.02,0.5,7)
  else:
    fN, vectorBinN, mAsigN, elegidosN= intercamb(listNodos,fI,vectoriBin,mAsignaciones,elegidos)
    return fN, vectorBinN, mAsigN, elegidosN
  
  ## VND
def VNS(listNodos,numPmedians):
  fI, vectorBin, mAsigI, elegidosI= constructive(listNodos,numPmedians)
  starT= time()
  maxT= 60
  j=1

  while j<=3 and time() < starT + maxT :
    fN, vectorBinN, mAsigN, elegidosN= vecindarios(j,listNodos,fI, vectorBin, mAsigI, elegidosI)
    if fN< fI:
      j=1
      fI= fN
      vectoriBin= vectorBinN
      mAsigI= mAsigN
      elegidosI= elegidosN
    else:
      j+=1

  return fI, vectorBin, mAsigI, elegidosI

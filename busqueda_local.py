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

def recocido_simulado(listNodos,fI,vectorBin, mAsigI, elegidosI,T0,Tf,r,L):
  start= time()
  finalet= 60
  finalt=0
  while finalet > finalt-start:
    T=T0
    while T > Tf:
      l=0
      while l<L:
        l+=1
        fN, vectorBinN, mAsigN, elegidosN= vecindarioMediana(listNodos,fI,vectorBin,mAsigI,elegidosI)
        d= fN-fI
        if d<0:
          fI = fN
          vectorBin= vectorBinN
          mAsigI= mAsigN
          elegidosI= elegidosN
        else:
          if random.random() < math.e**(d/T):
            fI = fN
            vectorBin= vectorBinN
            mAsigI= mAsigN
            elegidosI= elegidosN
      T= r*T 
    finalt= time()

  entrega_aux= [[] for i in range(numPmedians)]

  for i in range(len(entrega_aux)):
    entrega_aux[i].append(elegidosI[i][4])
    entrega_aux[i].append(len(mAsigI[i]))

  for i in range(len(entrega_aux)):
    for j in range(len(mAsigI[i])):
      entrega_aux[i].append(mAsigI[i][j])

  entrega2= [fI, 1000]
  entrega_aux.append(entrega2)

  disT=[]
  for i in range(len(elegidosI)):
    s=0
    for j in range(len(mAsigI[i])):
      s+= math.floor(math.sqrt((elegidosI[i][0]-listNodos[int(mAsigI[i][j]-1)][0])**2+(elegidosI[i][1]-listNodos[int(mAsigI[i][j]-1)][1])**2))
    disT.append(s)

  for i in range(len(disT)):
    entrega_aux[i].append(disT[i])
  dfEntrega= pd.DataFrame(entrega_aux)

  #return dfEntrega

  return fI, vectorBin, mAsigI, elegidosI

def umbrales(listNodos,fI,vectorBin, mAsigI,elegidosI,T0):
  starT= time()
  maxT= 60

  iter1=0
  iter2=0
  T=T0
  while time() < maxT + starT:
    iter1+=1
    fN, vectorBinN, mAsigN, elegidosN= vecindarioMediana(listNodos,fI,vectorBin,mAsigI,elegidosI)
    if fN < fI + T:
      fI= fN
      vectorBin= vectorBinN
      mAsigI= mAsigN
      elegidosI= elegidosN
      iter2+=1
      if iter1*0.1 <= iter2:
        if iter1*0.3 >= iter2:
          T=T
        else:
          T= T/2
      else:
        T= 2*T

  return fI, vectorBin, mAsigI, elegidosI

def destruccionRepar(listNodos,fI, vectBin,mAsignaciones2,elegidos,nDes):
  # mAsig_copy= copy.deepcopy(mAsignaciones)
  # eleg_copy= copy.deepcopy(elegidos)
  mAsignaciones= mAsignaciones2
  aleMed=[]
  aleNod=[]
  numNod=[]

  for i in range(nDes):
    l1= random.randint(0,len(mAsignaciones)-1)
    aleMed.append(l1)
    l2= random.randint(1, len(mAsignaciones[l1])-1)
    aleNod.append(l2)
    numNod.append(mAsignaciones[l1][l2])
    mAsignaciones[l1].remove(mAsignaciones[l1][l2])

  for i in range(len(numNod)):
    elegidos[aleMed[i]][2] += listNodos[numNod[i]-1][3]
    elegidos[aleMed[i]][3] -= listNodos[numNod[i]-1][3]

  pos=[[] for i in range(nDes)]
  for i in range(len(numNod)):
    for j in range(len(elegidos)):
      if listNodos[numNod[i]-1][3] <= elegidos[aleMed[i]][2]:
        pos[i].append(j)

  disRestantes= [[] for i in range(len(pos))]
  for i in range(len(pos)):
    for j in range(len(pos[i])):
      disRestantes[i].append(math.floor(math.sqrt((listNodos[numNod[i]-1][0]-elegidos[pos[i][j]][0])**2+(listNodos[numNod[i]-1][1]-elegidos[pos[i][j]][1])**2)))
  
  numNod_copy= copy.deepcopy(numNod)
  pos_copy= copy.deepcopy(pos)
  cont=0
  while len(numNod)>=1:
    p= np.where(disRestantes[cont]==np.min(disRestantes[cont]))[0][0]
    if elegidos[pos[cont][p]][2] >= listNodos[numNod[cont]-1][3]:
      elegidos[pos[cont][p]][2] -= listNodos[numNod[cont]-1][3]
      elegidos[pos[cont][p]][3] += listNodos[numNod[cont]-1][3]
      mAsignaciones[pos[cont][p]].append(numNod[cont])
      numNod.pop(0)
      pos.pop(0)
      disRestantes.pop(0)
    else:
      disRestantes[cont][p] += 1000000


  disT=[]
  for i in range(len(elegidos)):
    s=0
    for j in range(len(mAsignaciones[i])):
      s+= math.floor(math.sqrt((elegidos[i][0]-listNodos[int(mAsignaciones[i][j]-1)][0])**2+(elegidos[i][1]-listNodos[int(mAsignaciones[i][j]-1)][1])**2))
    disT.append(s)

  suma= sum(disT)

  if suma < fI:
    return suma, vectBin, mAsignaciones, elegidos
  else:
    return fI, vectBin,mAsignaciones2, elegidos
def mixed(j,listNodos,fI,vectoriBin,mAsignaciones,elegidos,T0,Tf,r,L,T,nDes):
  if j==1:
    fN, vectorBinN, mAsigN, elegidosN= recocido_simulado(listNodos,fI,vectoriBin,mAsignaciones,elegidos,T0,Tf,r,L)
    return fN, vectorBinN, mAsigN, elegidosN
  elif j==2:
    fN, vectorBinN, mAsigN, elegidosN= umbrales(listNodos,fI,vectoriBin,mAsignaciones,elegidos,T)
    return fN, vectorBinN, mAsigN, elegidosN
  else:
    fN, vectorBinN, mAsigN, elegidosN= destruccionRepar(listNodos,fI,vectoriBin,mAsignaciones,elegidos,nDes)
    return fN, vectorBinN, mAsigN, elegidosN
 
def algoritmo(listNodos,numPmedians,T0,Tf,r,L,T,nDes,z):
  fI, vectorBin, mAsigI, elegidosI= constructive(listNodos,numPmedians)
  # maxxx=[60, 60*60*5, 60*60*5, 60*60*5, 60*60*5, 60*60*5, 60,60,60,60, 300,300,300,300,60*60*5 ]
  starT= time()
  maxT= 60*5
  j=1

  while j<=3 and  time() < starT+maxT :
    fN, vectorBinN, mAsigN, elegidosN= mixed(j,listNodos,fI, vectorBin, mAsigI, elegidosI,T0,Tf,r,L,T,nDes)
    if fN< fI:
      fI= fN
      vectoriBin= vectorBinN
      mAsigI= mAsigN
      elegidosI= elegidosN
    else:
      j+=1
    finalT = time()
  
  finaletime= time()- starT

  entrega_aux= [[] for i in range(numPmedians)]

  for i in range(len(entrega_aux)):
    entrega_aux[i].append(elegidosI[i][4])
    entrega_aux[i].append(len(mAsigI[i]))

  for i in range(len(entrega_aux)):
    for j in range(len(mAsigI[i])):
      entrega_aux[i].append(mAsigI[i][j])

  entrega2= [fI, finaletime*1000]
  entrega_aux.append(entrega2)

  disT=[]
  for i in range(len(elegidosI)):
    s=0
    for j in range(len(mAsigI[i])):
      s+= math.floor(math.sqrt((elegidosI[i][0]-listNodos[int(mAsigI[i][j]-1)][0])**2+(elegidosI[i][1]-listNodos[int(mAsigI[i][j]-1)][1])**2))
    disT.append(s)

  for i in range(len(disT)):
    entrega_aux[i].append(disT[i])
  dfEntrega= pd.DataFrame(entrega_aux)

  return dfEntrega

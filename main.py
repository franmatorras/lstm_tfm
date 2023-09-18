import pandas as pd
import pyarrow.parquet as pq
import os
import matplotlib.pyplot as plt
import random
from funciones import *
from generateScenarios import *

import torch
import torch.nn as nn

import numpy as np
import torch.optim as optim
import torch.utils.data as data

from sklearn.preprocessing import MinMaxScaler

# archivo = 'prod_cat_3.csv'
# df = pd.read_csv(archivo, sep=';')

# a, b = reducirNumero(df)

# pintar_movs(a, b)

# print(movimientos('../historico_stock_mensual'))

b = [8862771333148826, 1646149874053495008, 2866593716170126901, -7537662523680295790]
a = pd.read_csv('stock_reducido.csv', sep=';')
# a, b = reducirNumero(df)

pintar_movs(a, b)
# entrenamiento_prods(df, n_epochs=1500, learning_rate=0.005, lookback=10)

# df = pd.read_csv('stock_procesado.csv', sep=';')
# df = pd.read_csv('stock_reducido.csv', sep=';')
# print(df)

# years = 12 
# t = np.asarray(range(0, years * 12))

# data1 = makeScenario1(t=t, scale=1000.0, T=12, offset=0.0)
# df1 = pd.DataFrame({'hashProducto': 'data1', 'stockActual': data1})

# data2 = makeScenario2(t=t, scale=1000.0, T=12.0, offset=0, scaleS=200.0, TS = 6.0, offsetS=0)
# # data2 = makeScenario2(t=t, scale=1000.0, T=6.0, offset=1, scaleS=1000.0, TS = 4.0, offsetS=0.8)
# df2 = pd.DataFrame({'hashProducto': 'data2', 'stockActual': data2})

# data3 = makeScenario3(t=t, scale=1000.0, T=6.0, offset=0, scaleS=300.0, TS=8.0, offsetS=0, growthRate=15)
# data3 = makeScenario3(t=t, scale=1000.0, T=3.0, offset=1, scaleS=600.0, TS=8.0, offsetS=1.5, growthRate=10)
# data3 = makeScenario3(t=t, scale=1000.0, T=6.0, offset=1, scaleS=300.0, TS=8.0, offsetS=1.5, growthRate=15)
# df3 = pd.DataFrame({'hashProducto': 'data3', 'stockActual': data3})

# df = pd.concat([df1, df2, df3])

# print(df)


# entrenamiento_prods(df3, n_epochs=1000, learning_rate=0.005, lookback=6, pintar=True)

# df, prod = reducirNumero(df)
# df = df.loc[df.hashProducto == -7062210747801418926]
# hash_productos = df['hashProducto'].unique()
# for i, hash_producto in enumerate(hash_productos):
#     # print(hash_producto)
#     df_temp = df.loc[df.hashProducto == hash_producto]
    
    # plt.plot(df_temp['fechaStock'], df_temp['stockActual'])
    # plt.savefig('figuras/prueba' + str(hash_producto) + '.png', bbox_inches='tight')
    # plt.close()

# df = pd.read_csv('stock_duplicado.csv', sep=';')

# # print(len(df['hashProducto'].unique()))

# # pintar_movs(df, df['hashProducto'].unique())
# # df = df.loc[df.hashProducto == -7062210747801418926][['fechaStock', 'stockActual']]
# # df = df.loc[df.hashProducto == 2866593716170126901][['fechaStock', 'stockActual']]
# # df = df.loc[df.hashProducto == 4501832429819381531][['fechaStock', 'stockActual']]
# # df = df.loc[df.hashProducto == 8862771333148826][['fechaStock', 'stockActual']] ## devuelve resultados razonables
# # df = df.loc[df.hashProducto == -7537662523680295790][['fechaStock', 'stockActual']]
# # df = df.loc[df.hashProducto == 1646149874053495008][['fechaStock', 'stockActual']]
# # df = df.loc[df.hashProducto == -4966158994461411202][['fechaStock', 'stockActual']]
# # df = df.loc[df.hashProducto == -2410846049734484603][['fechaStock', 'stockActual']]
# df = df.loc[df.hashProducto == -1644914503179508428][['fechaStock', 'stockActual']] ## devuelve algo

# 
# df['hashProducto'] = 'prod_prueba'
# print(df)

# entrenamiento_prods(df, n_epochs=1000, learning_rate=0.005, lookback=6, pintar=True)

# duplicar_stock(df)
# df = pd.read_csv('stock_duplicado.csv', sep=';')
# timeseries = df[["stockActual"]].values.astype('float32')

# plt.plot(timeseries)
# plt.show()
# pintar_modelo(timeseries, 11)


# pintar_modelo(timeseries, 13)
# print(np.arange(2,11))
# probar_timeseries(np.arange(2, 12))

import pandas as pd
import os
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# import torch.nn.functional as F

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    scaler = MinMaxScaler()
    dataset_scaled = scaler.fit_transform(dataset)
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset_scaled[i:i+lookback]
        target = dataset_scaled[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y), scaler

def movimientos(parent_folder):
    df_final = pd.DataFrame(columns=['idProducto', 'fechaStock', 'stockActual'])

    # Lee todos los archivos parquet de la carpeta
    for folder_name in os.listdir(parent_folder):
        # Construye la ruta completa de cada carpeta
        try:
            folder_path = os.path.join(parent_folder, folder_name)
            table = pq.ParquetDataset(folder_path).read()
            df_show = table.to_pandas().groupby(['idProducto', 'fechaStock'])['stockActual'].sum().reset_index()
            df_final = pd.concat([df_final, df_show])
            # print(df)
        except:
            print(folder_name)


    df_sorted = df_final.sort_values(['idProducto', 'fechaStock'])

    nombre_archivo = 'prod_cat_3'  # Nombre del archivo a abrir

    # Abrir el archivo en modo lectura
    with open(nombre_archivo + '.txt', 'r') as archivo:
        # Leer el contenido del archivo y dividirlo en elementos separados por espacios
        contenido = archivo.read().split()



    # print()

    # Función que comprueba si un grupo cumple la condición
    # def cumple_condicion(g):
    #     return all((g['stockActual'].shift() != g['stockActual']) & 
    #             (g['stockActual'].shift(-1) != g['stockActual']))
    
    def cumple_condicion(g, n):
        return (g['stockActual'] > 0).sum() >= n


    # Filtrar los grupos que cumplan la condición
    df_filtrado = df_sorted.loc[df_sorted.idProducto.isin(contenido)].groupby('idProducto').filter(lambda x: cumple_condicion(x, 10))

    # Contar el número de registros por cada idProducto
    conteo_productos = df_filtrado.groupby('idProducto').size()

    # Filtrar los registros para mantener solo los productos que aparecen 45 veces
    productos_mov = conteo_productos[conteo_productos == 45].index.tolist()
    df_filtrado = df_filtrado[df_filtrado['idProducto'].isin(productos_mov)]

    df_anonim = df_filtrado.copy()

    df_anonim['hashProducto'] = df_anonim['idProducto'].apply(hash)
    df_anonim.to_csv(nombre_archivo + '.csv', sep=';', index=False)

    return df_anonim[['hashProducto', 'fechaStock', 'stockActual']]

def reducirNumero(df_filtrado):
    # productos = df_filtrado['hashProducto'].drop_duplicates().sample(n=9).tolist()
    productos = df_filtrado['hashProducto'].drop_duplicates().head(n=20).to_list()
    df_show = df_filtrado.loc[df_filtrado.hashProducto.isin(productos)]
    return df_show, productos

def duplicar_stock(df):
    hash_productos = df['hashProducto'].unique()
    df_doble = pd.DataFrame(columns=['hashProducto', 'fechaStock', 'stockActual'])
    for hash in hash_productos:
        df_b = df.loc[df.hashProducto == hash]
        df_b = df_b.reset_index(drop=True)
        new_df = pd.DataFrame(columns=['fechaStock', 'stockActual'])
        new_df['fechaStock'] = pd.to_datetime(df_b['fechaStock']) + pd.Timedelta(days=15)
        new_df['fechaStock'] = new_df['fechaStock'].dt.strftime('%Y-%m-%d')
        new_df = new_df.drop(new_df.index[-1]).reset_index(drop=True)
        

        for index, row in new_df.iterrows():
            if index < len(df_b) - 1:
                current_stock = df_b.loc[index, 'stockActual']
                next_stock = df_b.loc[index + 1, 'stockActual']
                mid_stock = (current_stock + next_stock) / 2
                new_df.loc[index, 'stockActual'] = mid_stock

        new_df['hashProducto'] = hash
        df_b = pd.concat([df_b, new_df], ignore_index=True).sort_values(['fechaStock'])
        df_doble = pd.concat([df_doble, df_b], ignore_index=True)
    df_doble.to_csv('stock_duplicado.csv', sep=';', index=False)
    


def pintar_movs(df_show, productos):
    # crear subplots en una cuadrícula de 3x3
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    # recorrer los elementos y plotear un subplot para cada uno
    for i, prod in enumerate(productos):
        # filtrar el DataFrame
        df_ind = df_show[df_show['hashProducto'] == prod]
        
        # crear un subplot en la posición correspondiente
        ax = axs[i // 2, i % 2]

        lim = max(df_ind.stockActual)
        lim = lim + lim/10
        ax.set_ylim([0, lim])

        # plotear el DataFrame filtrado
        ax.plot(df_ind['fechaStock'], df_ind['stockActual'])
        ax.set_xticks(df_ind['fechaStock'][::12]) # mostrar solo cada décima etiqueta
        ax.set_xticklabels(df_ind['fechaStock'][::12])
        
        # establecer el título del subplot
        ax.set_title(prod)
        
        # establecer el nombre del eje X
        ax.set_xlabel('fechaStock')
        
        # establecer el nombre del eje Y
        ax.set_ylabel('stockActual')
        
    # ajustar la disposición de los subplots y mostrar el gráfico
    # plt.tight_layout()
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()


 
class AirModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        # self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        # x = nn.functional.relu(x) 
        # x = torch.sigmoid(x)
        # x = self.dropout(x)
        x = self.linear(x)
        
        return x

def guardar_plot(nombre_archivo):
    contador = 0
    ruta_base = 'figuras/segunda_ronda/ejemplo'
    ruta_archivo = ruta_base + nombre_archivo + '.png'

    while os.path.exists(ruta_archivo):
        contador += 1
        ruta_archivo = ruta_base + nombre_archivo + f'_{contador}.png'

    plt.tight_layout()
    plt.savefig(ruta_archivo, bbox_inches='tight')
    plt.close()
    plt.show()

# class AirModel(nn.Module):
#     def __init__(self, dropout_rate=0.5):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.linear = nn.Linear(50, 1)
#     def forward(self, x):
#         x, _ = self.lstm(x)
#         # x = nn.functional.relu(x) 
#         x = torch.sigmoid(x)
#         x = self.dropout(x)  
#         x = self.linear(x)
        
#         return x

def ejemplo_lstm(X_train, y_train, scaler_train, X_test, y_test, scaler_test, train_size, test_size, lookback, model=None, optimizer=None, loss_fn=None, loader=None, timeseries=None, pintar=False, n_epochs=1000, hash_producto=None):
    f_perdida = []
    f_perdida_test = []
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        f_perdida.append(loss.item())
        # Validation
        if epoch % 100 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_loss = loss_fn(y_pred, y_test)
            # print(test_loss)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        f_perdida_test.append(test_loss.item())
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
    
    

    with torch.no_grad():
        # shift train predictions for plotting
        train_plot = np.ones_like(timeseries) * np.nan
        y_pred = model(X_train)
        y_pred = y_pred[:, -1, :]
        train_pred = scaler_train.inverse_transform(y_pred)  # Desescalado de las predicciones
        train_plot[lookback:train_size, 0] = train_pred[:, 0].reshape(-1)
        
        test_plot = np.ones_like(timeseries) * np.nan
        y_pred = model(X_test)
        y_pred = y_pred[:, -1, :]
        test_pred = scaler_test.inverse_transform(y_pred)
        test_plot[train_size+lookback:len(timeseries), 0] = test_pred[:, 0].reshape(-1)
    
    # plot
    if pintar:

    # Crear una figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        # print(f_perdida_test)

        # Plot de timeseries en el primer subplot (arriba)
        ax1.plot(timeseries, c='b', label='Original')
        ax1.plot(train_plot, c='r', linestyle=':', label='Train')
        ax1.plot(test_plot, c='g', linestyle='--', label='Test')
        ax1.legend()
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Predicción')

        # Plot de f_perdida en el segundo subplot (abajo)
        ax2.plot(f_perdida, c='r', label='Train')
        ax2.plot(np.linspace(0, len(f_perdida), len(f_perdida_test)), f_perdida_test, c='g', linestyle='--', label='Test')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss Function')
        ax2.legend()

        # Ajustar los subplots y mostrar la figura
        
        guardar_plot(str(hash_producto))

    return model





def entrenamiento_prods(df, n_epochs=2000, learning_rate=0.01, lookback=4, pintar=False):
    
    model = AirModel()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    # print(df)

    hash_productos = df['hashProducto'].unique()
    for i, hash_producto in enumerate(hash_productos):
        print("Iteración %d: producto = %s" % (i, hash_producto))
        df_producto = df.loc[df['hashProducto'] == hash_producto]
        timeseries = df_producto[["stockActual"]].values.astype('float32')

        # train-test split for time series
        train_size = int(len(timeseries) * 0.67)
        test_size = len(timeseries) - train_size
        train, test = timeseries[:train_size], timeseries[train_size:]

        # lookback = 4
        X_train, y_train, scaler_train = create_dataset(train, lookback=lookback)
        X_test, y_test, scaler_test = create_dataset(test, lookback=lookback)
        loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
        if i + 1 == len(hash_productos) + 1:
            pintar = True

        model = ejemplo_lstm(X_train, y_train, scaler_train, X_test, y_test, scaler_test, train_size, test_size, lookback, model=model, optimizer=optimizer, loss_fn=loss_fn, loader=loader, timeseries=timeseries, pintar=pintar, n_epochs=n_epochs, hash_producto=hash_producto)
        # data_loaders.append(data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8))

    # Guardar el modelo después del entrenamiento
    # torch.save(model.state_dict(), 'modelo_lstm15.pth')

def pintar_modelo(timeseries, n):
    nombre = 'modelo_lstm' + str(n) +'.pth'
    model = AirModel()
    model.load_state_dict(torch.load(nombre))
    model.eval()

    scaler = MinMaxScaler()
    timeseries_scaled = scaler.fit_transform(timeseries)
    data_scaled = torch.from_numpy(timeseries_scaled)


    with torch.no_grad():
        predictions_scaled = model(data_scaled)
    predictions = scaler.inverse_transform(predictions_scaled)

    # Crea un array con los índices de tiempo correspondientes a tus datos
    indices_tiempo = range(len(timeseries))

    # Grafica la serie temporal original y las predicciones
    plt.figure(figsize=(10, 6))
    plt.plot(indices_tiempo, timeseries, label='Serie Temporal Original')
    plt.plot(indices_tiempo, predictions, label='Predicciones')
    plt.xlabel('Índices de Tiempo')
    plt.ylabel('Valores')
    plt.title('Serie Temporal Original vs. Predicciones')
    plt.legend()
    plt.show()

def probar_modelos(timeseries, n_list, pr=True):
    # print(n_list)
    mejor = []
    for n in n_list:
        nombre = 'modelo_lstm' + str(n) + '.pth'
        # print('modelo_lstm' + str(n) +'.pth')
        model = AirModel()
        model.load_state_dict(torch.load(nombre), strict=False)
        model.eval()

        scaler = MinMaxScaler()
        timeseries_scaled = scaler.fit_transform(timeseries)
        data_scaled = torch.from_numpy(timeseries_scaled)


        with torch.no_grad():
            predictions_scaled = model(data_scaled)
        predictions = scaler.inverse_transform(predictions_scaled)
            
        squared_errors = np.square(np.subtract(timeseries, predictions))
        mean_squared_error = np.mean(squared_errors)
        if pr:
            print('\tModelo ', n)
            print("\t\tError cuadrático medio:", mean_squared_error)
        mejor.append([n, mean_squared_error])

        # correlation = np.corrcoef(timeseries, predictions)[0, 1]
        # print("Coeficiente de correlación:", correlation)
    min_valor = min(mejor, key=lambda x: x[1])
    n_modelo = min_valor[0]

    if pr:
        print(min_valor)
    return n_modelo

def probar_timeseries(n_list, pr=False):
    # df = pd.read_csv('stock_reducido.csv', sep=';')
    df = pd.read_csv('stock_procesado.csv', sep=';')
    n_modelos = []
    for hash in df['hashProducto'].unique():
        if pr:
            print('producto: ', hash)
        df_loop = df.loc[df.hashProducto == hash][['fechaStock', 'stockActual']]
        timeseries = df_loop[["stockActual"]].values.astype('float32')
        n_modelos.append(probar_modelos(timeseries, n_list, pr))
    valores_unicos, conteos = np.unique(n_modelos, return_counts=True)
    # conteos = np.bincount(np.array(n_modelos))

    contador = dict(zip(valores_unicos, conteos))
    print('éxitos de modelos:\n', contador)
from time import time
import pandas as pd

def obtiene_tiempos(fun, args, num_it=100):
    tiempos_fun = []
    for i in range(num_it):
        arranca = time()
        x = fun(*args)
        para = time()
        tiempos_fun.append(para - arranca)
    return tiempos_fun

def compara_entradas_funs(funs, nombres_funs, lista_args, N=10):
    entradas = []
    funcion = []
    tiempos = []
    lista_dfs = []
    for i, args in enumerate(lista_args):
        for j, fun in enumerate(funs):
            t = obtiene_tiempos(fun, [args], N)
            tiempos += t
            n = len(t)
            entradas += [str(args.shape)]*n
            funcion += [nombres_funs[j]]*n
        df = pd.DataFrame({'Long_entrada':entradas,
                           'Funcion':funcion,
                           'Tiempo':tiempos})
        lista_dfs.append(df)
    df = pd.concat(lista_dfs).reset_index()
    return df

def get_nice_policy(shape):
    a = [0] + [1]*(shape[0]-2) + [2]
    a = [a]*shape[1]
    return [x for sublist in a for x in sublist]



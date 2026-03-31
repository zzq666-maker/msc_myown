import numpy as np
import os
import scipy.io
from sklearn.preprocessing import OneHotEncoder
from copy import deepcopy

from local_dataset import balanced_subject_records

def segmentar_senales(db, labels):
    """
    Divide las señales EEG en segmentos de 512 instantes con un traslape del 50%.
    
    Args:
        db (dict): Diccionario donde las claves son los nombres de los sujetos y los valores
                   son matrices de forma CxT_i (C = canales, T_i = tiempo).
    
    Returns:
        dict: Nuevo diccionario con los segmentos de cada sujeto.
    """
    segmentos_db = {}
    segmento_tamano = 512
    paso = int(segmento_tamano * 0.5)  # Traslape del 50%
    i = 0
    
    segmentos = []
    y = []
    sbjs = []
    
    for sujeto, senal in db.items():
        C, T = senal.shape
        
        # Crear segmentos con traslape
        for inicio in range(0, T - segmento_tamano + 1, paso):
            segmento = senal[:, inicio:inicio + segmento_tamano]
            segmentos.append(segmento)
            y.append(labels[i])
            sbjs.append(sujeto)

        i += 1
    return np.array(segmentos), np.array(y), sbjs
    
    

def get_segmented_data():
    """
    Load the locally balanced EEG dataset using deterministic subject selection.
    """
    sujetos_control_records, sujetos_TDAH_records, _ = balanced_subject_records(seed=42)
    sujetos_TDAH = [subject_id for subject_id, _ in sujetos_TDAH_records]
    sujetos_control = [subject_id for subject_id, _ in sujetos_control_records]
    
    diagnostico = {}
    
    for sbj in sujetos_TDAH:
        diagnostico[sbj] = 1
    
    for sbj in sujetos_control:
        diagnostico[sbj] = 0
    
    # organizamos los datos de los sujetos con TDAH en un diccionario
    eeg_tdah = {}
    
    for sbj, mat_file_path in sujetos_TDAH_records:
        data = scipy.io.loadmat(mat_file_path)
        columna = list(data.keys())[-1]
        eeg_tdah[sbj] = data[columna].T
    
    # organizamos los datos de los sujetos de control en un diccionario
    eeg_control = {}
    
    for sbj, mat_file_path in sujetos_control_records:
        data = scipy.io.loadmat(mat_file_path)
        columna = list(data.keys())[-1]
        eeg_control[sbj] = data[columna].T
    
    db = eeg_control | eeg_tdah
    zeros = np.zeros(len(eeg_control))
    ones = np.ones(len(eeg_tdah))
    labels = np.hstack((zeros, ones))
    
    X, y, sbjs = segmentar_senales(db, labels)
    
    encoder = OneHotEncoder(sparse_output=False)
    
    # X = np.expand_dims(X, axis=-1)
    y = encoder.fit_transform(y.reshape(-1, 1))

    return X, y, sbjs

# Human Activity Recognition using CNN-1D + MLP (PyTorch)

## Descripción
Este proyecto implementa un modelo de **Deep Learning híbrido** para la
clasificación de **actividades humanas** a partir de **series de tiempo multivariadas**,
utilizando el dataset **Human Activity Recognition Using Smartphones** del UCI Machine Learning Repository.

El modelo combina:
- **CNN 1D** para la extracción de patrones temporales
- **MLP (Multi-Layer Perceptron)** para la clasificación final

## Dataset
- Fuente: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- Señales multivariadas provenientes de acelerómetro y giroscopio
- Ventanas temporales con múltiples canales

El dataset **no se incluye en el repositorio**.  
Debe descargarse manualmente y ubicarse en:

```

data/UCI_HAR_Dataset/

```

## Arquitectura del modelo
- **Bloques CNN 1D**
  - Convoluciones 1D
  - Activación ReLU
  - MaxPooling
- **MLP**
  - Capas fully connected
  - Softmax para clasificación multiclase

## Implementación
El proyecto incluye:
- `Dataset` y `DataLoader` personalizados
- División **train / validation / test**
- Loop de entrenamiento en PyTorch
- Cálculo de:
  - Loss (CrossEntropyLoss)
  - Accuracy
- **Early Stopping** y selección del mejor modelo
- Guardado del modelo entrenado (`.pt`)

## Resultados
Durante el entrenamiento se generan las siguientes gráficas:
- Train loss vs epoch
- Validation loss vs epoch
- Train accuracy vs epoch
- Validation accuracy vs epoch


## Guardado del modelo
El mejor modelo se guarda automáticamente en:
```

models/cnn1d_model_IsabellaP.pt

````

## Ejecución
Instalar dependencias:
```bash
pip install -r requirements.txt
````

Entrenar el modelo:

ejecutar el notebook:

```bash
jupyter notebook notebooks/Tarea1_Tensores_PyTorch_.ipynb
```

## Tecnologías

* Python 3
* PyTorch
* NumPy
* Matplotlib
* Scikit-learn

## Autor

Isabella Perez Caviedes

Proyecto académico desarrollado para un reto de Deep Learning.

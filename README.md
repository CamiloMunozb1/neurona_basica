## Neurona Artificial Simple (Regresión Logística) - Version 1.0
Se presenta una  implementación puramente didáctica de una Neurona Artificial simple (también conocida como Regresión Logística de una sola capa) programada desde cero en Python. El objetivo es demostrar los principios fundamentales del aprendizaje automático sin depender de librerías de alto nivel como TensorFlow o PyTorch.

## Objetivo del Proyecto.
El proyecto implementa el ciclo completo de Machine Learning para un único dato de entrenamiento:
  - Forward Pass (Paso hacia Adelante): Cálculo de la predicción.
  - Loss Function (Función de Pérdida): Medición del error (Entropía Cruzada).
  - Backpropagation (Retropropagación): Cálculo de los gradientes (las derivadas).
  - Gradient Descent (Descenso del Gradiente): Ajuste de los pesos para minimizar el error.

## Hiperparámetros y Valores Iniciales.
Los siguientes valores se utilizan para iniciar el entrenamiento de la neurona. El modelo aprende ajustando el Peso (w) y el Sesgo (b).
  Parámetro	Símbolo	Valor Inicial
  Peso Inicial	(w)	1.0
  Sesgo Inicial	(b)	−0.5
  Entrada de Entrenamiento	(x)	0.8
  Valor Real/Objetivo	(y)	1
  Tasa de Aprendizaje	(α)	0.25
  Regularización	(λ)	0

## Nota del desarrollador.
Esta es una neurona basica que se ira actualizando con el tiempo, ahora bien. Las posibles mejoras para otros desarrolladores y que recomiendo es colocar 'epocas' o iteraciones donde puedan ver como la neurona va aprendiendo en cada iteracion y ampliar el numero de entradas
o 'batches' para aumentar el entrenamiento.

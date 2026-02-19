# Laboratorio 1 Procesamiento digital de señales
### Docente: Carolina Corredor Bedoya
## Integrantes:
- Liseth Yuliana Clavijo Mesa
- Adriana Valentina Alarcon Ramirez
###  Febrero 2026
# Introduccion:


<img width="516" height="344" alt="image" src="https://github.com/user-attachments/assets/f9928fb2-6421-445a-a7bb-1eb08fd10864" />

Las señales biomédicas constituyen una fuente esencial de información clínica, ya que permiten analizar el funcionamiento de diferentes sistemas fisiológicos. Sin embargo, dichas señales suelen estar afectadas por diversos tipos de ruido, lo que dificulta su interpretación. En este laboratorio se trabajó con señales fisiológicas reales y generadas, con el objetivo de calcular sus principales parámetros estadísticos, evaluar la relación señal-ruido (SNR) y comparar los resultados obtenidos en diferentes condiciones de adquisición.

# Marco teorico:
Las señales biomédicas son variaciones eléctricas, mecánicas o químicas generadas por el cuerpo humano que permiten analizar la actividad fisiológica. Entre las más comunes se encuentran el electrocardiograma (ECG), el electroencefalograma (EEG) y la electromiografía (EMG).

El análisis de estas señales requiere considerar tanto la información útil (amplitud, frecuencia, patrones característicos) como la presencia de ruido, entendido como cualquier perturbación que distorsiona la señal original y puede provenir de fuentes internas (movimientos, interferencias fisiológicas) o externas (equipos, ambiente).

Para caracterizar cuantitativamente una señal, se emplean estadísticos descriptivos, entre los que destacan:

·Media: valor promedio de la señal, indica la tendencia central.

·Desviación estándar: mide la dispersión de los datos respecto a la media.

·Coeficiente de variación: relaciona la desviación estándar con la media, expresando la variabilidad relativa.

·Histograma: representa gráficamente la distribución de frecuencias.

·Función de probabilidad: describe la probabilidad de ocurrencia de ciertos valores de la señal.

Además, la relación señal-ruido (SNR) es un parámetro fundamental que compara la potencia de la señal útil frente a la potencia del ruido. Un SNR alto implica una señal clara y confiable, mientras que un SNR bajo indica una señal fuertemente contaminada.

El uso de herramientas computacionales como Python, con librerías como NumPy, SciPy y Matplotlib, permite importar, procesar y analizar señales fisiológicas de manera eficiente. Asimismo, plataformas como PhysioNet ofrecen bases de datos abiertas que facilitan el acceso a señales reales para fines académicos e investigativos.

# OBJETIVOS: 
Objetivo General: Caracterizar una señal biomédica en función de 
parámetros estadísticos. 
## Objetivos Específicos 
- Identificar las principales magnitudes estadísticas que describen una señal 
biomédica. 
- Emplear funciones aritméticas y comandos específicos de un entorno de 
programación para calcular diferentes parámetros estadísticos de una 
señal biomédica. 
- Plantear hipótesis desde la fisiología que expliquen los valores estadísticos 
obtenidos.



# PROCEDIMIENTO, MÉTODO O ACTIVIDADES A DESARROLLAR EN LA PRÁCTICA: 

Las señales medidas de un entorno real, en este caso, las señales biomédicas están caracterizadas por contener información relevante, como amplitud y frecuencia e información que la contamina, denominada ruido.  
Adicionalmente, existe información que puede describir una señal biomédica a partir de variables estadísticas. Para esta práctica de laboratorio el estudiante deberá descargar una señal fisiológica y calcular los estadísticos que la describen, explicando para qué sirve cada uno. 
PARTE A.  
1. Entrar a bases de datos de señales fisiológicas como physionet, buscar y descargar una señal fisiológica de libre elección. Tenga en cuenta que, si por algún motivo no puede calcular todos los parámetros solicitados porque la señal es muy corta, deberá descargar una nueva señal.
2. Importar la señal en python y graficarla. Para esto pueden hacer uso de 
cualquier compilador, como spyder, google colab, sistema operativo Linux, etc. 
Se recomienda descargar la suite de Anaconda completa para utilizar 
Python (Spyder) en Windows. 
3. Calcular los estadísticos descriptivos de dos maneras diferentes cuando sea 
posible: la primera vez, programando las fórmulas desde cero; la segunda vez, 
haciendo uso de las funciones predefinidas de python.  
Los estadísticos que se espera obtener son:  
a. Media de la señal 
b. Desviación estándar 
c. Coeficiente de variación 
d. Histogramas 
e. Asimetría (skewness)
f.curtosis


# PARTE A.
Los registros ECG descargados desde PhysioNet se descargaron en formato WFDB, este formato consiste en dos archivos principales: 
1).dat estos son los datos binarios de la señal y 2).hea que es la informacion de muestreo, canales y ganacia del tal señal.
Estos dos archivos son los que permiten leer correctamente la señal ECG en copiladores como Phyton.
```python
import wfdb
record = wfdb.rdrecord('100') 
signal = record.p_signal
fs = record.fs
```
En la primera parte del código se cargan los datos asociado automaticamente el .hea y .dat, luego devuelve la matriz de la señal en forma de float y por ultimo muestra la frecuencia de muestreo en Hz.
En las señales ECG tipicamente hay dos derivación y normalmente se escoge uno para el analisis estadistico, seleccciona un vector 1D con las mediciones de anplitud del ECG
```python
ecg = signal[:,0]

tiempo = np.arange(0, 5*fs)
plt.figure()
plt.plot(tiempo/fs, ecg[0:5*fs])
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (mV)")
plt.title("ECG - Registro 100")
plt.show()
```
Se grafica la señal ECG en un tiempo de 5 segundos. 
<img width="712" height="550" alt="image" src="https://github.com/user-attachments/assets/f9192752-d68e-4ad6-b4ba-3e0876349ed0" />

estadisticos a calcular. *
1)Media de la señal. la media muestral representa el promedio de la amplitud del ECG, como la señal vibra alrededor de creo, tiende a estar cerca de cero.
​
```python
#media 
N=len(ecg)  
suma=0
for i in range(N):
    suma+=ecg[i]
media_m=suma/N
print("Media:",media_m)

Media: -0.30629897692306546
```
En señales fisiologicas bien centradas este valor suele verse cercano a 0 debido a la naturaleza oscilatoria de la señal.





















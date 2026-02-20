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

estadisticos a calcular. 

A) Media de la señal. la media muestral representa el promedio de la amplitud del ECG, como la señal vibra alrededor de creo, tiende a estar cerca de cero.

Para hallar la maedia se tendra en cuenta la siguiente formula:


<img width="190" height="98" alt="image" src="https://github.com/user-attachments/assets/6a448f87-f51c-4e76-99c2-dd2cf9b7b92a" />

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

B) Derivación estándar. 

Este estadistico cuantifica la variabilidad de la amplitud del ECG respecto a su valor medio, en el electrocardiograma esta dispersión esta influenciada principalmente por la amplitud de los complejos QRS 

<img width="321" height="81" alt="image" src="https://github.com/user-attachments/assets/e9e466dd-e9fb-4100-95f7-3bc8fcafa27c" />


```python
suma_cuadrados=0
for i in range(N):
    suma_cuadrados+=(ecg[i]-media_m)**2

varianza_m = suma_cuadrados / (N - 1)
desviacion_m = np.sqrt(varianza_m)

print("varianza m:" ,varianza_m)
print("Desviación estándar:",desviacion_m)
```
En la señal ECG la desviación estándar es de 0.193 mV esto significa que la señal se desvia ±0.19 mV respecto a la linea base en este caso la desviación es pequeña lo que indica una señal estable con poco ruido.

C) Coeficiente de variación.

El CV es una medida de dispersión relativa que se calcula como:

<img width="184" height="80" alt="image" src="https://github.com/user-attachments/assets/6bb83d51-0e5f-4bac-a804-9be82b11a587" />

en un ECG la media suele estar serca a 0 por lo que se se divide algo cercano a ero el CV se podria dsitapara y perder sentido, lo cual es normal en señales centradas.
```phyton

cv=abs(desviacion_m/media_m)
cv_porcentaje=cv*100

print("Coeficiente de variación:",cv)
print("Coeficiente de variación(%):",cv_porcentaje)
```

el coeficiente de variación no es el descriptor estadistico mas adecuado para señales como ECG ya que mas medidas dan cercanas a cero.

D) Histograma.

Es una representación de la frecuencia con las que aparecen ciertos valores, para en eje X la amplitud del ECG (mV), en el eje Y se encuentra la frecuencia de ocurrencia, el histograma permite ver si la señal se parce a distribucion normal, tiene colas y si es simétrica o no.
```phyton
plt.figure()
plt.hist(ecg,bins=100,density=True)
plt.xlabel("Amplitud (mV)")
plt.ylabel("Densidad")
plt.title("Histograma normalizado del ECG")
plt.show()
```
<img width="688" height="555" alt="image" src="https://github.com/user-attachments/assets/170f3866-b8bb-42f6-af34-597496e0b5a9" />

En el histograma se observa una concentración grande al rededor de cero, correspindiente al comportamiento basal del ECG, se observan colas hacia valores positivos asociasos a cojmplejos QRS esto por los picos y sus distribución no perfecatemnte gaussiana, se representó el histograma en términos de densidad para aproximar la función de distribución de probabilidad de la señal y permitir un análisis estadístico independiente del número total de muestras.

E) Asimetria:


<img width="328" height="111" alt="image" src="https://github.com/user-attachments/assets/0eedbf6a-79e8-45e5-9e2d-dacbd2f0dfb7" />

la asimetria mide si la distribución tiene cola hacia la derecha entonces skewness positiva, pero si por otro lado tine cola hacia la izquierda entonces skewness negativa, la distibucion es simetrica cercana a cero.
```phyton

N=len(ecg)
media_m=np.mean(ecg)
desviacion_m=np.std(ecg,ddof=1)
z=(ecg-media_m)/desviacion_m
suma_cubos=np.sum(z**3)
skewness_muestral=(N/((N-1)*(N-2)))*suma_cubos
print("Skewness muestral corregida:",skewness_muestral)
from scipy.stats import norm
plt.figure()

#Histograma en densidad
plt.hist(ecg,bins=100,density=True,alpha=0.6)
# Ajuste normal con misma media y desviación
x=np.linspace(min(ecg),max(ecg),1000)
plt.plot(x,norm.pdf(x,media_m,desviacion_m))
plt.xlabel("Amplitud (mV)")
plt.ylabel("Densidad")
plt.title("Histograma vs Distribución Normal")
plt.show()

```
<img width="687" height="558" alt="image" src="https://github.com/user-attachments/assets/d4fe0f00-fd36-48c5-8e0f-3d7f41cd812d" />

El coeficiente de asimetría positivo y elevado indica una distribución con cola pronunciada hacia valores positivos, atribuible a la presencia de complejos QRS de alta amplitud.

F)Curtosis

<img width="352" height="107" alt="image" src="https://github.com/user-attachments/assets/b1391e78-7258-478a-a3d4-d62ac7b46eb8" />

La curtosis cuantifica el grado de concentración y el peso de las colas deuna distribución con respecto a la normal, en un ECG mide que tan pronunciados y extremos son los picos  en especial del complejo QRS en comparación con la otra señal.
```phyton
#CURTOSIS
suma_cuartos=0

for i in range(N):
    suma_cuartos+=((ecg[i]-media_m)/desviacion_m)**4
curtosis_manual=suma_cuartos/N
exceso_curtosis=curtosis_manual-3
print("Curtosis:",curtosis_manual)
print("Exceso de curtosis:",exceso_curtosis)
```
normal=0

leptocúrtica >0

platicúrtica <0


el valor psitivo elevado de la curtosis indica una distribución leptocúrtica, caracterizada por un pico central pronunciado, atribuido a la presecia de complejos QRS de alta amplitud 

# PARTE B

En esta parte se realizo el analisis estadistico descriptivo de la señal ECG en el dominio temporal a diferencia de la parte A, donde se analizarlos los datos puntuales en esta parte se evaluara la señal completa con el fin de caracterizar el comporamiento global, la señal ECG es una señal biomédica discreta obtenida mediente muestreo, cuya amplitud representa la actividade electrica cardíaca en función del tiempo.

Se descargo la señal de phyron y fue procesada para luego gragicarla.
```phyton
import numpy as np

ecg = np.loadtxt("senal_ecg.txt", skiprows=1)
print("Número de muestras:", len(ecg))
print("Primeras 10 muestras:", ecg[:10])

data=np.loadtxt("senal_ecg.txt", skiprows=1)
tiempo=data[:,0]
ecg=data[:,1]
print("Número de muestras:", len(ecg))

import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
plt.plot(ecg)

plt.title("Señal ECG")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()
```

<img width="715" height="265" alt="image" src="https://github.com/user-attachments/assets/f0420d58-dae2-4d66-a531-d1527e8294ed" />

Se calcularon las siguientes medidas estadísticas:

A) Media
```phyton
import numpy as np
media=np.mean(ecg)
print("Media:", media)
```
B) Desviación estándar
```phyton
desviacion=np.std(ecg, ddof=1)
print("Desviación estándar:", desviacion)
```

C) Coeficiente de variación
```phyton
varianza=np.var(ecg, ddof=1)
print("Varianza:", varianza)
cV=desviacion/media
print("CV:", cv)
```

D) Histograma de distribución
```phyton
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.hist(ecg, bins=50)
plt.title("Histograma de la señal ECG")
plt.xlabel("Amplitud")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()
```
<img width="703" height="473" alt="image" src="https://github.com/user-attachments/assets/1debfe38-855c-4657-a5e0-962784da85a7" />


E) Asimetría (Skewness)
```phyton
from scipy.stats import skew
asimetria = skew(ecg, bias=False)
print("Asimetría:", asimetria)
```

F) Curtosis 

```phyton
from scipy.stats import kurtosis
curt=kurtosis(ecg, bias=False)
print("Curtosis:",curt)
```

En esta parte el analisis estadistico nos muestra que la señal presenta variabilidad significativa, no sigue un distribución normal, posee asímetria positiva y curtosis elevada.
El comportamiento estadistico de la señal esta influenciado por el complejo QRS 
El analisisi descriptivo realizado en esta parte permite caracterizar cuantitativamente la señal ECG en el dominio temporal.


















# PARTE C
Para esta parte, usar la misma señal de la parte B: 
1. Investigar qué es la relación señal ruido (SNR):
## SEÑAL SNR 

La relación señal-ruido, también conocida como SNR por sus siglas en inglés (Signal-to-Noise Ratio), es un concepto básico cuando hablamos de cualquier sistema que maneje señales. De todos modos, es en el mundo del audio donde toma especial relevancia. Simplificándolo mucho, la SNR es una medida que compara la potencia de una señal deseada, en este caso, el sonido que se va a escuchar, con la potencia del ruido de fondo no deseado que está presente en el mismo sistema.

<img width="601" height="350" alt="image" src="https://github.com/user-attachments/assets/27707dc0-7c84-4ce9-b2bf-dc6b2b8d2731" />


## Parte A ruido gaussiano y medir el SNR 

En primer lugar, se generó ruido gaussiano para simular la presencia de interferencias aleatorias en la señal ECG, similares al ruido térmico o electrónico presente en sistemas de adquisición biomédica.
La desviación estándar del ruido se definió como el 5 % de la desviación estándar de la señal ECG original, con el fin de controlar la intensidad del ruido y evitar una degradación excesiva de la señal:
### σₙ = 0.05 · std(señal ECG)
Posteriormente, se generó un vector de ruido con distribución normal, media cero y desviación estándar σₙ. Este ruido se sumó punto a punto a la señal ECG original, obteniendo así una señal contaminada con ruido gaussiano.
Para cuantificar el efecto del ruido sobre la señal, se calculó la Relación Señal-Ruido (SNR) en decibelios (dB).

```phyton

import numpy as np
sigma_g=0.05*np.std(ec_g_1d_signal)
gaussian_noise=np.random.normal(0, sigma_g, len(ec_g_1d_signal))
ec_g_gaussian_noisy= ec_g_1d_signal + gaussian_noise
power_signal_gaussian=np.mean(ec_g_1d_signal**2)
power_noise_gaussian=np.mean(gaussian_noise**2)
snr_gaussian=10*np.log10(power_signal_gaussian/power_noise_gaussian)
print(f"SNR para el ruido gaussiano (dB): {snr_gaussian}")

```

Primero, se estimó la potencia de la señal original, calculada como el valor medio del cuadrado de la señal ECG:

### Pₛ = mean(ECG²)

Luego, se calculó la potencia del ruido gaussiano de forma análoga:

### Pₙ = mean(ruido²)

Finalmente, el SNR se obtuvo mediante la expresión:

### SNR (dB) = 10 · log₁₀(Pₛ / Pₙ)

Este valor indica cuántas veces la potencia de la señal es mayor que la del ruido; a mayor SNR, menor degradación de la señal.

SNR para el ruido gaussiano (dB): 33.73230538099961
### Grafica de la señal contaminada vs señal original

<img width="1245" height="547" alt="image" src="https://github.com/user-attachments/assets/e824240d-562f-4329-bd77-9d49184c95d2" />

```phyton

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))
plt.plot(tiempo, ec_g_1d_signal, label='Señal ECG original', alpha=0.7)
plt.plot(tiempo, ec_g_gaussian_noisy, label='Señal con ruido gaussiano', alpha=0.7)
plt.title('Señal original vs. Señal ECG con ruido gaussiano')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (mV)')
plt.legend()
plt.grid(True)
plt.show()

```

La figura muestra una comparación temporal entre la señal ECG original y la señal contaminada con ruido gaussiano, donde la señal original presenta complejos QRS bien definidos y una línea base estable, mientras que la señal con ruido conserva la morfología general del ECG pero exhibe pequeñas fluctuaciones aleatorias distribuidas uniformemente a lo largo de toda la señal; estas perturbaciones, características del ruido gaussiano, no generan picos abruptos sino variaciones suaves alrededor del valor original, permitiendo que los eventos principales como los picos R sigan siendo claramente identificables, aunque con menor nitidez, lo cual evidencia que la estructura del ECG no se ve gravemente distorsionada y es consistente con el valor de SNR obtenido, que indica una relación señal-ruido moderada.


### Parte B ruido impulso y medir el SNR

se seleccionó la señal ECG original correspondiente a una sola derivación. Posteriormente, se definió un porcentaje de muestras a contaminar (1 % del total), con el objetivo de simular interferencias breves y de gran amplitud típicas del ruido impulso, como desconexiones momentáneas de electrodos o movimientos bruscos del paciente. A partir de este porcentaje, se calculó el número total de impulsos y se generó un vector de ruido inicialmente nulo. Los índices donde se introduciría el ruido fueron seleccionados de manera aleatoria, garantizando que los impulsos se distribuyeran de forma no periódica a lo largo del tiempo. En dichos índices se asignaron valores aleatorios dentro de un rango proporcional a la amplitud máxima de la señal ECG original, permitiendo que los impulsos presentaran amplitudes tanto positivas como negativas. Finalmente, este ruido impulso se sumó a la señal original para obtener la señal ECG contaminada.

```phyton

import numpy as np

ec_g_1d_signal = data[:, 1]
percentage_impulses = 0.01
impulse_amplitude_factor = 1.0
max_ecg_amplitude = np.max(np.abs(ec_g_1d_signal))
num_impulses = int(len(ec_g_1d_signal) * percentage_impulses)
impulse_noise = np.zeros_like(ec_g_1d_signal)
impulse_indices = np.random.choice(len(ec_g_1d_signal), num_impulses, replace=False)
impulse_noise[impulse_indices] = np.random.uniform(
    -max_ecg_amplitude * impulse_amplitude_factor,
    max_ecg_amplitude * impulse_amplitude_factor,
    num_impulses
)
ec_g_impulse_noisy = ec_g_1d_signal + impulse_noise
power_signal = np.mean(ec_g_1d_signal**2)
power_noise = np.mean(impulse_noise**2)
snr_impulse = 10 * np.log10(power_signal / power_noise)
print(f"SNR con ruido impulso (dB): {snr_impulse}")

```
SNR con ruido impulso (dB): 19.948019014079836

Una vez generada la señal con ruido impulso, se calculó la potencia de la señal ECG original como el valor medio del cuadrado de sus muestras. De forma análoga, se estimó la potencia del ruido impulso a partir del vector de ruido generado. Con estas potencias se calculó la relación señal-ruido (SNR) en decibelios, lo que permitió cuantificar el nivel de degradación introducido por este tipo de ruido.

### Grafica de la señal contaminada vs señal original

<img width="1245" height="547" alt="image" src="https://github.com/user-attachments/assets/47d33bce-a624-41e7-92c0-5db51d07d908" />

La gráfica muestra una comparación temporal entre la señal ECG original y la señal contaminada con ruido tipo impulso, donde la señal original conserva una morfología clara con complejos QRS bien definidos y una línea base relativamente estable, mientras que la señal con ruido presenta picos abruptos y aislados de gran amplitud, tanto positivos como negativos, que aparecen de manera esporádica a lo largo del tiempo; estos impulsos sobresalen de la señal fisiológica y generan distorsiones locales significativas que pueden enmascarar eventos relevantes del ECG, evidenciando que, aunque el ruido impulso afecta solo un número reducido de muestras, su impacto sobre la calidad de la señal es considerable, lo cual se refleja en un SNR más bajo en comparación con otros tipos de ruido.


```phyton
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))
plt.plot(tiempo, ec_g_1d_signal, label='Señal electrocardiografica original', alpha=0.7)
plt.plot(tiempo, ec_g_impulse_noisy, label='Señal impulso', alpha=0.7)
plt.title('Señal original vs señal de impulso')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (mV)')
plt.legend()
plt.grid(True)
plt.show()

```

### Parte C ruido tipo artefacto y medir el SNR

En este caso, la señal ECG original fue contaminada con ruido tipo artefacto mediante la simulación de una deriva de la línea base, fenómeno común en registros electrocardiográficos y asociado a movimientos respiratorios, cambios de postura o desplazamientos lentos de los electrodos. Para ello, se generó una señal sinusoidal de baja frecuencia (0.05 Hz) y amplitud controlada (0.1), la cual representa una variación lenta y periódica de la línea base. Esta señal de ruido se calculó en función del tiempo y posteriormente se sumó punto a punto a la señal ECG original, obteniéndose así una señal contaminada con ruido de artefacto.
Una vez obtenida la señal con ruido de artefacto, se calculó la potencia de la señal ECG original como el promedio del cuadrado de sus muestras. De manera similar, se estimó la potencia del ruido de artefacto a partir de la señal sinusoidal generada. Con estas potencias se calculó la relación señal-ruido (SNR) en decibelios, permitiendo evaluar cuantitativamente el impacto de la deriva de la línea base sobre la calidad de la señal.

```phyton

import numpy as np
amplitude_wander = 0.1
frequency_wander = 0.05
baseline_wander_noise = amplitude_wander * np.sin(2 * np.pi * frequency_wander * (tiempo - tiempo[0]))
ec_g_artifact_noisy = ec_g_1d_signal + baseline_wander_noise
power_signal_artifact = np.mean(ec_g_1d_signal**2)
power_noise_artifact = np.mean(baseline_wander_noise**2)
snr_artifact = 10 * np.log10(power_signal_artifact / power_noise_artifact)
print(f"SNR con ruido de artefacto (dB): {snr_artifact}")

```

Una vez obtenida la señal con ruido de artefacto, se calculó la potencia de la señal ECG original como el promedio del cuadrado de sus muestras. De manera similar, se estimó la potencia del ruido de artefacto a partir de la señal sinusoidal generada. Con estas potencias se calculó la relación señal-ruido (SNR) en decibelios, permitiendo evaluar cuantitativamente el impacto de la deriva de la línea base sobre la calidad de la señal.


### Grafica de la señal contaminada vs señal original


<img width="1245" height="547" alt="image" src="https://github.com/user-attachments/assets/d281b217-91d1-4e51-8caf-a745fd6aa3b0" />

La gráfica presenta una comparación temporal entre la señal ECG original y la señal contaminada con ruido de artefacto, donde la señal original mantiene complejos QRS claramente definidos y una línea base estable, mientras que la señal con artefacto exhibe una oscilación lenta y periódica de la línea base que desplaza la señal completa hacia valores positivos y negativos a lo largo del tiempo; este tipo de ruido no introduce picos abruptos ni perturbaciones de alta frecuencia, pero sí modifica progresivamente el nivel de referencia del ECG, lo que puede dificultar la identificación precisa de segmentos como el ST o la amplitud real de las ondas, evidenciando una degradación gradual de la señal que se refleja en el valor del SNR obtenido.

```phyton
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))
plt.plot(tiempo, ec_g_1d_signal, label='Señal Electrocardiografica original', alpha=0.7)
plt.plot(tiempo, ec_g_artifact_noisy, label='Señal con ruido de artefacto', alpha=0.7)
plt.title('Señal original vs. Señal con ruido de artefacto')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (mV)')
plt.legend()
plt.grid(True)
plt.show()
```
## ANALISIS
## DIAGRAMAS DE FLUJO 
# PARTE A
# PARTE B
# PARTE C

# Preguntas para la discucion 

- ¿Los valores estadísticos calculados sobre la señal sintética son 
exactamente iguales a los obtenidos a partir de la señal real? ¿Por qué?

No, los valores estadísticos de una señal sistentica y unba señal real difieren, aunque pueden ser similares en orden de margnitud,esto se debe a que la señal sintética es generada mediante un modelo matemático idealizado, mientras que la señal real incorpora variabilidad fisiologica, ruido instrumental y posibles artefactos de medición.

En la parte a se trabajo una señal sintética, en la parte b se trabajo una señal real de ECG.
En la media, la señal sintetica, la media probablemente estuvo muy cercana a cero o perfectamenre centrada mientras que en la señal real tambien cercana a cero pero presento desviaciones pequeñas que se pueden atribuir a offset intrumental, ruido de adquisición, para la desviacion estandar y varianza en la señal real la dispersión suele ser mayor porque los complejos QRS no son idénticos, hay variabilidad del latido, ecxiste ruido fisiologico. La señal sintetica en cambio es mas regular y controlada, la asimetria la señal real muestra uhna mayor asimetria debido a que los QRS general extremos positivos y la señal sinetica puede ser menor.
La señal real presenta características no gaussianas más marcadas que la señal sintética, lo cual se evidencia en el aumento de la curtosis y la asimetría.


- ¿Afecta el tipo de ruido el valor de la SNR calculado? ¿Cuáles podrían ser 
las razones? 


-Sí, el tipo de ruido sí afecta el valor de la SNR, porque no todos los ruidos alteran la señal de la misma forma. El ruido gaussiano se reparte a lo largo de toda la señal con pequeñas variaciones, por lo que la SNR suele disminuir de manera moderada; en cambio, el ruido impulso, aunque aparece en pocas muestras, tiene amplitudes muy altas que aumentan bastante la potencia del ruido y hacen que la SNR sea mucho menor; por su parte, el ruido tipo artefacto, como la deriva de la línea base, no genera picos bruscos, pero introduce una variación lenta que se mantiene en el tiempo y termina afectando la SNR de forma progresiva. En general, la diferencia en la amplitud, la duración y la forma en que cada ruido se distribuye en la señal explica por qué la SNR cambia según el tipo de ruido aplicado.

# Conclusión 

El análisis descriptivo permitió cuantificar el comportamiento estadístico de la señal ECG en el dominio temporal. Las medidas de tendencia central, dispersión y forma evidenciaron que la señal presenta variabilidad significativa y no sigue una distribución normal ideal. Los valores de asimetría y curtosis obtenidos demostraron que la señal posee características no gaussianas, principalmente debido a la presencia de complejos QRS, los cuales generan picos de alta amplitud y colas pesadas en la distribución, esto confirma que el ECG no puede modelarse estrictamente como un proceso aleatorio normal.
La desviación estándar y la varianza reflejan la energía estadística de la señal y su variabilidad fisiológica. En la señal real se evidenció mayor dispersión debido a la variabilidad latido a latido y al ruido inherente al proceso de adquisición, el histograma permitió visualizar la distribución empírica de las amplitudes, evidenciando concentración alrededor de valores bajos y presencia de valores extremos asociados a los complejos ventriculares. Esta representación gráfica complementa la interpretación numérica de los estadísticos.

















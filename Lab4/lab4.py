#####################################################
#       Lab 4 - Redes 1-2019 - Nicolás Gutierrez      #
#####################################################

# Version  :     5.0
# Fecha    :  20-06-2019


# A continuacion se muestra el cuarto lab de Redes,
# correspondiente a la materia de modulación.

# Importación de módulos: 

from scipy import interpolate 
import base64
import pylab as py
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.io.wavfile import read,write
from numpy import linspace,std
from scipy.fftpack import fftfreq
from scipy.signal import butter, filtfilt

'''
#Entradas: señal y tasa de muestreo del audio.
#Salidas: Frecuencias de muestreo, y la transformada de fourier normaliza.
#Función: calcular la transforma de la señal de audio.
'''
def calcFFT(tasaDeMuestreo,senal):
    #Se calcula la transformada
    fft = scipy.fft(senal)
    #Normalizar transformada dividiendola por el largo de la señal
    fftNormalized = fft/len(senal)
    #Genera las frecuencias de muestreo de acuerdo al largo de fftNormalize y el inverso de la tasa de muetreo
    xfft = np.fft.fftfreq(len(fftNormalized), 1 / tasaDeMuestreo)
    return xfft, fftNormalized

#Graficar la Transformada de fourier
'''
#Entradas: la transforma de fourier y la frecuencia de muestreo.
#Salida: grafica de latransformada de fourier.
#Función: generar el grafico de la transformada de fourier del audio original.
'''
def graficarTransformadaDeFourier(xft,ft,porcentaje,color):
    
    plt.xlabel("Frequencia [Hz]")
    plt.ylabel("Amplitud ")
    plt.plot(xft,abs(ft),color,label='señal modulada al '+str(porcentaje))
    plt.legend()


"""
#Funcion:   Función que realiza la modulación AM a una señal. Esto se hace 
            implementando una señal portada la cual se multiplica con la 
            señal original.

#Entradas:  la tasa de muestreo de la señal original, la señal interpolada
            el porcentaje con el que se quiere hacer la modulación y la 
            frecuencia con la que se hara la señal portadora.

#Salidas:   Esta funcion retorna tando la señal modulada y la señal la portadora
"""
def modulation_am(muestreo,senal_interpolada,porcentaje,frecuencia_portadora):

    largo_AM = len(senal_interpolada)

    tiempo_AM = np.linspace(0, largo_AM/muestreo, num = largo_AM)

    portadora = np.cos(2*np.pi* frecuencia_portadora * tiempo_AM)

    modulacionAM = porcentaje * senal_interpolada * portadora

    return modulacionAM,portadora


"""
#Funcion:   Función que realiza la modulación FM a una señal. Esto se hace 
            implementando una señal portada la cual se multiplica con la 
            señal original.

#Entradas:  la tasa de muestreo de la señal original, la señal interpolada
            el porcentaje con el que se quiere hacer la modulación y la 
            frecuencia con la que se hara la señal portadora.

#Salidas:   Esta funcion retorna tando la señal modulada y la señal la portadora
"""
def modulation_fm(muestreo,senal_interpolada,porcentaje,frecuencia_portadora):

    largo_FM = len(senal_interpolada)

    tiempo_FM = np.linspace(0, largo_FM/muestreo, num = largo_FM)

    sumatoria = porcentaje * np.cumsum(senal_interpolada)/muestreo

    aux = 2*np.pi*frecuencia_portadora*tiempo_FM

    z = np.cos(aux + sumatoria)

    portadora = np.cos(aux)

    return z,portadora

"""
#Funcion:   Que obtine una cantidad igual de puntos a partir de una función dada
            La utilidad de esta funcion es dejar a todas las funciones del mismo
            tamaño

#Entradas:  la señal original con su frecuencia de muestreo

#Salidas:   La misma señal de entrada pero con mas puntos.
"""
def interpolacion(senal,frecuencia):

    Tiempo = np.linspace(0, len(senal)/frecuencia, num=len(senal))

    interpolada = interpolate.interp1d(Tiempo, senal)

    Tiempo2 = np.linspace(0, len(senal)/frecuencia, len(senal))

    interpolada = interpolada(Tiempo2)

    return interpolada


"""
#Funcion:   Función que realiza la demodulación AM a una señal. Esto se hace 
            implementando un filtro paso bajo (que solo puedan bajar las bajas
            señales).

#Entradas:  La señal modulada en AM, la portadora de la señal y la tasa de muestreo.

#Salidas:   la señal filtrada. 
"""
def demodulacionAM(modulada,portadora,muestreo):

    demodulada = modulada * portadora

    b, a = butter(3,4000,'low',fs=muestreo)

    senal_recuperada = filtfilt(b, a,demodulada)

    return  senal_recuperada

"""
#Funcion:   Solo grafica la señal original del wav
"""
def grafico_senal_original(senal,frecuencia):
    plt.figure('señal original')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('f(t)')
    plt.title('Señal original')
    Tiempo = np.linspace(0, len(senal) / frecuencia, num=len(senal))
    plt.plot(Tiempo, senal)
    
"""
#Funcion:   Solo grafica la señal de la portadora (coseno)
"""
def grafico_portadora(portadora,frecuencia):
    plt.figure("Portadora")
    plt.xlabel('Tiempo (s)')
    plt.ylabel('f(t)')
    plt.title('Señal portadora')
    Tiempo = np.linspace(0, len(portadora) / frecuencia, num=len(portadora))
    plt.plot(Tiempo, portadora)
    plt.xlim(0,1)

"""
#Funcion:   añade un grafico de demodulación a una figura existente. 
"""
def grafico_demodulacion(senal, frecuencia,color,porcentaje):
    plt.xlabel('Tiempo (s)')
    plt.ylabel('f(t)')
    Tiempo = np.linspace(0, len(senal) / frecuencia, num=len(senal))
    plt.plot(Tiempo,senal,color,label="Señal demodulada de "+str(porcentaje))
    plt.legend()

"""
#Funcion:   añade un grafico de modulación a una figura existente. 
"""
def Grafico(senal,frecuencia,audio,porcentaje,color):
    plt.xlabel('Tiempo (s)')
    plt.ylabel('f(t)')
    Tiempo = np.linspace(0, len(audio) / frecuencia, num=len(audio))
    plt.plot(Tiempo, senal , color, label = 'señal modulada al '+str(porcentaje))
    plt.legend()



#####################
#       Main        #
#####################


tasa_muestreo, datos = wavfile.read("handel.wav")

grafico_senal_original(datos,tasa_muestreo)

frecuencia_portadora = 2*tasa_muestreo 

####### Parte Modulacion AM ##########


senal_interpolada = interpolacion(datos,tasa_muestreo)

porcentajes = [15,100,125]
colores = ['g','r','b']
i=1
print("obteniendo señal modulada AM...")
plt.figure("Modulacion AM")
plt.title('Modulación AM de la función al 15%,100%,150%')
for porcentaje in porcentajes:
    senal_modulada_am,senal_portadora = modulation_am(tasa_muestreo,senal_interpolada,porcentaje,frecuencia_portadora)
    plt.subplot(3,1,i)
    Grafico(senal_modulada_am,tasa_muestreo,senal_interpolada,porcentaje,colores[i-1])
    i=i+1

i=1
print("obteniendo transformada de fourier AM ...")
plt.figure("Transformada de Fourier AM")
plt.title("Transformada de fourier de las señales moduladas AM 15 100 y 125")

print("imprimiendo figuras...")
for porcentaje in porcentajes:
    senal_modulada_am,senal_portadora = modulation_am(tasa_muestreo,senal_interpolada,porcentaje,frecuencia_portadora)
    a,b = calcFFT(tasa_muestreo,senal_modulada_am)
    plt.subplot(3,1,i)
    graficarTransformadaDeFourier(a,b,porcentaje,colores[i-1])
    i=i+1


grafico_portadora(senal_portadora,tasa_muestreo)

plt.show()

####### Parte Modulacion FM ##########


i=1
print("obteniendo señal modulada FM...")
plt.figure("Modulacion FM")
plt.title('Modulación FM de la función al 15%,100%,150% ')
for porcentaje in porcentajes:
    senal_modulada_fm,senal_portadora = modulation_fm(tasa_muestreo,senal_interpolada,porcentaje,frecuencia_portadora)
    plt.subplot(3,1,i)
    Grafico(senal_modulada_fm,tasa_muestreo,senal_interpolada,porcentaje,colores[i-1])
    i=i+1

i=1
print("obteniendo transformada de fourier FM ...")
plt.figure("Transformada de Fourier FM")
plt.title("Transformada de fourier de las señales moduladas FM 15 100 y 125")

print("imprimiendo figuras...")
for porcentaje in porcentajes:
    senal_modulada_fm,senal_portadora = modulation_fm(tasa_muestreo,senal_interpolada,porcentaje,frecuencia_portadora)
    a,b = calcFFT(tasa_muestreo,senal_modulada_fm)
    plt.subplot(3,1,i)
    graficarTransformadaDeFourier(a,b,porcentaje,colores[i-1])
    i=i+1

plt.show()


############### Demodulación AM ###############

i=1
print("obteniendo demodulación AM")
plt.figure("Demodulacion AM")
plt.title("Demodulación AM con 15 100 y 125")

print("imprimiendo figuras...")
for porcentaje in porcentajes:
    senal_modulada_am,senal_portadora = modulation_am(tasa_muestreo,senal_interpolada,porcentaje,frecuencia_portadora)
    recuperada = demodulacionAM(senal_modulada_am,senal_portadora,tasa_muestreo)
    write('demodulacion_'+str(porcentaje)+'.wav',tasa_muestreo,recuperada)
    plt.subplot(3,1,i)
    grafico_demodulacion(recuperada,tasa_muestreo,colores[i-1],porcentaje)
    i=i+1

plt.show()


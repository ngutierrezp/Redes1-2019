#Autores: Nicolas gutierrez- Manuel López
#Version: 3.0
#Fecha de entrega: 19-04-2019.
#Objetivo: Obtener la transformada de Fourier de un audio.

#Bloque de importaciones
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.io.wavfile as sci_wav
from scipy.fftpack import fft, fftfreq

#Bloque de Funciones

#CALCULO DE LA TRANSFORMADA DE FOURIER
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
def graficarTransformadaDeFourier(xft,ft):
    plt.title("Transformada de fourier")
    plt.xlabel("Frequencia [Hz]")
    plt.ylabel("Amplitud ")
    plt.plot(xft,abs(ft))

#Graficar la Transformada de fourier truncada
'''
#Entradas: la transforma de fourier truncada y la frecuencia de muestreo.
#Salida: grafica de la transformada de fourier truncada.
#Función: generar el grafico de la transformada de fourier truncada.
'''
def graficarTransformadaDeFourierTruncada(xft,ft,sensibilidad):
    plt.title("Transformada de fourier truncada"+str(sensibilidad))
    plt.xlabel("Frequencia [Hz]")
    plt.ylabel("Amplitud ")
    plt.plot(xft,abs(ft))

#CÁLCULO DE LA TRANSFORMADA INVERSA FOURIER
'''
#Entrada: tranformada de fourier del audio original, largo de la frecuencias de muestreo.
#Salida: la tranformada inversa.
#Función: calcular la antitranformada de fourier.
'''
def transformadaInversa(ft):
    #Se calcula la transformada inversa
    fourierTInv = scipy.ifft(ft)
    fourierTInv=fourierTInv.real
    return fourierTInv

#Obtener la señal tiempo
'''
#Entradas: tasa de muestreo y señal del audio.
#Salida: el vector del tiempo.
#Funcion: generar el vector del tiempo del audio original.
'''
def obtenerSenalDeTiempo(tasaDeMuestreo,senal):
    
    senal_len = float(len(senal))
    tAux = float(senal_len)/float(tasaDeMuestreo)
    t = np.linspace(0, tAux, senal_len)
    return t

#Graficar señal del tiempo
'''
#Entradas: Señal del audio Original u el vector del tiempo.
#Salida: grafica de la amplitud versus tiempo.
#Función: generar la grafica de la amplitud versus tiempo.
'''
def graficarSenalDelTiempo(senal,t):
    plt.plot(t,senal)
    plt.title("Amplitud vs tiempo")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud ")

#Graficar señal del tiempo (transformada inversa)
'''
#Entradas: señal de la transformada inversa de fourier y el vector del tiempo.
#Salida: grafica de la amplitud versus tiempo.
#Función: generar la grafica de la señal de la transformada inversa de fourier versus tiempo.
'''
def graficarSenalDelTiempoI(senal,t):
    plt.plot(t,senal)
    plt.title("Amplitud vs tiempo (Transformada Inversa)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud ")

#Graficar señal del tiempo (transformada Truncada)
'''
#Entradas: señal de la transformada inversa truncada de fourier y el vector del tiempo.
#Salida: grafica de la amplitud versus tiempo.
#Función: generar la grafica de la señal de la transformada inversa de fourier versus tiempo.
'''
def graficarSenalDelTiempoT(senal,t,sensibilidad):
    plt.plot(t,senal)
    plt.title("Amplitud vs tiempo (Transformada Truncada)"+str(sensibilidad))
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud ")

#Truncar Tranformada de Fourier
'''
#Entradas: transformada de fourier y limite de amplitud.
#Salida: transformada de fourier truncada.
#Funcion: truncar la transformada de fourier ultilizando un limite de amplitud.
'''
def truncarTransformadaDeFourier(senal, limiteDeAmplitud):
    senalAux=senal
    for x in senal:
        if x>0 and x<limiteDeAmplitud:
            index=np.where(senal==x)
            senalAux[index]=0
        elif x<0 and x>-limiteDeAmplitud:
            index=np.where(senal==x)
            senalAux[index]=0

    return senalAux
            

'''
#Entada: vacio.
#Salida: tasa de muestreo y la señal del audio original
#Funcion: leer el audio y obtener la tasa de muestreo y la señal del audio.
'''
#Leer el audio
def leerElAudio():
    file_wav=sci_wav.read("handel.wav")
    tasaDeMuestreo = file_wav[0]
    senal=file_wav[1]
    return tasaDeMuestreo,senal                     

#mostras todas las graficas
'''
#Entradas: tasa de muestreo y señal del audio original.
#Salida: 5 graficas:
#Funcion: Entregar las diferentes graficas solicitadas.
'''
def graficas(tasaDeMuestreo,senal,sensibilidad):
    #obtener la grafica audio versus tiempo
    plt.figure(1)
    graficarSenalDelTiempo(senal,obtenerSenalDeTiempo(tasaDeMuestreo,senal))

    #Obtener la Transformada de fourier
    xfft,fftN=calcFFT(tasaDeMuestreo,senal)
    # Graficar La transformada del audio Original
    plt.figure(2)
    graficarTransformadaDeFourier(xfft,fftN)


    # obtener la inversa
    inversaT=transformadaInversa(fftN)
    #graficar la inversa
    plt.figure(3)
    graficarSenalDelTiempoI(inversaT,obtenerSenalDeTiempo(tasaDeMuestreo,inversaT))

    #obtener la transformada Truncada de fourier
    transformadaT=truncarTransformadaDeFourier(fftN,sensibilidad)
    #graficar la tranformada de fourier truncada

    plt.figure(4)
    graficarTransformadaDeFourierTruncada(xfft,transformadaT,sensibilidad)

    #obtener la transformada invensa de la tranformada truncada de fourier
    inversaTruncada=transformadaInversa(transformadaT)
    #grafica de la inversa Truncada
    plt.figure(5)
    
    graficarSenalDelTiempoT(inversaTruncada,obtenerSenalDeTiempo(tasaDeMuestreo,inversaTruncada),sensibilidad)

    sci_wav.write("HandelTruncado"+str(sensibilidad)+".wav",tasaDeMuestreo,inversaTruncada)
    
    plt.show()


#Solicar una amplitud para truncar la transformada.
def solicitarAmplitud():
    try:
        amplitud=int(input("Ingrese una aplitud para poder truncar la transformada de Fourier (recomendable entre 0 a 300): "))
        return amplitud
    except:
        print ("Error no es un numero")
        return solicitarAmplitud()


##########################
####Bloque principal#####
##########################

tasaDeMuestreo,senal=leerElAudio()
amplitud=solicitarAmplitud()
graficas(tasaDeMuestreo,senal,amplitud)



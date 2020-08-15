#Autores: Nicolas gutierrez
#Version: 3.0
#Fecha de entrega: 04.05.2019
#Objetivo: Aplicar filtros para una señal de audio

#Bloque de importaciones
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.io.wavfile as sci_wav
from scipy.fftpack import fft, fftfreq
from scipy import signal


#Bloque de Funciones

#CALCULO DE LA TRANSFORMADA DE FOURIER


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



'''
#Entradas: tranformada de fourier del audio original, largo de la frecuencias de muestreo,
        orden del filtro y los limites de corte del filtro (inferior y superior)
#Salida: grafica de latransformada de fourier con los parametros dados
#Función: generar el grafico de la transformada de fourier del audio original con la señal filtrada.
'''
def graficarTransformadaDeFourierFiltrada(xft,ft,orden,inferior,superior):
    if inferior == None:
        inferior = ""
    if superior == None:
        superior = ""
    inferior=str(inferior)
    superior=str(superior)
    plt.title("Transformada de fourier con orden de : "+str(orden)+" y con limites: ["+inferior+","+superior+"]")
    plt.xlabel("Frequencia [Hz]")
    plt.ylabel("Amplitud ")
    plt.plot(xft,abs(ft))

#CÁLCULO DEL FILTRO
'''
#Entrada: orden = grado de orden con el que opera el filtro ( a mayor orden mayor rectitud en el corte)
          inferior = limite inferior en Frecuencia
          Superior = limite superior en Frecuencia
          muestreo = Tasa de muestreo (se obtiene del archivo.wav)
          senal = señal de audio obtenida del archivo.wav
#Salida: senal filtrada
#Función: dependiendo de los pararametros de entrada, aplica un filtro (Bajo,Alto o Banda)
'''

def butter_filter(orden,inferior,superior,muestreo,senal):

    b=0
    a=0
    # Dependiendo de los datos que sean entregados es como se va a comportar la función.
    # Si el limite inferior no es entregado, se asumira que se requiere un paso Alto
    if inferior == None:
        b, a = signal.butter(3, superior, 'low',fs=muestreo)
    
    # Si el limite superior no es entregado, se asumira que se requiere un paso Bajo
    elif superior == None:
        b, a = signal.butter(3, inferior, 'high' , fs=muestreo)
    elif superior ==None and inferior == None:
        print("no hay limites para ejecutar")
        return -1
    else:
        b, a = signal.butter(3, [inferior,superior], 'band' , fs=muestreo)
    
    #senal_filtrada = signal.filtfilt(b,a,senal)
    senal_filtrada = signal.lfilter(b,a,senal)
    return senal_filtrada

def obtenerFiltros(orden,inferior,superior,muestreo,senal):

    # Filtro Paso Bajo
    Bajo= butter_filter(orden,None,superior,muestreo,senal)
    

    # Filtro Paso Alto

    Alto = butter_filter(orden,inferior,None,muestreo,senal)

    # Filtro Pasa Banda

    Banda = butter_filter(orden,inferior,superior,muestreo,senal)

    return Bajo,Alto,Banda

def espectograma(senal, tasa,titulo):
    freq, tim, data = signal.spectrogram(senal, tasa)
    plt.pcolormesh(tim, freq, np.log(data))
    plt.colorbar()
    plt.title(titulo)
    plt.xlabel("Tiempo [S]")
    plt.ylabel("Frecuencia [Hz]")


########################## BLOQUE PRINCIPAL ################################


# Definiciones de constantes globales
orden = 8
inf=1500
supe=2000
tasa,senal = leerElAudio()
bajo,alto,banda = obtenerFiltros(orden,inf,supe,tasa,senal)


###### Señal original
data, datanorm= calcFFT(tasa,senal)

plt.figure(1)
graficarTransformadaDeFourier(data,datanorm)
plt.figure(2)
espectograma(senal,tasa,"Espectograma de Handel original")

###### Aplicacion y muestreo para un filtro de Paso Bajo
data_baja_filt, dataNomr_bajo_filt = calcFFT(tasa,bajo)
plt.figure(3)
graficarTransformadaDeFourierFiltrada(data_baja_filt,dataNomr_bajo_filt,orden,None,supe)
BajoI = transformadaInversa(dataNomr_bajo_filt)
sci_wav.write("Test_bajo.wav",tasa,BajoI)
plt.figure(4)
espectograma(BajoI,tasa,"Espectograma de Handel con un filtro Paso Bajo")

###### Aplicacion y muestreo para un filtro de Paso Alto
data_Alta_filt, dataNomr_Alto_filt = calcFFT(tasa,alto)
plt.figure(5)
graficarTransformadaDeFourierFiltrada(data_Alta_filt,dataNomr_Alto_filt,orden,inf,None)
altoI = transformadaInversa(dataNomr_Alto_filt)
sci_wav.write("Test_alto.wav",tasa,altoI)
plt.figure(6)
espectograma(altoI,tasa,"Espectograma de Handel con un filtro Paso Alto")


###### Aplicacion y muestreo para un filtro de Paso Banda
data_Banda_filt, dataNomr_banda_filt = calcFFT(tasa,banda)
plt.figure(7)
graficarTransformadaDeFourierFiltrada(data_Banda_filt,dataNomr_banda_filt,orden,inf,supe)
bandaI= transformadaInversa(data_Banda_filt)
sci_wav.write("Test_banda.wav",tasa,bandaI)
plt.figure(8)
espectograma(bandaI,tasa,"Espectograma de Handel con un filtro Pasa Banda")

plt.show()



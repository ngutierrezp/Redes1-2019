from random import randint, uniform,random
from matplotlib import pyplot
import numpy as np

def randomBits(length):
	'''
	Entrada: cantidad de numeros de elementos a generar
	Funcion: se genera un arreglo de números 0 y 1
	Salida: Arreglo de bits
	'''
	bits = []
	while len(bits) < length:
		bits.append(randint(0,1))
	return bits

def Plotter(signal,modulation,demodulation):
	'''
	Entradas: Señal de bits, modulación ook y demodulación ook
	Descripción: Se crean tres gráficos de amplitud vs tiempo
	Salida: Gráfico con los tres valores de entrada
	'''
	pyplot.figure("Señal de bits original")
	pyplot.title('Señal Digital original')
	pyplot.ylabel('Amplitud')
	xaxis = np.arange(0, len(signal))
	yaxis = np.array(signal)
	pyplot.step(xaxis, yaxis)
	pyplot.ylim(-0.5,1.5)
	pyplot.xlabel('Tiempo')

	pyplot.figure("Señal modulada en OOK")
	pyplot.plot(modulation)
	pyplot.title('Señal modulada ook')
	pyplot.xlabel('Tiempo')
	pyplot.ylabel('Amplitud')


	pyplot.figure("Señal demodulada en OOK")
	pyplot.title('Señal demodulada ook')
	pyplot.xlabel('Tiempo')
	pyplot.ylabel('Amplitud')
	pyplot.ylim(-0.5,1.5)
	xaxis_d = np.arange(0, len(demodulation))
	yaxis_d = np.array(demodulation)
	pyplot.step(xaxis_d,yaxis_d)

	pyplot.show()


def OOKmodulation(signal, bitRate):
	'''
	Entrada:  Lista de bits que se necesita enviar -> Modular y una tasa de bits para crear la funcion portadora.
	Funcion: Crea una señal modulada con 2 funciones portadoras con diferentes amplitudes:
        La amplitud 0 correspondera al 0
        La amplitud 15 correspondera al 1
		La idea es que una señal sea distinta de 0
	Salida: Señal modulada en ook
	'''
	ook = []
	time = np.linspace(0,1,2*bitRate)
	port_0 = 0*np.cos(2*np.pi*time) # arreglo de 0´s
	port_1 = 15*np.cos(2*np.pi*time)

	for i in signal:
		if i == 0:
			for j in port_0:
				ook.append(j)
		elif i == 1:
			for j in port_1:
				ook.append(j)
	return ook


def OOKDemodulatio(signal, bitRate):

	'''
	Entradas: señal modulada y la tasa de bits por segundo

	Funcion: En el arreglo de entrada, se recorre y por una tasa de bits cuando hay una señal con amplitud distinta de 0
		se fija un 1, cuando en la tasa de bits recorrida se suma 0, se fija un 0
	Salida: Señal digital demodulada
	'''
	newSignal = []
	rate = bitRate*2
	local=0
	i=0
	while  i < len(signal):
		j = 0
		local = 0
		while j < rate:
			local = local+signal[i]
			j = j + 1
			i = i + 1
		if local !=  0:
			newSignal.append(1)
		else:
			newSignal.append(0)
	return newSignal


    #######################################
    #            Main
    #######################################

bits = randomBits(100)

rate = 10 # 10 bits por segundo

modulation = OOKmodulation(bits,rate)

demo = OOKDemodulatio(modulation,rate)

Plotter(bits,modulation,demo)
# Version: 3.6
# Fecha : 08-05-2019
# Nicolas Gutierrez

from PIL import Image # Modulo para trabajar con imagenes
import numpy as np
import scipy
import matplotlib.pyplot as plt



def getImage(dir_image):
    """
        Funcion que obtiene la imagen y la transforma en un arreglo
        de numpy -> Matriz, el cual contiene los pixeles de la imagen.
        Cabe destacar que la matriz solo posee los pixeles de la imagen
        en escala de grises.

        @param  --  dir_image Corresponde a un string de la direccion de la imagen 
        con su extensión.

        @return --  La funcion tiene 2 retornos:
            array   -> Corresponde a la matriz de pixeles de la imagen.
            img     -> Corresponde a la imagen abierta.
    """
    img = Image.open(dir_image)
    array = np.array(img)
    return array,img


def rectifyMatrix(Matrix_A, Kernel_row, Kernel_col):
    """
        Funcion que ajusta la matriz obtenida de la convolución entre la imagen
        y el kernel. Producto de la comvolución se pierde una diferencia de dimenciones
        igual a las dimensiones del kernel - 1  (para ambos extremos).
        Lo que hace la funcion es añadir a la imagen convolucionada, las filas y columnas 
        perdidas. Se le agregan 0 en todas estas partes.

        @param  --  Matrix_A corresponde a la matriz de convolución.
        
        @param  --  Kernel_row corresponde a la dimensión de las filas del kernel.
        
        @param  --  Kernel_col corresponde a la dimensión de las columnas del kernel.

        @return --  Entrega una matriz de igual dimension que la original pero en este caso
        la matriz que devuelve tendra un borde negro ya que son las filas y columnas agregas
        con valores 0.

    """

    # Se agregan filas que se perdieron con la convolucion entre el kernel
    # la matriz    
    
    i_row = int((Kernel_row - 1) / 2)
    
    j_col = int((Kernel_col - 1) / 2)
    
    row = np.zeros((Matrix_A.shape[0]),dtype=int)

    for i in range(0,i_row):
        Matrix_A = np.insert(Matrix_A,0,row,0)
        Matrix_A = np.insert(Matrix_A,Matrix_A.shape[0],row,0)

    col = np.zeros((Matrix_A.shape[0]),dtype=int)

    for j in range(0,j_col):
        Matrix_A = np.insert(Matrix_A,0,col,1)
        Matrix_A = np.insert(Matrix_A,Matrix_A.shape[1],col,1)

    return Matrix_A
        





def convolve(matrix_K, matrix_A):
    """
        Función que realiza la convolución entre dos matrices.
        Lo que se hace es hace son 4 iteraciones, las dos primeras
        corresponden en a las iteraciones de filas y columnas de la
        matriz mas grande, las otras dos iteraciones iteran filas
        y columnas la matriz mas pequeña o el Kernel a implementar

        @param --  matrix_K    Corresponde a la matriz pequeña o kernel

        @param --  matrix_A    Corresponde a la matriz grande a la cual
        se le aplica el kernel

        @return -- Una matriz mas pequeña correspondiente a la
        convolucion de las dos matricez, la matriz resultante
        tendra las dimenciones de la resta de las dimenciones
        de las matrices A - K.
    """
    
    # Asumiendo que la matriz A es mucho mas grande que la K
    # se define la matriz K como la matriz de kernel.

    kernel_row, kernel_col = matrix_K.shape
    matrix_row, matrix_col = matrix_A.shape

    result_dimensions = (matrix_row-kernel_row,matrix_col-kernel_col)
    x= np.ones(result_dimensions)
    for i in range(0, matrix_row-kernel_row):
        for j in range(0,matrix_col-kernel_col):
            sum = 0
            for k in range(0, kernel_row):
                for p in range(0, kernel_col):
                    sum = sum + matrix_K[k,p]*matrix_A[i+k,j+p]
            x[i-k,j-p] = sum
            
    return x


def FourierTransform(img,figure,title):
    """
        Funcion que calcula la transformada de Fourier.
        Código obtenido de @DennisDaMennis en StackOverFlow
            https://stackoverflow.com/questions/41147483/numpy-function-fft-fft2-giving-an-error-cannot-do-a-non-empty-take-from-an

        
        Como se lleva la imagen al dominio de las frecuencias, la escala de los valores obtenidos
        con fftshift(), escapa del nivel para ser muestreados, por lo que es necesario aplicar
        una pequeña corrección a estos valores sometiendolos a un logaritmo.

        @param img  --  es la matriz de imagen que resulta al abrir la imagen, debe contener los 
        valores en escala de grises

        
        @param figure   --  corresponde a un número que representa la figura la a la cual se le aplica el grafico

        @param title    --  corresponde a un str el cual será mostrado en el titulo del grafico.

        @return --  La funcion no retorna nada, pero crea un grafico el cual será mostrado con el 
        llamado a la funcion plt.show()
    """
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift)) # Discretización de valores. 

    plt.figure(figure),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Transformada de Fourier 2D (imagen: '+title+")")
    plt.colorbar()


#############################################################
#                       Bloque Principal                    #
#############################################################

## Definicion de kernel para filtro

const = 1/256

Matrix_kernel1 =   [[1, 4, 6, 4,1],
                    [4,16,24,16,4],
                    [6,24,36,24,6],
                    [4,16,24,16,4],
                    [1, 4, 6, 4,1]]

Matrix_kernel2 =   [[1,2,0,-2,-1],
                    [1,2,0,-2,-1],
                    [1,2,0,-2,-1],
                    [1,2,0,-2,-1],
                    [1,2,0,-2,-1]]

Kernel_suavizado= np.array(Matrix_kernel1)
Kernel_bordes = np.array(Matrix_kernel2)

M_image,img_ori = getImage("original.bmp") # obtención de la imgen

FourierTransform(img_ori,1,"original") # Transformada de fourier de la imagen original

#---------------------------------------------------------------------------------------
#Calculos para Kernel suavizado

Kernel_suavizado = const* Kernel_suavizado

M_image_suave = convolve(Kernel_suavizado,M_image)

k_row,k_col = Kernel_suavizado.shape

M_image_rec = rectifyMatrix(M_image_suave,k_row,k_col)

out_img = Image.fromarray(M_image_rec.astype('uint8'))
FourierTransform(out_img,2,"suavizada")

out_img.save("suavizado.bmp")


# #---------------------------------------------------------------------------------------
# #Calculos para Kernel de deteccion de bordes 

saturacion = const

Kernel_bordes = saturacion * Kernel_bordes

M_image_borde = convolve(Kernel_bordes,M_image)

k_row,k_col = Kernel_bordes.shape

M_image_rec = rectifyMatrix(M_image_borde,k_row,k_col)

out_img = Image.fromarray(M_image_rec.astype('uint8'))
FourierTransform(out_img,3,"detección de Bordes")

out_img.save("borde.bmp")

############

plt.show()
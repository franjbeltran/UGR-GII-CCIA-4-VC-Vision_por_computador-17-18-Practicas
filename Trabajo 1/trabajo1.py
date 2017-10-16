# -*- coding: utf-8 -*-
"""

Francisco Javier Caracuel Beltrán

VC - Visión por Computador

4º - GII - CCIA - ETSIIT - UGR

Curso 2017/2018

"""

import cv2
import numpy as np
import math
import copy
from matplotlib import pyplot as plt

###############################################################################
# Configuración general
#

# Ruta hacia las imágenes
path = "imagenes/"

# Título para las imágenes
img_title = ""

# Número de columnas que tendrá la tabla con la salida de las imágenes
num_cols = 3

# Esquema de color que se utiliza por defecto para convertir las imágenes
cmap = cv2.COLOR_RGB2GRAY

# Esquema de color que se utiliza por defecto en plt.imshow()
plt.rcParams['image.cmap'] = 'gray'

#
################################################################################

################################################################################
# Utils
#


def set_c_map(imgs, cmap = cv2.COLOR_RGB2BGR):
    """

    Asigna un esquema de color a todas las imágenes que se reciben.

    Si se recibe una lista, las imágenes verán reflejadas su nuevo esquema de
    color directamente.
    Si se recibe una imagen, es necesario hacer una asignación con el resultado
    de aplicar esta función.

    ------------
    Para aplicar el color se puede utilizar:
    - cv2.COLOR_RGB2BGR (color)
    - cv2.COLOR_RGB2GRAY (escala de grises)
    - ...

    """

    # Se comprueba si el elemento que se recibe es una imagen o es una lista
    # con imágenes.
    if isinstance(imgs, list):

        # Se guarda la longitud de la lista de imágenes
        length = len(imgs)

        # Es una lista de imágenes, por lo que se recorre cada una para cambiar
        # el esquema de color
        for i in range(length):
            imgs[i] = cv2.cvtColor(imgs[i], cmap)


    else:

        # Si es una imagen se le aplica el esquema de color
        imgs = cv2.cvtColor(imgs, cmap)

    return imgs

def power_two(n):
    """
    Calcula el logaritmo en base 2 de un número, truncando los decimales
    """
    return int(math.log(n, 2))

def next_power_two(n):
    """
    Calcula el siguiente número potencia de 2 de otro número recibido por
    parámetro.
    """
    return pow(2, power_two(n)+1)

#
################################################################################

################################################################################
# Apartado 1
#

########
# Sección A
#


def show_images(imgs, names = list(), cols = 3, title = ""):
    """

    Dada una lista de imágenes (imgs) y una lista de nombres (names), muestra en
    una tabla todas estas imágenes con su nombre correspondiente.
    Por defecto, el número de columnas que se van a mostrar es 3.
    Por defecto, el título que acompaña a cada imagen es "".

    """

    # Se guarda la cantidad de imágenes que se van a mostrar
    imgs_length = len(imgs)

    # Si la lista está vacía, se crea una con el tamaño de imágenes y rellena de
    # espacios en blanco para que no haya ningún error al mostrarlo
    if not names:
        names = [""]*imgs_length

    # Si hay menos imágenes que número de columnas se ha establecido, se
    # disminuye el número de columnas al número de imágenes y el número de filas
    # a 1.
    if imgs_length <= cols:

        cols = imgs_length
        rows = 1

    # Si hay más imágenes, el número de filas viene determinado por el número de
    # imágenes. Se debe redondear siempre al alza para poder colocar las
    # imágenes en la última fila
    else:

        rows = math.ceil(imgs_length/cols)

    # Se recorren todas las imágenes para colocarlas una a una en la posición
    # que les corresponde en la tabla
    for i, img in enumerate(imgs):

        # Se indica el número de filas y columnas y la posición que ocupa la
        # imagen actual en esa tabla
        plt.subplot(rows, cols, i+1)

        # Se cambia el esquema de color de la imagen
        #img = cv2.cvtColor(img, cmap)

        # Se indica el título que tendrá la imagen
        plt.title(title+names[i])

        # Se indica el valor que se mostrará en el eje X
        #plt.xlabel(i)

        # Se eliminan las marcas de los ejes X e Y
        plt.xticks([])
        plt.yticks([])

        # Se muestra la imagen en su lugar correspondiente
        plt.imshow(img)

    # Se visualiza la tabla con todas las imágenes
    plt.show()

#
########

########
# Sección B
#


def convolution_b(img, sigma, mask = -1, border = -1):
    """

    Dada una imagen (img), realiza una convolución con máscara gaussiana. El
    tamaño de la máscara por defecto se calcula a partir del sigma recibido,
    como 6*sigma+1. Esto es debido a que la medida más óptima es crear un
    intervalo de 3*sigma por cada lado, lo que hace que se tenga 6*sigma. Para
    no dejar sin contar el punto intermedio se suma 1, lo que termina resultando
    6*sigma+1.
    Se permite especificar una máscara determinada.
    Se puede indicar un borde, que por defecto está deshabilitado.

    -------------------
    Opciones para el borde:
    - cv2.BORDER_REPLICATE
    - cv2.BORDER_REFLECT
    - cv2.BORDER_REFLECT_101
    - cv2.BORDER_WRAP
    - cv2.BORDER_CONSTANT
    -------------------

    Devuelve la imagen con la transformación realizada.

    """

    # Opciones para el borde:
    # - BORDER_REPLICATE
    # - BORDER_REFLECT
    # - BORDER_REFLECT_101
    # - BORDER_WRAP
    # - BORDER_CONSTANT

    # Se comprueba si no se ha recibido la máscara. En el caso de no recibirla
    # se le asigna el valor por defecto 6*sigma+1
    if mask == -1:
        mask = 6*sigma+1

    # Si no se recibe ningún borde, la convolución será por defecto. En caso de
    # recibirlo, se le indica por parámetro el especificado
    if border == -1:
        img = cv2.GaussianBlur(img, (mask, mask), sigma)
    else:
        img = cv2.GaussianBlur(img, (mask, mask), sigma, borderType = border)

    # Se devuelve la imagen con la convolución realizada
    return img

#
########

########
# Sección C
#

def convolution_c(img , kernel = None, kernel_y = None, sigma = 0, border = cv2.BORDER_DEFAULT, x = 1, y = 1, normalize = True):
    """

    Dada una imagen (img) y un núcleo (kernel) realiza una convolución de la
    imagen utilizando dicho núcleo.
    Se aplicará sucesivamente el núcleo por todas las filas de la imagen y con
    esta transformación se vuelve a aplicar a todas las columnas.
    Si se recibe un kernel "None", se extrae el kernel a través de la
    función cv2.getGaussianKernel().
    Si se recibe un kernel_y "None", será el mismo kernel ya calculado.
    Se puede indicar un borde, que por defecto está deshabilitado.
    x indica si realiza la convolución por filas. Por defecto, sí la realiza.
    Si x = -1, no realiza la convolución por filas.
    y indica si realiza la convolución por columnas. Por defecto, sí la realiza.
    Si y = -1, no realiza la convolución por columnas.
    Permite la opción de normalizar, que por defecto está activa. Se le puede
    enviar el parámetro normalize = False para que no la haga.

    -------------------
    Opciones para el borde:
    - cv2.BORDER_REPLICATE
    - cv2.BORDER_REFLECT
    - cv2.BORDER_REFLECT_101
    - cv2.BORDER_WRAP
    - cv2.BORDER_CONSTANT
    -------------------

    Devuelve la imagen con la transformación realizada.

    """

    # Si se modifican los valores de img, se verán reflejados en el resto de
    # imágenes que hagan uso de la imagen base. Para evitar esto, se hace un
    # copiado en el que no es por referencia
    img = copy.deepcopy(img)

    # Se comprueba si se calcula el kernel
    if kernel is None:
        kernel = cv2.getGaussianKernel(6*sigma+1, sigma)

    # Se comprueba si se calcula el kernel_y
    if kernel_y is None:
        kernel_y = kernel

    # Se obtiene el número de filas y columnas que tiene la imagen. La función
    # shape devuelve también los canales siempre que la imagen no esté en
    # escala de grises, por lo que se comprueba cuantos valores devuelve la
    # función y en base a ello se guardan en dos o en tres variables (para
    # evitar errores de ejecución)
    shape = img.shape

    # Para no tener error en el ámbito de las filas y las columnas se declaran
    # en este punto
    # También se pueden obtener las filas como len(img) y las columnas como
    # len(img[0])
    rows = 0
    cols = 0

    # Si la imagen está en escala de grises devuelve dos valores
    if len(shape) == 2:
        rows, cols = shape

    # Si la imagen tiene un esquema de color devuelve tres valores
    elif len(shape) == 3:
        rows, cols, channels = shape

    # Es posible utilizar la función cv2.filter2D() sin tener en cuenta los
    # canales de la imagen, ya que esta función internamente se encarga de su
    # procesamiento.

    # Se comprueba si debe realizar la convolución por filas
    if x >= 0:

        # Se recorren las filas de la imagen haciendo la convolución
        for i in range(rows):
            # Se modifica la fila "i" con la convolución, indicando la fila "i",
            # el núcleo que se quiere utilizar y la profundidad por defecto de la
            # imagen.
            # La fila de una imagen es una lista (vector) con "cols" enteros. El
            # resultado de aplicar filter2D devuelve una lista de "cols" listas,
            # por lo que al hacer la asignación da error. Se debe guardar el
            # resultado de hacer convolución y más tarde transformar la lista de
            # listas en una lista de enteros.
            resConv = cv2.filter2D(img[i, :], -1, kernel, borderType=border)

            # Se guarda la lista de enteros creada en su fila correspondiente
            #img[i, :] = [val for sublist in resConv for val in sublist]
            img[i, :] = [sublist[0] for sublist in resConv]

    # Se comprueba si debe realizar la convolución por columnas
    if y >= 0:

        # Se recorren las columnas de la imagen haciendo la convolución
        for i in range(cols):
            # Se modifica la columna "i" con la convolución, indicando la columna
            # "i", el núcleo que se quiere utilizar y la profundidad por defecto de
            # la imagen
            # La columna de una imagen es una lista (vector) con "rows" enteros. El
            # resultado de aplicar filter2D devuelve una lista de "rows" listas,
            # por lo que al hacer la asignación da error. Se debe guardar el
            # resultado de hacer convolución y más tarde transformar la lista de
            # listas en una lista de enteros.
            resConv = cv2.filter2D(img[:, i], -1, kernel_y, borderType=border)

            # Se guarda la lista de enteros creada en su fila correspondiente
            #img[:, i] = [val for sublist in resConv for val in sublist]
            img[:, i] = [sublist[0] for sublist in resConv]

    # Se normalizan los resultados calculados
    if normalize:
        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

    # Se devuelve la imagen con la convolución realizada
    return img

#
########

########
# Sección D
#

def convolution_d(img , kernel = None, ksize = 3, sigma = 0, border = cv2.BORDER_DEFAULT, dx = 1, dy = 1):
    """

    Dada una imagen (img), realiza convolución con núcleo de primera derivada
    de tamaño ksize. Si se recibe un sigma, hace alisamiento con dicho sigma.
    Se puede indicar un borde, que por defecto está deshabilitado.

    -------------------
    Opciones para el borde:
    - cv2.BORDER_REPLICATE
    - cv2.BORDER_REFLECT
    - cv2.BORDER_REFLECT_101
    - cv2.BORDER_WRAP
    - cv2.BORDER_CONSTANT
    -------------------

    Devuelve dos imágenes con la convolución con respecto a la x y con respecto
    a la y.

    """

    # Si se modifican los valores de img, se verán reflejados en el resto de
    # imágenes que hagan uso de la imagen base. Para evitar esto, se hace un
    # copiado en el que no es por referencia
    img = copy.deepcopy(img)

    # Se crean los kernels con respecto a x y con respecto a y para aplicarlos
    # independientemente
    kernel_x = kernel
    kernel_y = kernel

    # Se comprueba si se calcula el kernel
    if kernel is None:

        # Se calcula el kernel de la primera derivada. Esto devuelve dos
        # kernels, uno con respecto a x y otro con respecto a y de tamaño ksize.
        # Se guarda cada uno donde corresponde.
        kernel = cv2.getDerivKernels(dx, dy, ksize)
        kernel_x = kernel[0]
        kernel_y = kernel[1]

        # Para aplicar convolución y no correlación se debe dar la vuelta al
        # kernel. Indicando -1 da la vuelta en el eje x e y, como solo tiene
        # uno, es correcto.
        #kernel_x = cv2.flip(kernel_x, -1)
        #kernel_y = cv2.flip(kernel_y, -1)

    # Antes de hacer la convolución, se alisa para evitar ruido.
    if sigma > 0:
        img = convolution_c(img, sigma=sigma)

    # Como se devuelven dos imágenes y se le aplica a una, convolución por filas
    # y a la otra por columnas, se crean dos imágenes nuevas
    img_x = copy.deepcopy(img)
    img_y = copy.deepcopy(img)

    # Se hace la convolución por filas. Se indica que no se haga por columnas
    img_x = convolution_c(img_x, kernel_x, y = -1, border = border)

    # Se hace la convolución por columnas. Se indica que no se haga por filas
    img_y = convolution_c(img_y, kernel_y, x = -1, border = border)

    # Se devuelve la imagen con la convolución realizada
    return img_x, img_y

#
########

########
# Sección E
#

def convolution_e(img , kernel = None, ksize = 3, sigma = 0, border = cv2.BORDER_DEFAULT, dx = 2, dy = 2):
    """

    Dada una imagen (img), realiza convolución con núcleo de segunda derivada
    de tamaño ksize. Si se recibe un sigma, hace alisamiento con dicho sigma.
    Se puede indicar un borde, que por defecto está deshabilitado.

    -------------------
    Opciones para el borde:
    - cv2.BORDER_REPLICATE
    - cv2.BORDER_REFLECT
    - cv2.BORDER_REFLECT_101
    - cv2.BORDER_WRAP
    - cv2.BORDER_CONSTANT
    -------------------

    Devuelve dos imágenes con la convolución con respecto a la x y con respecto
    a la y.

    """

    # La función convolution_d realiza esto mismo pero con la primera derivada.
    # Se aprovecha que ya se encuentra definida y se le envían los mismos
    # parámetros que recibe esta función, pero indicando que se realiza la
    # segunda derivada (dx, dy por defecto de esta función)
    return convolution_d(img, kernel, ksize, sigma, border, dx, dy)

#
########

########
# Sección F
#

def convolution_f(img , kernel = None, ksize = 3, sigma = 0, border = cv2.BORDER_DEFAULT, dx = 2, dy = 2):
    """

    Dada una imagen (img), realiza convolución con núcleo Laplaciana-Gaussiana
    de tamaño ksize. Si se recibe un sigma, hace alisamiento con dicho sigma.
    Se puede indicar un borde, que por defecto está deshabilitado.

    -------------------
    Opciones para el borde:
    - cv2.BORDER_REPLICATE
    - cv2.BORDER_REFLECT
    - cv2.BORDER_REFLECT_101
    - cv2.BORDER_WRAP
    - cv2.BORDER_CONSTANT
    -------------------

    Devuelve la imagen resultado de la convolución con el núcleo.

    """

    # Se comprueba si se calcula el kernel
    if kernel is None:

        # Se calcula el kernel de la primera derivada. Esto devuelve dos
        # kernels, uno con respecto a x y otro con respecto a y de tamaño ksize.
        # Se guarda cada uno donde corresponde.
        kernel = cv2.getDerivKernels(dx, dy, ksize)
        kernel_x = kernel[0]
        kernel_y = kernel[1]

        # Para aplicar convolución y no correlación se debe dar la vuelta al
        # kernel. Indicando -1 da la vuelta en el eje x e y, como solo tiene
        # uno, es correcto.
        #kernel_x = cv2.flip(kernel_x, -1)
        #kernel_y = cv2.flip(kernel_y, -1)

        # El resultado es la suma de la derivada segunda con respecto a x y con
        # respecto a y
        kernel = kernel_x + kernel_y

    # Se aplica el alisamiento si sigma > 0
    if sigma > 0:
        img = convolution_c(img, sigma = sigma)

    # Se convoluciona la imagen con la segunda derivada calculada anteriormente
    return convolution_c(img, kernel, border = border)

#
########

########
# Sección G
#

def show_pyr(imgs):
    """

    Función que muestra una serie de imágenes que recibe en una lista por
    parámetro en forma de pirámide.
    Como requisito para su correcto funcionamiento, las imágenes deben
    decrementar su tamaño en la mitad a medida que ocupan una posición posterior
    en la lista.

    Devuelve una sola imagen con forma de pirámide donde se encuentran todas las
    recibidas.

    """

    # Se crea una imagen inicialmente vacía que albergará todas las subimágenes
    # que se reciben.
    # El ancho de la imagen general será el ancho de la primera más el ancho
    # de la segunda (que será la mitad de la primera).

    # El ancho se calcula como len(img[0])+len(img[0])*0.5
    shape = imgs[0].shape

    height = shape[0]
    width = shape[1]

    # Se crea la imagen general con las medidas para que entren todas
    img = np.zeros((height, width+math.ceil(width*0.5)))

    # Se copia la primera imagen desde el punto de partida hasta el tamaño que
    # tiene
    img[0:height, 0:width] = imgs[0]

    # Se guarda la posición desde donde deben comenzar las imágenes
    init_col = width
    init_row = 0

    # Número de imágenes
    num_imgs = len(imgs)

    # Se recorren el resto de imágenes para colocarlas donde corresponde
    for i in range(1, num_imgs):

        # Se consigue el tamaño de la imagen actual
        shape = imgs[i].shape

        height = shape[0]
        width = shape[1]

        # Se hace el copiado de la imagen actual como se ha hecho con la primera
        img[init_row:init_row+height, init_col:init_col+width] = imgs[i]

        # Se aumenta el contador desde donde se colocará la siguiente imagen
        init_row += height

    return img

def generate_gaussian_pyr_imgs(img, n = 4, sigma = 0, border = cv2.BORDER_DEFAULT):
    """

    Función que, a partir de una imagen, genera n imágenes (por defecto, 4) de
    la mitad de tamaño cada vez.
    Para la generación de las imágenes se hace uso de la función cv2.pyrDown()
    que internamente aplica el alisado antes de reducir la imagen, por lo que no
    es necesario realizarlo antes.
    El valor de sigma indica el alisamiento que se realiza antes de generar
    todas las imágenes.
    Se puede indicar un borde, que por defecto está deshabilitado.

    -------------------
    Opciones para el borde:
    - cv2.BORDER_REPLICATE
    - cv2.BORDER_REFLECT
    - cv2.BORDER_REFLECT_101
    - cv2.BORDER_WRAP
    - cv2.BORDER_CONSTANT
    -------------------

    Devuelve una lista con n+1 imágenes, donde la primera es la original, la
    segunda es la mitad del tamaño de la primera, etc.

    """

    # Antes de generar las imágenes, se alisa para evitar ruido.
    if sigma > 0:
        img = convolution_c(img, sigma=sigma)

    # Se resta 1 al número de imagenes que se quiere que aparezcan
    n -= 1

    # Se crea la lista donde se alojan las imágenes
    imgs = list()

    # Se añade la imagen original a la lista
    imgs.append(img)

    # Se añaden tantas imágenes como se haya indicado
    for i in range(n):

        imgs.append(cv2.pyrDown(imgs[i], borderType=border))

    return imgs

#
########

########
# Sección H
#

def generate_laplacian_pyr_imgs(img, n = 4, sigma = 0, border = cv2.BORDER_DEFAULT):
    """

    Función que, a partir de una imagen, genera n imágenes (por defecto, 4) de
    la mitad de tamaño cada vez.
    Para la generación de las imágenes se hace uso de la función cv2.pyrDown()
    que internamente aplica el alisado antes de reducir la imagen, por lo que no
    es necesario realizarlo antes.
    El valor de sigma indica el alisamiento que se realiza antes de generar
    todas las imágenes.
    Se puede indicar un borde, que por defecto está deshabilitado.

    -------------------
    Opciones para el borde:
    - cv2.BORDER_REPLICATE
    - cv2.BORDER_REFLECT
    - cv2.BORDER_REFLECT_101
    - cv2.BORDER_WRAP
    - cv2.BORDER_CONSTANT
    -------------------

    Devuelve una lista con n+1 imágenes, donde la primera es la original, la
    segunda es la mitad del tamaño de la primera, etc.

    """

    # Antes de generar las imágenes, se alisa para evitar ruido.
    if sigma > 0:
        img = convolution_c(img, sigma=sigma)

    # Se resta 1 al número de imagenes que se quiere que aparezcan
    n -= 1

    # Se hace copia de la imagen para no modificar la original
    img = copy.deepcopy(img)

    # Se crea la lista donde se alojan todas las imágenes que finalmente se
    # mostrarán en la pirámide Laplaciana
    imgs = list()

    # Se crean las sucesivas imágenes de la pirámide Laplaciana dependiendo del
    # número que se reciba en "n"
    for i in range(n):

        # Se obtiene el tamaño de la imagen para comprobar si las filas o
        # columnas son pares o impares, ya que si son impares tiene problemas
        # para realizar los cálculos con las imágenes reducidas
        shape = img.shape

        # Para que yo haya problemas con pyrDown y pyrUp se debe tener una
        # imagen potencia de 2, por lo que se crea un contenedor con el
        # siguiente número potencia de 2 que pueda albergar la imagen
        new_rows = next_power_two(shape[0])
        new_cols = next_power_two(shape[1])

        # Se crea la imagen contenedor con las nuevas medidas
        img_aux = np.zeros((new_rows, new_cols))

        # Se copia la imagen actual en la imagen contenedor
        img_aux[0:shape[0], 0:shape[1]] = img

        # Se reduce la imagen contenedor a la mitad
        img_down = cv2.pyrDown(img_aux, borderType=border)

        # Se vuelve a ampliar la imagen a su tamaño original
        img_up = cv2.pyrUp(img_down)

        # Para poder hacer los cálculos, se guarda la parte que interesa de la
        # imagen contenedor
        img_aux = np.zeros((shape[0], shape[1]))

        img_aux = img_up[0:shape[0], 0:shape[1]]

        # Lo que interesa es la diferencia entre la imagen original y la
        # resultante de la reducción y ampliación
        img_l = img - img_aux

        # Se añade la imagen generada en la lista
        imgs.append(img_l)

        # Para la siguiente iteración es necesario comenzar desde la reducida
        img = cv2.pyrDown(img, borderType=border)

    # Para poder reconstruir la imagen general, es necesaria que la última
    # imagen sea la original de tamaño más pequeño que se haya generado
    imgs.append(img)

    return imgs

#
########

#
################################################################################

################################################################################
# Apartado 2
#

########
# Sección 1
#

def generate_low_high_imgs(img1, img2, sigma1 = 0, sigma2 = 0):
    """
    Función que recibe dos imágenes (img1 y img2) y devuelve la imagen con las
    frecuencias bajas de img1 y las frecuencias altas de img2.
    Se puede especificar el sigma de cada imagen para ajustarla dependiendo
    de cada imagen.
    """

    # Se calculan las frecuencias bajas aplicando un filtro gaussiano
    img1 = convolution_c(img1, sigma = sigma1)

    # Se calculan las frecuencias altas con la diferencia de la imagen original
    # y la imagen alisada
    img2 = img2 - convolution_c(img2, sigma = sigma2, normalize = False)

    return img1, img2

#
########

########
# Sección 2
#

def generate_low_high_hybrid_imgs(img1, img2, sigma1 = 0, sigma2 = 0):
    """
    Función que recibe dos imágenes (img1 y img2) y devuelve la imagen con las
    frecuencias bajas de img1, las frecuencias altas de img2 y la imagen híbrida
    combinando ambas.
    Hace uso de la función generate_low_high_imgs() que ya las calcula.
    Se puede especificar el sigma de cada imagen para ajustarla dependiendo
    de cada imagen.
    """

    # Se calculan las frecuencias bajas aplicando un filtro gaussiano
    img1, img2 = generate_low_high_imgs(img1, img2, sigma1, sigma2)

    # La imagen híbrida es la suma de la imagen con frecuencias bajas y altas
    img_hybrid = img1 + img2

    return img1, img2, img_hybrid

def show_hybrid(img1, img2, sigma1 = 0, sigma2 = 0):
    """
    Muestra las imágenes de frecuencia baja, alta e híbrida en una ventana
    """

    # Lista con las imágenes
    imgs = list()

    # Se generan las imágenes
    img_low, img_high, img_hybrid = generate_low_high_hybrid_imgs(img1, img2, sigma1, sigma2)

    # Se muestra la imagen con frecuencias bajas
    imgs.append(img_low)

    # Se muestra la imagen con frecuencias altas
    imgs.append(img_high)

    # Se muestra la imagen híbrida
    imgs.append(img_hybrid)

    # Se añaden los títulos a las imágenes
    imgs_title = list()

    imgs_title.append("Low")
    imgs_title.append("High")
    imgs_title.append("Hybrid")

    show_images(imgs, imgs_title)

#
########

########
# Sección 3
#

# No necesita función

#
########

################################################################################
# Pruebas
#

# En caso de utilizar este fichero de manera externa para el uso de sus
# funciones no se ejecutará ningún código
if __name__ == "__main__":

    ############################################################################
    # Apartado 1
    #

    ########
    # Sección A
    #

    """
    Una función que sea capaz de representar varias imágenes con sus títulos en
    una misma ventana. Usar esta función en todos los demás apartados.
    """

    # Se cargan las imágenes a mostrar, se aplica el esquema de color y se le
    # asigna un título
    img1 = cv2.imread(path+'cat.bmp')
    img1 = set_c_map(img1, cmap)
    img_title1 = "Gato"

    img2 = cv2.imread(path+'dog.bmp')
    img2 = set_c_map(img2, cmap)
    img_title2 = "Perro"

    img3 = cv2.imread(path+'plane.bmp')
    img3 = set_c_map(img3, cmap)
    img_title3 = "Avión"

    img4 = cv2.imread(path+'bird.bmp')
    img4 = set_c_map(img4, cmap)
    img_title4 = "Pájaro"

    img5 = cv2.imread(path+'einstein.bmp')
    img5 = set_c_map(img5, cmap)
    img_title5 = "Einstein"

    img6 = cv2.imread(path+'marilyn.bmp')
    img6 = set_c_map(img6, cmap)
    img_title6 = "Marilyn"

    # Se añaden las imágenes a la lista de imágenes
    imgs = []
    imgs.append(img1)
    imgs.append(img2)
    imgs.append(img3)
    imgs.append(img4)
    imgs.append(img5)
    imgs.append(img6)

    # Se añaden los títulos que se van a mostrar con las imágenes
    imgs_title = []
    imgs_title.append(img_title1)
    imgs_title.append(img_title2)
    imgs_title.append(img_title3)
    imgs_title.append(img_title4)
    imgs_title.append(img_title5)
    imgs_title.append(img_title6)

    # Se muestran las imágenes que se leen inicialmente
    show_images(imgs, imgs_title)

    #
    ########

    # Se vacían las listas con las imágenes y los nombres
    imgs.clear()
    imgs_title.clear()

    ########
    # Sección B
    #

    """
    Una función de convolución con máscara gaussiana de tamaño variable y sigma
    variable. Mostrar ejemplos de funcionamiento usando dos tipos de bordes y
    dos valores distintos de sigma.
    """

    # Se aplica convolución a la tercera imagen y se muestran los resultados
    imgs.append(img3)
    imgs.append(convolution_b(img3, 3))
    imgs.append(convolution_b(img3, 3, border = cv2.BORDER_REPLICATE))
    imgs.append(convolution_b(img3, 7))
    imgs.append(convolution_b(img3, 7, border = cv2.BORDER_CONSTANT))

    # Se modifican los nombres a las imágenes creadas
    imgs_title.append("Original")
    imgs_title.append("Sigma = 3")
    imgs_title.append("Sigma = 3, Replicate")
    imgs_title.append("Sigma = 7")
    imgs_title.append("Sigma = 7, Constant")

    # Se muestran las imágenes con las transformaciones
    show_images(imgs, imgs_title)

    #
    ########

    # Se vacían las listas con las imágenes y los nombres
    imgs.clear()
    imgs_title.clear()

    ########
    # Sección C
    #

    """
    Una función de convolución con núcleo separable de tamaño variable. Mostrar
    ejemplos de funcionamiento usando dos tipos de bordes y dos valores
    distintos de sigma.
    """

    # Se aplica convolución a la tercera imagen y se muestran los resultados
    imgs.append(img3)
    imgs.append(convolution_c(img3, sigma=3))
    imgs.append(convolution_c(img3, sigma=3, border = cv2.BORDER_REPLICATE))
    imgs.append(convolution_c(img3, sigma=7))
    imgs.append(convolution_c(img3, sigma=7, border = cv2.BORDER_CONSTANT))

    # Se modifican los nombres a las imágenes creadas
    imgs_title.append("Original")
    imgs_title.append("Sigma = 3")
    imgs_title.append("Sigma = 3, Replicate")
    imgs_title.append("Sigma = 7")
    imgs_title.append("Sigma = 7, Constant")

    show_images(imgs, imgs_title)

    #
    ########

    # Se vacían las listas con las imágenes y los nombres
    imgs.clear()
    imgs_title.clear()

    ########
    # Sección D
    #

    """
    Una función de convolución con núcleo de 1ª derivada de tamaño variable.
    Mostrar ejemplos de funcionamiento usando dos tipos de bordes y dos valores
    distintos de sigma.
    """

    # Se aplica convolución a la tercera imagen y se muestran los resultados
    imgs.append(img3)

    # Se hace convolución con núcleo de primera derivada de tamaño 3 sin borde
    img_x1, img_y1 = convolution_d(img3, ksize = 3)
    imgs.append(img_x1)
    imgs.append(img_y1)

    # Se hace convolución con núcleo de primera derivada de tamaño 3 con borde
    img_x2, img_y2 = convolution_d(img3, ksize = 3, border = cv2.BORDER_REPLICATE)
    imgs.append(img_x2)
    imgs.append(img_y2)

    # Se hace convolución con núcleo de primera derivada de tamaño 7 sin borde
    img_x3, img_y3 = convolution_d(img3, ksize = 7)
    imgs.append(img_x3)
    imgs.append(img_y3)

    # Se hace convolución con núcleo de primera derivada de tamaño 7 sin borde
    img_x4, img_y4 = convolution_d(img3, ksize = 7, border = cv2.BORDER_CONSTANT)
    imgs.append(img_x4)
    imgs.append(img_y4)

    # Se modifican los nombres a las imágenes creadas
    imgs_title.append("Original")
    imgs_title.append("ksize = 3, dx = 1")
    imgs_title.append("ksize = 3, dy = 1")
    imgs_title.append("ksize = 3, dx = 1, Replicate")
    imgs_title.append("ksize = 3, dy = 1, Replicate")
    imgs_title.append("ksize = 7, dx = 1")
    imgs_title.append("ksize = 7, dy = 1")
    imgs_title.append("ksize = 7, dx = 1, Constant")
    imgs_title.append("ksize = 7, dy = 1, Constant")

    show_images(imgs, imgs_title)

    #
    ########

    # Se vacían las listas con las imágenes y los nombres
    imgs.clear()
    imgs_title.clear()

    ########
    # Sección E
    #

    """
    Una función de convolución con núcleo de 2ª derivada de tamaño variable.
    Mostrar ejemplos de funcionamiento usando dos tipos de bordes y dos valores
    distintos de sigma.
    """

    # Se aplica convolución a la tercera imagen y se muestran los resultados
    imgs.append(img3)

    # Se hace convolución con núcleo de segunda derivada de tamaño 3 sin borde
    img_x1, img_y1 = convolution_e(img3, ksize = 3)
    imgs.append(img_x1)
    imgs.append(img_y1)

    # Se hace convolución con núcleo de segunda derivada de tamaño 3 con borde
    img_x2, img_y2 = convolution_e(img3, ksize = 3, border = cv2.BORDER_REPLICATE)
    imgs.append(img_x2)
    imgs.append(img_y2)

    # Se hace convolución con núcleo de segunda derivada de tamaño 7 sin borde
    img_x3, img_y3 = convolution_e(img3, ksize = 7)
    imgs.append(img_x3)
    imgs.append(img_y3)

    # Se hace convolución con núcleo de segunda derivada de tamaño 7 sin borde
    img_x4, img_y4 = convolution_e(img3, ksize = 7, border = cv2.BORDER_CONSTANT)
    imgs.append(img_x4)
    imgs.append(img_y4)

    # Se modifican los nombres a las imágenes creadas
    imgs_title.append("Original")
    imgs_title.append("ksize = 3, dx = 2")
    imgs_title.append("ksize = 3, dy = 2")
    imgs_title.append("ksize = 3, dx = 2, Replicate")
    imgs_title.append("ksize = 3, dy = 2, Replicate")
    imgs_title.append("ksize = 7, dx = 2")
    imgs_title.append("ksize = 7, dy = 2")
    imgs_title.append("ksize = 7, dx = 2, Constant")
    imgs_title.append("ksize = 7, dy = 2, Constant")

    show_images(imgs, imgs_title)

    #
    ########

    # Se vacían las listas con las imágenes y los nombres
    imgs.clear()
    imgs_title.clear()

    ########
    # Sección F
    #

    """
    Una función de convolución con núcleo Laplaciana-Gaussiana de tamaño
    variable. Mostrar ejemplos de funcionamiento usando dos tipos de bordes y
    dos valores distintos de sigma.
    """

    # Se aplica convolución a la tercera imagen y se muestran los resultados
    imgs.append(img3)

    # Se hace convolución con núcleo Laplaciana-Gaussiana de tamaño 3
    imgs.append(convolution_f(img3, ksize = 3, sigma = 1))

    # Se hace convolución con núcleo Laplaciana-Gaussiana de tamaño 3 y bordes
    imgs.append(convolution_f(img3, ksize = 3, border = cv2.BORDER_REPLICATE))

    # Se hace convolución con núcleo Laplaciana-Gaussiana de tamaño 7 y sigma 3
    # para el alisamiento
    imgs.append(convolution_f(img3, ksize = 7, sigma = 3))

    # Se hace convolución con núcleo Laplaciana-Gaussiana de tamaño 7 y bordes
    imgs.append(convolution_f(img3, ksize = 7, border = cv2.BORDER_CONSTANT))

    # Se modifican los nombres a las imágenes creadas
    imgs_title.append("Original")
    imgs_title.append("ksize = 3, Lap, sigma = 1")
    imgs_title.append("ksize = 3, Lap, Replicate")
    imgs_title.append("ksize = 7, Lap, sigma = 1")
    imgs_title.append("ksize = 7, Lap, Constant")

    show_images(imgs, imgs_title)

    #
    ########

    # Se vacían las listas con las imágenes y los nombres
    imgs.clear()
    imgs_title.clear()

    ########
    # Sección G
    #

    """
    Una función que genere una representación en pirámide Gaussiana de 4 niveles
    de una imagen. Mostrar ejemplos de funcionamiento usando dos tipos de bordes
    y dos valores distintos de sigma.
    """

    # Se muestra la pirámide gaussiana sin bordes
    imgs.append(show_pyr(generate_gaussian_pyr_imgs(img3)))

    # Se muestra la pirámide gaussiana con bordes
    imgs.append(show_pyr(generate_gaussian_pyr_imgs(img3, border = cv2.BORDER_REPLICATE)))

    imgs_title.append("Gaussian Pyramid")
    imgs_title.append("Gaussian Pyramid, Replicate")

    show_images(imgs, imgs_title)

    #
    ########

    # Se vacían las listas con las imágenes y los nombres
    imgs.clear()
    imgs_title.clear()

    ########
    # Sección H
    #

    """
    Una función que genere una representación en pirámide Laplaciana de 4
    niveles de una imagen. Mostrar ejemplos de funcionamiento usando dos tipos
    de bordes y dos valores distintos de sigma.
    """

    # Se muestra la pirámide gaussiana sin bordes
    imgs.append(show_pyr(generate_laplacian_pyr_imgs(img3, sigma = 0)))

    # Se muestra la pirámide gaussiana con bordes
    imgs.append(show_pyr(generate_laplacian_pyr_imgs(img3, sigma = 1, border = cv2.BORDER_REPLICATE)))

    imgs_title.append("Laplacian Pyramid")
    imgs_title.append("Laplacian Pyramid, Replicate")

    show_images(imgs, imgs_title)

    #
    ########

    # Se vacían las listas con las imágenes y los nombres
    imgs.clear()
    imgs_title.clear()

    #
    ############################################################################

    ############################################################################
    # Apartado 2
    #

    ########
    # Sección 1
    #

    """
    Implementar una función que genere las imágenes de baja y alta frecuencia
    a partir de las parejas de imágenes.
    """

    # Se generan las imágenes con frecuencias bajas y altas
    img_low, img_high = generate_low_high_imgs(img3, img4, 5, 1)

    # Se muestra la imagen con frecuencias bajas
    imgs.append(img_low)

    # Se muestra la imagen con frecuencias altas
    imgs.append(img_high)

    # Se añaden los títulos a las imágenes
    imgs_title.append("Low")
    imgs_title.append("High")

    show_images(imgs, imgs_title)

    #
    ########

    # Se vacían las listas con las imágenes y los nombres
    imgs.clear()
    imgs_title.clear()

    ########
    # Sección 2
    #

    """
    Escribir una función que muestre las tres imágenes (baja, alta e híbrida) en
    una misma ventana.
    """

    # Se muestran las imágenes
    show_hybrid(img3, img4, 7, 1)

    #
    ########

    ########
    # Sección 3
    #

    """
    Realizar la composición con al menos 3 de las parejas de imágenes
    """

    show_hybrid(img3, img4, 7, 1)

    show_hybrid(img2, img1, 7, 3)

    show_hybrid(img6, img5, 3, 3)

    #
    ########

#
###############################################################################

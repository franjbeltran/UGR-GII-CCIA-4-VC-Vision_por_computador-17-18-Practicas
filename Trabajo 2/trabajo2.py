# -*- coding: utf-8 -*-
"""

Francisco Javier Caracuel Beltrán

VC - Visión por Computador

4º - GII - CCIA - ETSIIT - UGR

Curso 2017/2018

Trabajo 2 - Dirección de puntos relevantes y Construcción de panoramas s

"""

import cv2
import numpy as np
import copy
import math
import random
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

################################################################################
# Configuración general
#

# Ruta hacia las imágenes
path = "imagenes/"

# Texto que se muestra para continuar ejecutando el fichero
continue_text = "Pulsa \"Enter\" para continuar..."

# Título para las imágenes
img_title = ""

# Número de columnas que tendrá la tabla con la salida de las imágenes
num_cols = 3

# Esquema de color que se utiliza por defecto para convertir las imágenes
cmap = cv2.COLOR_RGB2GRAY

# Se indica que se muestren la longitud completa de las matrices
np.set_printoptions(threshold=np.nan)

# Se inicializa SIFT
orb = cv2.ORB_create()
sift = cv2.xfeatures2d.SIFT_create()

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

def show_images(imgs, names = list(), cols = num_cols, title = "", gray = True):
    """
    Dada una lista de imágenes (imgs) y una lista de nombres (names), muestra en
    una tabla todas estas imágenes con su nombre correspondiente.
    Por defecto, el número de columnas que se van a mostrar es 3.
    Por defecto, el título que acompaña a cada imagen es "".
    """

    if gray:
        # Esquema de color que se utiliza por defecto en plt.imshow()
        plt.rcParams['image.cmap'] = 'gray'
    else:
        plt.rcParams['image.cmap'] = 'viridis'

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

        # La imagen se recibe con flotantes, por lo que se hace el cambio a
        # enteros
        #img = copy.deepcopy(img).astype(int)

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

def convolution_c(img , kernel_x = None, kernel_y = None, sigma = 0, border = cv2.BORDER_DEFAULT, normalize = True, own = False):
    """
    Dada una imagen (img) y dos núcleos (kernel_x, kernel_y) realiza una
    convolución de la imagen utilizando dichos núcleos.
    Se aplicará sucesivamente el núcleo por todas las filas de la imagen y con
    esta transformación se vuelve a aplicar a todas las columnas.
    Si se recibe un kernel_x "None", se extrae el kernel a través de la
    función cv2.getGaussianKernel().
    Si se recibe un kernel_y "None", será el mismo kernel ya calculado en
    kernel_x.
    Se puede indicar un borde, que por defecto está deshabilitado. Nota: si se
    utiliza la propia función de convolución añadida a partir del Bonus 2, la
    opción de borde no se contempla al utilizarse por defecto un borde
    reflejado.
    Permite la opción de normalizar, que por defecto está activa. Se le puede
    enviar el parámetro normalize = False para que no la haga.
    En el Bonus 2 se desarrolla la propia función de convolución y se adapta
    en este apartado (ya que se utiliza para el resto de práctica) para poder
    utilizarla. El parámetro "own" indica si se utiliza o no la propia función
    de convolución. Por defecto no se utiliza.

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
    if kernel_x is None:
        kernel_x = cv2.getGaussianKernel(6*sigma+1, sigma)

    # Se comprueba si se calcula el kernel_y
    if kernel_y is None:
        kernel_y = kernel_x

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

        ####
        # Ampliación Bonus 2
        #

        # A partir del Bonus 2, la función de convolución es propia, por lo que
        # en este punto se indica si se utiliza la propia o cv2.filter2D().
        if own:

            # La función de convolución propia devuelve una estructura
            # diferente a cv2.filter2D(). cv2.filter2D() devuelve una lista de
            # listas, la función de convolución propia devuelve una lista con
            # todos los elementos. El contenido de ambas es el mismo.
            img[i, :] = filter_2d(img[i, :], kernel_x)

        # Si no se utiliza la propia, se utiliza cv2.filter2D
        else:

            resConv = cv2.filter2D(img[i, :], -1, kernel_x, borderType=border)

            # Se guarda la lista de enteros creada en su fila correspondiente.
            # Como cv2.filter2D devuelve una lista de listas, de este modo se
            # convierte a lista de valores y se asigna a la fila correspondiente
            img[i, :] = [sublist[0] for sublist in resConv]

        #
        ####

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

        ####
        # Ampliación Bonus 2
        #

        # A partir del Bonus 2, la función de convolución es propia, por lo que
        # en este punto se indica si se utiliza la propia o cv2.filter2D().
        if own:

            # La función de convolución propia devuelve una estructura
            # diferente a cv2.filter2D(). cv2.filter2D() devuelve una lista de
            # listas, la función de convolución propia devuelve una lista con
            # todos los elementos. El contenido de ambas es el mismo.
            img[:, i] = filter_2d(img[:, i], kernel_y)

        # Si no se utiliza la propia, se utiliza cv2.filter2D
        else:

            resConv = cv2.filter2D(img[:, i], -1, kernel_y, borderType=border)

            # Se guarda la lista de enteros creada en su fila correspondiente.
            # Como cv2.filter2D devuelve una lista de listas, de este modo se
            # convierte a lista de valores y se asigna a la fila correspondiente
            img[:, i] = [sublist[0] for sublist in resConv]

        #
        ####

    # Se normalizan los resultados calculados
    if normalize:
        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

    # Se devuelve la imagen con la convolución realizada
    return img

def convolution_d(img , kernel_x = None, kernel_y = None, ksize = 3, sigma = 0, border = cv2.BORDER_DEFAULT, dx = 1, dy = 1, own = False):
    """
    Dada una imagen (img), realiza convolución con núcleo de primera derivada
    de tamaño ksize. Si se recibe un sigma, hace alisamiento con dicho sigma.
    Se puede indicar un borde, que por defecto está deshabilitado.
    Si se establece "own" a True utiliza la función de convolución propia.

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

    # Antes de hacer la convolución, se alisa para evitar ruido.
    if sigma > 0:
        img = convolution_c(img, sigma = sigma, own = own)

    # Si se modifican los valores de img, se verán reflejados en el resto de
    # imágenes que hagan uso de la imagen base. Para evitar esto, se hace un
    # copiado en el que no es por referencia
    img_x = copy.deepcopy(img)
    img_y = copy.deepcopy(img)

    # Se comprueba si se calcula el kernel_x
    if kernel_x is None:

        # Se calcula el kernel de la primera derivada con respecto a x.
        kernel_x = cv2.getDerivKernels(dx, 0, ksize)

        # Para aplicar convolución y no correlación se debe dar la vuelta al
        # kernel. Indicando -1 da la vuelta en el eje x e y, como solo tiene
        # uno, es correcto.
        #kernel_x = cv2.flip(kernel_x, -1)

    # Se comprueba si se calcula el kernel_y
    if kernel_y is None:

        # Se calcula el kernel de la primera derivada con respecto a y.
        kernel_y = cv2.getDerivKernels(0, dy, ksize)

        # Para aplicar convolución y no correlación se debe dar la vuelta al
        # kernel. Indicando -1 da la vuelta en el eje x e y, como solo tiene
        # uno, es correcto.
        #kernel_y = cv2.flip(kernel_y, -1)

    # Se hace la convolución de la 1º derivada con respecto a x.
    img_x = convolution_c(img_x, kernel_x[0], kernel_x[1], border = border, own = own)

    # Se hace la convolución de la 1º derivada con respecto a y.
    img_y = convolution_c(img_y, kernel_y[0], kernel_y[1], border = border, own = own)

    # Se devuelve la imagen con la convolución realizada
    return img_x, img_y

def generate_gaussian_pyr_imgs(img, n = 4, sigma = 0, sigma_down = 1, \
                                border = cv2.BORDER_DEFAULT, resize = False):
    """
    Función que, a partir de una imagen, genera n imágenes (por defecto, 4) de
    la mitad de tamaño cada vez.
    Para la generación de las imágenes se hace uso de la función cv2.pyrDown()
    que internamente aplica el alisado antes de reducir la imagen, por lo que no
    es necesario realizarlo antes.
    El valor de sigma indica el alisamiento que se realiza antes de generar
    todas las imágenes.
    sigma_down indica el alisamiento que se le realiza cuando se reduce
    manualmente.
    Se puede indicar un borde, que por defecto está deshabilitado.

    -------------------
    Opciones para el borde:
    - cv2.BORDER_REPLICATE
    - cv2.BORDER_REFLECT
    - cv2.BORDER_REFLECT_101
    - cv2.BORDER_WRAP
    - cv2.BORDER_CONSTANT
    -------------------

    Si se indica que se redimensione (resize = True) devolverá todas las
    imágenes del tamaño de la imagen original.

    Devuelve una lista con n+1 imágenes, donde la primera es la original, la
    segunda es la mitad del tamaño de la primera, etc.
    """

    # Antes de generar las imágenes, se alisa para evitar ruido.
    if sigma > 0:
        img = convolution_c(img, sigma=sigma, own = own)

    # Se crea la lista donde se alojan las imágenes
    imgs = list()

    # Se añade la imagen original a la lista
    imgs.append(img)

    if resize:

        # Se añaden tantas imágenes como se haya indicado
        for i in range(n):
            imgs.append(cv2.pyrUp(cv2.pyrDown(imgs[i], borderType=border), \
                        borderType=border))

    else:

        # Se añaden tantas imágenes como se haya indicado
        for i in range(n):
            imgs.append(cv2.pyrDown(imgs[i], borderType=border))

    return imgs

#
################################################################################

################################################################################
# Ejercicio 1
#

########
# Sección A
#

def get_block_size(sigma = 1.5):
    """
    Devuelve el tamaño de los vecinos (block_size) que se va a utilizar para
    obtener los puntos Harris. El valor se fija al valor correspondiente al uso
    de máscaras gaussianas de sigma 1.5. El tamaño de la máscara Gaussiana es
    6*1.5+1.
    """

    return int(6*sigma+1)

def get_ksize(sigma = 1):
    """
    Devuelve el parámetro de apertura (ksize) que se va a utilizar para
    obtener los puntos Harris. El valor se fija al valor correspondiente al uso
    de máscaras gaussianas de sigma 1. El tamaño de la máscara Gaussiana es
    6*1+1.
    """

    return int(6*sigma+1)

def selection_criteria_harris(l1, l2, k = 0.04):
    """
    Si se quiere obtener los puntos Harris de una imagen utilizando el operador
    Harris, es necesario seguir un criterio. En este caso se hará uso de los
    autovalores de M (l1, l2) y de la constante k.
    El criterio es: l1*l2 - k * (l1+l2)^2
    """

    # Se devuelve el valor dado siguiendo el criterio
    return ((l1*l2) - (k*((l1+l2)*(l1+l2))))

def is_center_local_max(data):
    """
    Recibe una matriz (data) y devuelve si el centro es el valor máximo de
    todos los elementos (True/False).
    """

    # Se guardan las dimensiones del objeto recibido para calcular su centro
    shape = data.shape

    # Se guarda el centro de los datos
    center = data[round(shape[0]/2)-1, round(shape[0]/2)-1]

    # Se devuelve si el centro es el máximo o no
    return center == np.max(data)

def not_max_suppression_harris(harris, threshold, env):
    """
    Suprime los valores no-máximos de una matriz (harris). Elimina los puntos
    que, aunque tengan un valor alto de criterio Harris, no son máximos locales
    de su entorno (env) para un tamaño de entorno prefijado.
    Permite la utilización de un umbral (threshold) para eliminar aquellos
    puntos Harris que se consideren elevados.
    Devuelve una lista con los keyPoints y su valor correspondiente.
    """

    # Se guardan las dimensiones de la imagen
    shape = harris.shape

    # Se consigue el mínimo valor para rellenar los bordes de la matriz con él
    min_harris = np.min(harris)

    # Se crea una nueva matriz donde se copiarán los puntos harris con un borde
    # de tamaño env relleno con el valor mínimo de los puntos Harris para que
    # no entre en conflicto al calcular el máximo local
    new_harris = np.ndarray(shape=(shape[0]+env+env, shape[1]+env+env))
    new_harris[:, :] = min_harris
    new_harris[env:shape[0]+env, env:shape[1]+env] = harris.copy()

    # Se crea una matriz rellena completamente a 255 con las mismas dimensiones
    # que la imagen actual
    res_harris = np.ndarray(shape=(shape[0]+env+env, shape[1]+env+env))
    res_harris[:, :] = 255

    # Los puntos Harris recibidos pueden tener multitud de valores. Si se
    # quieren obtener los más representativos se deben eliminar los que no
    # tienen un valor alto. En una zona completamente blanca pueden encontrarse
    # máximos locales, por lo que aparecerían en la imagen. Para evitar esto,
    # se indica un umbral, por el que por debajo de él no se tienen en cuenta
    # esos puntos
    threshold = 10**threshold

    # Lista con las coordenadas de los máximos locales y el valor que tiene
    # cada uno
    max_points = []

    # Se recorre cada posición de la matriz de puntos Harris (se ignora el
    # borde creado manualmente)
    for row in range(env, shape[0]+env):
        for col in range(env, shape[1]+env):

            # Si el punto actual tiene valor 255 o si su punto Harris está por
            # encima del umbral indicado se comprueba si es un máximo local
            if res_harris[row, col] == 255 and new_harris[row, col]>threshold:

                # Se obtiene el rango de datos que se deben conseguir para
                # comprobar que es máximo local, teniendo en cuenta el valor
                # de env
                row_insp_init = row - env
                row_insp_end = row + env
                col_insp_init = col - env
                col_insp_end = col + env

                # Se obtiene la matriz con los datos que rodean al punto actual
                data = new_harris[row_insp_init:row_insp_end+1, \
                                    col_insp_init:col_insp_end+1]

                # Se comprueba si el punto central es máximo local
                if is_center_local_max(data):

                    # En caso de ser máximo local, todo el rango de datos
                    # seleccionados se cambian a 0 para no volver a comprobarlos
                    res_harris[row_insp_init:row_insp_end+1, \
                                col_insp_init:col_insp_end+1] = 0

                    # Se guarda el punto actual real como máximo local
                    # utilizando la estructura cv2.KeyPoint
                    max_points.append([cv2.KeyPoint(row-env, col-env, _size=0, \
                                    _angle=0), harris[row-env, col-env]])

    # Se devuelven las coordenadas de los puntos máximos locales y su valor
    return max_points

def get_harris(img, sigma_block_size = 1.5, sigma_ksize = 1, k = 0.04, \
                threshold = -10000, env = 5, scale = -1):
    """
    Obtiene una lista potencial de los puntos Harris de una imagen (img).
    Los valores de los parámetros utilizados dependen del sigma que se
    recibe (sigma_block_size, sigma_ksize).
    Recibe (k), (threshol) y (env) utilizados en las funciones creadas para la
    obtención de los puntos.
    Se puede indicar la escala (scale) para devolver la escala a la que
    pertenecen los puntos generados junto a estos.
    """

    # Se obtiene block_size y ksize
    block_size = get_block_size(sigma_block_size)
    ksize = get_ksize(sigma_ksize)

    # Se calculan los autovalores y los autovectores de la imagen
    vals_and_vecs = cv2.cornerEigenValsAndVecs(img, block_size, ksize)

    # El resultado es una matriz de 3 dimensiones, donde se encuentra cada pixel
    # de la imagen y por cada pixel se tienen 6 valores:
    # - l1, l2: autovalores no ordenados de M
    # - x1, y1: autovalores correspondientes a l1
    # - x2, y2: autovalores correspondientes a l2
    # Se tiene que trabajar solo con los autovalores de M, por lo que se
    # separa en 6 canales el resultado y se guarda l1 y l2.
    vals_and_vecs = cv2.split(vals_and_vecs)
    l1 = vals_and_vecs[0]
    l2 = vals_and_vecs[1]

    # Se utiliza el criterio de Harris implementado para obtener la matriz
    # con el valor asociado a cada pixel
    harris = selection_criteria_harris(l1, l2, k)

    # Se suprimen los valores no máximos
    max_points = not_max_suppression_harris(harris, threshold, env)

    # Se guarda la longitud para recorrer todos los puntos y añadir la escala
    length = len(max_points)

    # Se añade la escala a cada punto
    for i in range(length):
        max_points[i][0].size = scale

    # Se devuelven los puntos
    return max_points

def get_best_harris(points, n = 1000):
    """
    Recibe una lista con los puntos Harris obtenidos formados por:
    (coordenadas (x, y), valor Harris, escala)
    Se puede especificar el número máximo de puntos Harris (n) que se quieren
    devolver, siendo elegidos los (n) de mayor valor.
    """

    # Se consigue una lista con los valores de los puntos
    values = [point[1] for point in points]

    # Se ordenan los puntos por el valor Harris de mayor a menor, guardando solo
    # los n (1000) puntos con valor más alto
    index_values_sorted = np.argsort(values)[::-1]
    index_values_sorted = index_values_sorted[0:n]

    # Se devuelven los puntos seleccionados
    return [points[i] for i in index_values_sorted]


def show_circles(img, points, radius = 2, orientations = False, color1 = 0, \
                    color2 = 255):
    """
    Dada una imagen (img), pinta sobre ella unos círculos cuyo centro se
    encuentra en unas coordenadas dadas por unos puntos (points).
    Estos puntos deben tener cuatro elementos: coordenadas (x, y), valor Harris,
    la escala (utilizada para el radio del círculo) y la orientación.
    Se puede indicar un radio (radius) para el punto y la línea.
    color1 indica el color de los puntos.
    color2 indica el color de las líneas de las orientaciones.
    """

    # Se hace un copiado de la imagen para no modificar la original
    img = copy.deepcopy(img)

    # En points se tienen las coordenadas de los puntos que se van a rodear con
    #un círculo
    for point in points:

        x = int(point[0].pt[0])
        y = int(point[0].pt[1])
        size = int(point[0].size)

        cv2.circle(img, center=(y, x), \
                    radius=size*radius, color=color1, thickness=2)

    # Si se indica que se dibujen las orientaciones
    if orientations:

        # Se recorren todos los puntos
        for point in points:

            # Se calcula el primer punto de la línea
            pt1 = (int(point[0].pt[1]), int(point[0].pt[0]))

            # Se calcula el segundo punto de la línea utilizando el ángulo y
            # se debe multiplicar por el radio utilizado al mostrar el punto
            pt2 = (int(point[0].pt[1]+np.sin(point[0].angle)*point[0].size*radius), \
                    int(point[0].pt[0]+np.cos(point[0].angle)*point[0].size*radius))

            # Se pinta la línea del punto actual
            cv2.line(img, pt1, pt2, color2)

    # Se devuelve la imagen con los puntos
    return img

#
########

########
# Sección B
#
# - http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
#

def refine_harris_points(img, points, size, env = 5, zero_zone = -1, \
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)):
    """
    Refina la posición de unos puntos (points) recibidos sobre una imagen (img)
    teniendo en cuenta el tamaño de la ventana (env) utilizada en la obtención
    de los puntos Harris.
    Zero_zone indica la mitad del tamaño de la zona muerta que se queda en el
    medio de la zona de búsqueda. Si se indica -1 no se le da tamaño.
    """

    # Se crea un array de numpy (requerido por la función cornerSubPix) con
    # las coordenadas de los puntos que se quieren refinar
    new_points = list()
    index = list()

    # Se recorren todos los puntos y solo se obtienen los que pertenezcan a la
    # al nivel de la pirámide correspondiente
    for i, point in enumerate(points):

        if point[0].size == size:
            new_points.append(point[0].pt)
            index.append(i)

    new_points = np.float32(new_points)

    # Se refinan los puntos (deben ser float)
    cv2.cornerSubPix(img, new_points, (env, env), \
                        (zero_zone, zero_zone), criteria)

    # Se modifican las coordenadas con los puntos refinados
    for i, point in enumerate(new_points):
        points[index[i]][0].pt = (point[0], point[1])

    return points

#
########

########
# Sección C
#

def get_orientations(img, points, sigma, own = True):
    """
    A partir de una imagen (img) y los puntos Harris calculados (points),
    obtiene la orientación que tendrá cada uno de los puntos.
    Se especifica un sigma (sigma) para aplicar antes de calcular las derivadas
    un alisamiento con ese sigma.
    Devuelve la orientación de cada punto.
    """

    # Se calculan las derivadas con respecto de x y de y de la imagen, aplicando
    # antes un alisamiento definido por sigma
    img_x, img_y = convolution_d(img, sigma = sigma, own = own)

    # Para aplicar el arcotangente es necesario modificar la estructura del
    # array de puntos de manera que existan dos filas (x, y) y tantas columnas
    # como puntos se tengan, por lo que se hace la transpuesta.
    new_points = np.array([(int(point[0].pt[0]), int(point[0].pt[1])) for point in points]).T

    # Se calcula la arcotangente de cada punto que devuelve la orientación que
    # tendra dicho punto. Esa operación se realiza para todos los puntos que se
    # tienen en el array
    orientations = np.arctan2(img_x, img_y)[new_points[0], new_points[1]]

    for point, orientation in zip(points, orientations):
        point[0].angle = orientation/np.pi*180

    # Se devuelven los puntos con sus orientaciones
    return points

#
########

########
# Sección D
#

def get_descriptors(img, keypoints):
    """
    Calcula los descriptores de los KeyPoints de la imagen y los devuelve junto
    a los KeyPoints en formato cv2.KeyPoint.
    """

    # Se convierte la imagen a enteros para que se puedan calcular los
    # descriptores
    img = np.uint8(img)

    # De la lista con los puntos que se tenía, se crea otra lista con el tipo
    # de datos cv2.KeyPoint para poder utilizar sift.compute()
    keypoints = [point[0] for point in keypoints]

    # Se calculan los descriptores
    return sift.compute(img, keypoints)

#
########

#
################################################################################

################################################################################
# Ejercicio 2
#
# - https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
#

def get_keypoints_descriptors(img, own = False):
    """
    Dada una imagen, calcula sus keypoints y descriptores y los devuelve
    """

    if own:

        # Se obtienen los puntos Harris
        keypoints = get_harris(img, scale = 1)

        # Se guardan los mejores
        keypoints = get_best_harris(points)

        # Se actualizan los puntos con las orientaciones que tienen
        keypoints = get_orientations(img, points, sigma=5, own = False)

        # Se calculan los descriptores utilizando los keypoints generados
        # manualmente
        keypoints, descriptors = get_descriptors(img, keypoints)

    else:

        # Se extraen los keyPoints y los descriptores de las dos primeras
        # imágenes
        keypoints, descriptors = sift.detectAndCompute(img, None)

    # Se devuelven los keypoints y descriptores calculados
    return keypoints, descriptors


def get_matches_bf_cc(img1, img2, n = 30, flag = 2, get_data = False):
    """
    Dadas dos imágenes (img1, img2), calcula los keypoints y los descriptores
    para obtener los matches de ambas imágenes con la técnica
    "BruteForce+crossCheck".
    Se puede indicar el número de matches que se quieren mostrar (n).
    El flag permite indicar si se quieren que se muestren los keypoints y los
    matches (0) o solo los matches (2).
    Si se indica get_data = True, se devolverán los keypoints, descriptores y
    los matches.
    Devuelve una imagen que se compone de las dos imágenes con sus matches.
    """

    # Se obtienen los keypoints y los descriptores de las dos imágenes
    keypoints1, descriptors1 = get_keypoints_descriptors(img1)
    keypoints2, descriptors2 = get_keypoints_descriptors(img2)

    # Se crea el objeto BFMatcher de OpenCV activando la validación cruzada
    bf = cv2.BFMatcher(crossCheck = True)

    # Se consiguen los puntos con los que hace match
    matches = bf.match(descriptors1, descriptors2)

    # Se ordenan los matches dependiendo de la distancia entre ambos puntos
    # guardando solo los n mejores. Solo se hace si se quiere una mejora de
    # los matches
    #matches = sorted(matches, key = lambda x:x.distance)[0:n]

    # Se guardan solo algunos puntos (n) aleatorios
    matches = random.sample(matches, n)

    # Se crea la imagen que se compone de ambas imágenes con los matches
    # generados. El sexto parámetro es la variable donde se quiere guardar la
    # salida
    matcher = cv2.drawMatches(yos1, keypoints1, yos2, keypoints2, matches, \
                                None, flags = flag)

    # El resultado normal será la imagen con los matches
    res = matcher

    # Si se ha indicado que se devuelvan los datos, el resultado serán todos Los
    # datos calculados
    if get_data:
        res = ((keypoints1, descriptors1), (keypoints2, descriptors2), matches)

    return res

def get_matches_knn(img1, img2, k = 2, ratio = 0.75, n = 30, flag = 2, \
                        get_data = False, improve = False):
    """
    Dadas dos imágenes (img1, img2), calcula los keypoints y los descriptores
    para obtener los matches de ambas imágenes con la técnica
    "Lowe-Average-2NN", utilizando los k vecinos más cercanos (por defecto, 2).
    Para obtener los mejores matches, se calcula utilizando un ratio (ratio).
    Se puede indicar el número de matches que se quieren mostrar (n).
    El flag permite indicar si se quieren que se muestren los keypoints y los
    matches (0) o solo los matches (2).
    Si se indica get_data = True, se devolverán los keypoints, descriptores y
    los matches.
    Si se indica el flag "improve" como True, elegirá los mejores matches.
    Devuelve una imagen que se compone de las dos imágenes con sus matches.
    """

    # Se obtienen los keypoints y los descriptores de las dos imágenes
    keypoints1, descriptors1 = get_keypoints_descriptors(img1)
    keypoints2, descriptors2 = get_keypoints_descriptors(img2)

    # Se crea el objeto BFMatcher de OpenCV
    bf = cv2.BFMatcher()

    # Se consiguen los puntos con los que hace match indicando los vecinos más
    # cercanos con los que se hace la comprobación
    matches = bf.knnMatch(descriptors1, descriptors2, k)

    # Si se indica que se elijan los mejores matches
    if improve:

        # Solo se hace para hacer una mejora de los matches
        # Se guardan los puntos que cumplan con un radio en concreto
        good = []

        # Se recorren todos los matches
        for p1, p2 in matches:

            # Si la distancia del punto de la primera imagen es menor que la
            # distancia del segundo punto aplicándole un ratio, el punto se
            # considera bueno
            if p1.distance < ratio*p2.distance:
                good.append([p1])

        # Se ordenan los matches dependiendo de la distancia entre ambos puntos
        # guardando solo los n mejores
        matches = sorted(good, key = lambda x:x[0].distance)

    # Se guardan solo algunos puntos (n) aleatorios
    matches = random.sample(matches, n)

    # Se crea la imagen que se compone de ambas imágenes con los matches
    # generados. El sexto parámetro es la variable donde se quiere guardar la
    # salida
    matcher = cv2.drawMatchesKnn(yos1, keypoints1, yos2, keypoints2, matches, \
                                None, flags = flag)

    # El resultado normal será la imagen con los matches
    res = matcher

    # Si se ha indicado que se devuelvan los datos, el resultado serán todos Los
    # datos calculados
    if get_data:
        res = ((keypoints1, descriptors1), (keypoints2, descriptors2), matches)

    return res

#
################################################################################

################################################################################
# Ejercicio 3
#
# - https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
# - https://github.com/tsherlock/panorama/blob/master/pano_stitcher.py

def get_homography(img1, img2, improve = False):
    """
    Obtiene la homografía entre dos imágenes y la devuelve.
    Si se indica el flag "improve" como True, elegirá los mejores matches.
    """

    # Se calculan los keypoints, descriptores y matches de ambas imágenes
    data = get_matches_knn(img1, img2, get_data = True, improve = improve)

    # Se obtienen los keypoints, descriptores y matches entre ellos
    keypoints1 = data[0][0]
    descriptors1 = data[0][1]

    keypoints2 = data[1][0]
    descriptors2 = data[1][1]

    matches = data[2]

    # Se crean los puntos de la imagen fuente generando una matriz de filas
    # como matches hay y 2 columnas
    src_points = np.float32([keypoints1[point[0].queryIdx].pt \
                                    for point in matches]).reshape(-1, 1, 2)

    # Se crean los puntos de la imagen destino generando una matriz de filas
    # como matches hay y 2 columnas
    dst_points = np.float32([keypoints2[point[0].trainIdx].pt \
                                    for point in matches]).reshape(-1, 1, 2)

    # Para generar una homografía son necesarios al menos cuatro puntos, si no
    # se obtienen esos cuatro puntos, se devuelve una matriz 3x3 vacía
    if len(src_points) > 4:
        M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 1)
    else:
        M = np.ndarray((3, 3), dtype=cv2.float32)

    # Se devuelve la homografía
    return M

def get_homography_to_center(img, width, height):
    """
    Calcula la homografía necesaria para llevar una imagen (img) al centro de
    un mosaico de tamaño (width x height)
    """

    # Se obtienen las dimensiones de la imagen central
    height_center, width_center, z_center = img.shape

    # Se calculan las coordenadas de inicio de la imagen central
    t_width = width/2 - width_center/2
    t_height = height/2 - height_center/2

    # Se crea una matriz 3x3 con los valores a 0 menos la diagonal a 1 y los
    # valores de la traslación
    m0 = np.array([[1, 0, t_width], [0, 1, t_height], [0, 0, 1]], \
                    dtype=np.float32)

    return m0

def get_mosaic(*imgs, crop = True, improve = False):
    """
    Genera un mosaico a partir de las imágenes recibidas (imgs).
    Se puede especificar si se recortan los bordes de la imagen (crop = True).
    Si se indica el flag "improve" como True, elegirá los mejores matches.
    Devuelve una imagen que contiene el mosaico.
    """

    # Se calcula cuál es el número de la imagen que se encuentra en el centro
    index_img_center = int(len(imgs)/2)

    # Se obtiene la imagen que está en el centro
    img_center =  imgs[index_img_center]

    # Se calcula el ancho del mosaico como la suma de todos los anchos de las
    # imágenes
    width = sum([img_center.shape[1] for img in imgs])

    # Se calcula la altura del mosaico
    height = imgs[0].shape[0]*2

    # Para componer el mosaico hace falta una homografía principal que se
    # encargue de llevar las imágenes desde su posición hasta el centro.
    # Se debe calcular manualmente indicando el desplazamiento de la imagen
    # central hasta el centro del mosaico
    m0 = get_homography_to_center(img_center, width, height)

    # Una vez que se tiene la homografía principal, se coloca la imagen central
    # en el centro del mosaico
    img = cv2.warpPerspective(img_center, m0, (width, height), \
                                borderMode=cv2.BORDER_TRANSPARENT)

    # Se crea una lista con las homografías
    homographies = [None] * len(imgs)

    # Se añade la principal en la posición que ocupa la imagen central
    homographies[index_img_center] = m0

    # Se recorren todas las imágenes hacia la izquierda, haciendo las
    # homografías de unas con otras y colocándolas hacia el centro de la imagen
    # con la homografía principal
    for i in range(0, index_img_center)[::-1]:

        # Se calcula la homografía de la imagen actual con la de su derecha
        m = get_homography(imgs[i], imgs[i+1], improve = improve)

        # Se le aplica a la homografía la transformación que permite llevar
        # la imagen a la parte central
        m = np.dot(homographies[i+1], m)

        # Se guarda la homografía actual para que sea utilizada por el resto
        homographies[i] = m

        # Se aplica la transformación a la imagen para juntarla con las que ya
        # se encuentran en el mosaico
        img = cv2.warpPerspective(imgs[i], m, (width, height), \
                                    dst=img, borderMode=cv2.BORDER_TRANSPARENT)

    # Se recorren todas las imágenes hacia la derecha, haciendo las homografías
    # de unas con otras y colocándolas hacia el centro de la imagen con la
    # homografía principal
    for i in range(index_img_center+1, len(imgs)):

        # Se calcula la homografía de la imagen actual con la de su izquierda
        m = get_homography(imgs[i], imgs[i-1], improve = improve)

        # Se le aplica a la homografía la transformación que permite llevar
        # la imagen a la parte central
        m = np.dot(homographies[i-1], m)

        # Se guarda la homografía actual para que sea utilizada por el resto
        homographies[i] = m

        # Se aplica la transformación a la imagen para juntarla con las que ya
        # se encuentran en el mosaico
        img = cv2.warpPerspective(imgs[i], m, (width, height), \
                                    dst=img, borderMode=cv2.BORDER_TRANSPARENT)

    # Si se ha indicado, se recortan los bordes de la imagen
    if crop:
        img = crop_image(img)

    # Se devuelve la imagen
    return img

def crop_image(img):
    """
    Recorta todos los bordes negros a una imagen y la devuelve.
    Código completamente obtenido de:
    - https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
    """

    # Se convierte la imagen a escala de grises
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Interesa guardar el segundo valor devuelto que contiene una matriz del
    # tamaño de la imagen con valores binarios, diferenciando el negro del
    # resto de colores
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Se llama a la función de OpenCV que devuelve las coordenadas de todos Los
    # puntos que forman el contorno
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, \
                                    cv2.CHAIN_APPROX_SIMPLE)[0]

    # Se obtienen las esquinas de los puntos obtenidos anteriormente
    x, y, w, h = cv2.boundingRect(contours)

    # Se devuelve la imagen delimitada por las esquinas
    return img[y:y+h,x:x+w]

#
################################################################################

################################################################################
# Ejecuciones
#

# En caso de utilizar este fichero de manera externa para el uso de sus
# funciones no se ejecutará ningún código
if __name__ == "__main__":

    # Se insertan los ejercicios que se quieren ejecutar
    ex = list()

    # 1.a:
    ex.append(1)

    # 1.b
    ex.append(2)

    # 1.c:
    ex.append(3)

    # 1.d:
    ex.append(4)

    # 2:
    ex.append(5)

    # 3:
    ex.append(6)

    # 4:
    ex.append(7)

    # Listas para la visualización de las imágenes
    imgs = []
    imgs_title = []

    # Se cargan las imágenes a mostrar, se aplica el esquema de color, se
    # convierten los valores a flotantes para trabajar bien con ellos y se le
    # asigna un título
    img1 = cv2.imread(path+'yosemite1.jpg')
    img1 = set_c_map(img1, cmap)
    img1 = np.float32(img1)
    img_title1 = "Yosemite1"

    # Imágenes Yosemite
    yos1 = mpimg.imread(path+'yosemite1.jpg')
    #yos1 = set_c_map(yos1)
    #yos1 = np.float32(yos1)
    yos_title1 = "Yosemite1"

    yos2 = mpimg.imread(path+'yosemite2.jpg')
    #yos2 = set_c_map(yos2, cmap)
    #yos2 = np.float32(yos2)
    yos_title2 = "Yosemite2"

    yos3 = mpimg.imread(path+'yosemite3.jpg')
    #yos3 = set_c_map(yos3, cmap)
    #yos3 = np.float32(yos3)
    yos_title3 = "Yosemite3"

    yos4 = mpimg.imread(path+'yosemite4.jpg')
    #yos4 = set_c_map(yos4, cmap)
    #yos4 = np.float32(yos4)
    yos_title4 = "Yosemite4"

    yos5 = mpimg.imread(path+'yosemite5.jpg')
    #yos5 = set_c_map(yos5, cmap)
    #yos5 = np.float32(yos5)
    yos_title5 = "Yosemite5"

    yos6 = mpimg.imread(path+'yosemite6.jpg')
    #yos6 = set_c_map(yos6, cmap)
    #yos6 = np.float32(yos6)
    yos_title6 = "Yosemite6"

    yos7 = mpimg.imread(path+'yosemite7.jpg')
    #yos7 = set_c_map(yos7, cmap)
    #yos7 = np.float32(yos7)
    yos_title7 = "Yosemite7"

    # Imágenes Mosaico
    mos2 = mpimg.imread(path+'mosaico002.jpg')
    #yos1 = set_c_map(yos1)
    #yos1 = np.float32(yos1)
    mos_title2 = "Mos2"

    mos3 = mpimg.imread(path+'mosaico003.jpg')
    #yos1 = set_c_map(yos1)
    #yos1 = np.float32(yos1)
    mos_title3 = "Mos3"

    mos4 = mpimg.imread(path+'mosaico004.jpg')
    #yos1 = set_c_map(yos1)
    #yos1 = np.float32(yos1)
    mos_title4 = "Mos4"

    mos5 = mpimg.imread(path+'mosaico005.jpg')
    #yos1 = set_c_map(yos1)
    #yos1 = np.float32(yos1)
    mos_title5 = "Mos5"

    mos6 = mpimg.imread(path+'mosaico006.jpg')
    #yos1 = set_c_map(yos1)
    #yos1 = np.float32(yos1)
    mos_title6 = "Mos6"

    mos7 = mpimg.imread(path+'mosaico007.jpg')
    #yos1 = set_c_map(yos1)
    #yos1 = np.float32(yos1)
    mos_title7 = "Mos7"

    mos8 = mpimg.imread(path+'mosaico008.jpg')
    #yos1 = set_c_map(yos1)
    #yos1 = np.float32(yos1)
    mos_title8 = "Mos8"

    mos9 = mpimg.imread(path+'mosaico009.jpg')
    #yos1 = set_c_map(yos1)
    #yos1 = np.float32(yos1)
    mos_title9 = "Mos9"

    mos10 = mpimg.imread(path+'mosaico010.jpg')
    #yos1 = set_c_map(yos1)
    #yos1 = np.float32(yos1)
    mos_title10 = "Mos10"

    mos11 = mpimg.imread(path+'mosaico011.jpg')
    #yos1 = set_c_map(yos1)
    #yos1 = np.float32(yos1)
    mos_title11 = "Mos11"

    owna = mpimg.imread(path+'owna.jpg')
    ownb = mpimg.imread(path+'ownb.jpg')
    ownc = mpimg.imread(path+'ownc.jpg')

    own1 = mpimg.imread(path+'own1.jpg')
    own2 = mpimg.imread(path+'own2.jpg')
    own3 = mpimg.imread(path+'own3.jpg')
    own4 = mpimg.imread(path+'own4.jpg')
    own5 = mpimg.imread(path+'own5.jpg')
    own6 = mpimg.imread(path+'own6.jpg')
    own7 = mpimg.imread(path+'own7.jpg')
    own8 = mpimg.imread(path+'own8.jpg')
    own9 = mpimg.imread(path+'own9.jpg')
    own10 = mpimg.imread(path+'own10.jpg')
    own11 = mpimg.imread(path+'own11.jpg')
    own12 = mpimg.imread(path+'own12.jpg')
    own13 = mpimg.imread(path+'own13.jpg')

    ############################################################################
    # Ejercicio 1
    #

    ########
    # Sección A
    #

    """
    Una función que sea capaz de representar varias imágenes con sus títulos en
    una misma ventana. Usar esta función en todos los demás apartados.
    """

    if 1 in ex:

        # Se añade la imagen original a la lista de imágenes
        imgs.append(img1)

        # Se añade un título a la imagen
        imgs_title.append(img_title1)

        # Se obtienen las imágenes de la pirámide Gaussiana indicando que se
        # mantenga el tamaño de la imagen original en las imágenes de los
        # distintos niveles de la pirámide
        img_gauss = generate_gaussian_pyr_imgs(img1, resize = True)

        # Se crea la lista de puntos Harris que se van a mostrar
        points = []

        # Se recorre cada imagen y se le calculan los puntos Harris
        for i, img in enumerate(img_gauss):
            points = points + get_harris(img, scale = i+1)

        # Se escogen los 500 mejores
        points500 = get_best_harris(points, 500)

        # Se añade la imagen a la lista de imágenes
        imgs.append(show_circles(img1, points500))

        # Se añade un título a la imagen
        imgs_title.append(img_title1+": 500 puntos")

        # Cuando se tiene una lista con los puntos Harris en todas las escalas,
        # se escogen los 1000 mejores
        points1000 = get_best_harris(points, 1000)

        # Se añade la imagen a la lista de imágenes
        imgs.append(show_circles(img1, points1000))

        # Se añade un título a la imagen
        imgs_title.append(img_title1+": 1000 puntos")

        # Se escogen los 1500 mejores
        points1500 = get_best_harris(points, 1500)

        # Se añade la imagen a la lista de imágenes
        imgs.append(show_circles(img1, points1500))

        # Se añade un título a la imagen
        imgs_title.append(img_title1+": 1500 puntos")

        # Se muestran las imágenes que se leen inicialmente
        show_images(imgs, imgs_title, cols = 2)

        input(continue_text)

    #
    ########

    # Se vacían las listas con las imágenes y los nombres
    imgs.clear()
    imgs_title.clear()

    ########
    # Sección B
    #

    """
    Extraer los valores (cx, cy, escala) de cada uno de los puntos resultantes
    en el apartado anterior y refinar su posición espacial a nivel sub-pixel
    usando la función de OpenCV cornerSubPix() con la imagen del nivel de
    pirámide correspondiente. Actualizar los datos (cx, cy, escala) de cada uno
    de los puntos encontrados.
    """

    if 2 in ex:

        # Se añade la imagen original a la lista de imágenes
        imgs.append(img1)

        # Se añade un título a la imagen
        imgs_title.append(img_title1)

        # Se obtienen las imágenes de la pirámide Gaussiana indicando que se
        # mantenga el tamaño de la imagen original en las imágenes de los
        # distintos niveles de la pirámide
        img_gauss = generate_gaussian_pyr_imgs(img1, resize = True)

        # Se crea la lista de puntos Harris que se van a mostrar
        points = []

        # Se recorre cada imagen y se le calculan los puntos Harris
        for i, img in enumerate(img_gauss):
            points = points + get_harris(img, scale = i+1)

        # Se crea la lista de puntos Harris que se van a refinar
        refine_points = []

        # Se recorre cada imagen y se le calculan los puntos Harris
        for i, img in enumerate(img_gauss):
            refine_points = refine_points + \
                                    refine_harris_points(img, points, i+1)

        points = refine_points

        # Se escogen los 500 mejores
        points500 = get_best_harris(points, 500)

        # Se añade la imagen a la lista de imágenes
        imgs.append(show_circles(img1, points500))

        # Se añade un título a la imagen
        imgs_title.append(img_title1+": 500 puntos")

        # Cuando se tiene una lista con los puntos Harris en todas las escalas,
        # se escogen los 1000 mejores
        points1000 = get_best_harris(points, 1000)

        # Se añade la imagen a la lista de imágenes
        imgs.append(show_circles(img1, points1000))

        # Se añade un título a la imagen
        imgs_title.append(img_title1+": 1000 puntos")

        # Se escogen los 1500 mejores
        points1500 = get_best_harris(points, 1500)

        # Se añade la imagen a la lista de imágenes
        imgs.append(show_circles(img1, points1500))

        # Se añade un título a la imagen
        imgs_title.append(img_title1+": 1500 puntos")

        # Se muestran las imágenes que se leen inicialmente
        show_images(imgs, imgs_title, cols = 2)

        input(continue_text)

    #
    ########

    # Se vacían las listas con las imágenes y los nombres
    imgs.clear()
    imgs_title.clear()

    ########
    # Sección C
    #

    """
    Calcular la orientación relevante de cada punto Harris usando el
    arcotangente del gradiente en cada punto
    """

    if 3 in ex:

        # Se añade la imagen original a la lista de imágenes
        #imgs.append(img1)

        # Se añade un título a la imagen
        #imgs_title.append(img_title1)

        # Se obtienen las imágenes de la pirámide Gaussiana indicando que se
        # mantenga el tamaño de la imagen original en las imágenes de los
        # distintos niveles de la pirámide
        img_gauss = generate_gaussian_pyr_imgs(img1, resize = True)

        # Se crea la lista de puntos Harris que se van a mostrar
        points = []

        # Se recorre cada imagen y se le calculan los puntos Harris
        for i, img in enumerate(img_gauss):
            points = points + get_harris(img, scale = i+1)

        # Se escogen los 500 mejores
        points500 = get_best_harris(points, 500)

        # Se actualizan los puntos con las orientaciones que tienen
        points500 = get_orientations(img1, points500, sigma=5, own = False)

        # Se añade la imagen a la lista de imágenes
        imgs.append(show_circles(img1, points500, orientations = True))

        # Se añade un título a la imagen
        imgs_title.append(img_title1+": 500 puntos")

        # Se muestran las imágenes que se leen inicialmente
        show_images(imgs, imgs_title, cols = 1)

        # Se vacían las listas con las imágenes y los nombres
        imgs.clear()
        imgs_title.clear()

        # Se escogen los 1000 mejores
        points1000 = get_best_harris(points, 1000)

        # Se actualizan los puntos con las orientaciones que tienen
        points1000 = get_orientations(img1, points1000, sigma=5, own = False)

        # Se añade la imagen a la lista de imágenes
        imgs.append(show_circles(img1, points1000, orientations = True))

        # Se añade un título a la imagen
        imgs_title.append(img_title1+": 1000 puntos")

        # Se muestran las imágenes que se leen inicialmente
        show_images(imgs, imgs_title, cols = 1)

        # Se vacían las listas con las imágenes y los nombres
        imgs.clear()
        imgs_title.clear()

        # Se escogen los 1500 mejores
        points1500 = get_best_harris(points, 1500)

        # Se actualizan los puntos con las orientaciones que tienen
        points1500 = get_orientations(img1, points1500, sigma=5, own = False)

        # Se añade la imagen a la lista de imágenes
        imgs.append(show_circles(img1, points1500, orientations = True))

        # Se añade un título a la imagen
        imgs_title.append(img_title1+": 1500 puntos")

        # Se muestran las imágenes que se leen inicialmente
        show_images(imgs, imgs_title, cols = 1)

        input(continue_text)

    #
    ########

    # Se vacían las listas con las imágenes y los nombres
    imgs.clear()
    imgs_title.clear()

    ########
    # Sección D
    #

    """
    Usar el vector de keyPoint extraidos para calcular los descriptores SIFT
    asociados a cada punto
    """

    if 4 in ex:

        # Se obtienen las imágenes de la pirámide Gaussiana indicando que se
        # mantenga el tamaño de la imagen original en las imágenes de los
        # distintos niveles de la pirámide
        img_gauss = generate_gaussian_pyr_imgs(img1, resize = True)

        # Se crea la lista de puntos Harris que se van a mostrar
        points = []

        # Se recorre cada imagen y se le calculan los puntos Harris
        for i, img in enumerate(img_gauss):
            points = points + get_harris(img, scale = i+1)

        # Se escogen los 500 mejores
        points500 = get_best_harris(points, 500)

        # Se actualizan los puntos con las orientaciones que tienen
        points500 = get_orientations(img1, points500, sigma=5, own = False)

        # Se obtienen los keypoints en formato cv2.KeyPoint junto con sus
        # descriptores
        keypoints, descriptors = get_descriptors(img1, points500)

        # Se visualizan los descriptores calculados
        print(descriptors)

        input(continue_text)

    #
    ########

    #
    ############################################################################

    ############################################################################
    # Ejercicio 2
    #

    """
    Usar el detector SIFT de OpenCV sobre las imágenes de Yosemite.rar. Extraer
    sus listas de keyPoints y descriptores asociados. Establecer las
    correspondencias existentes entre ellos usando el objeto BFMatcher de
    OpenCV.
    """

    if 5 in ex:

        # Se añade la imagen con los matches calculados por
        # bruteForce+crossCheck
        imgs.append(get_matches_bf_cc(yos1, yos2))

        # Se añade un título a la imagen
        imgs_title.append("BruteForce+CrossCheck")

        # Se añade la imagen con los matches calculados por Lowe-Average-2NN
        imgs.append(get_matches_knn(yos1, yos2))

        # Se añade un título a la imagen
        imgs_title.append("Lowe-Average-2NN")

        # Se muestran las imágenes que se leen inicialmente
        show_images(imgs, imgs_title, cols = 1, gray = False)

        input(continue_text)

    #
    ############################################################################

    # Se vacían las listas con las imágenes y los nombres
    imgs.clear()
    imgs_title.clear()

    ############################################################################
    # Ejercicio 3
    #

    """
    Escribir una función que genere un mosaico de calidad a partir de n=3
    imágenes relacionadas por homografías, sus listas de keyPoints calculados
    de acuerdo al punto anterior y las correspondencias encontradas entre dichas
    listas.
    """

    if 6 in ex:

        # Se añade el mosaico de las imágenes que se quieran
        imgs.append(get_mosaic(yos1, yos2, yos3, improve = True))

        # Se añade un título a la imagen
        imgs_title.append("Yosemite Mosaic")

        # Se muestran las imágenes que se leen inicialmente
        show_images(imgs, imgs_title, cols = 1, gray = False)

        # Se vacían las listas con las imágenes y los nombres
        imgs.clear()
        imgs_title.clear()

        # Se añade el mosaico de las imágenes que se quieran
        imgs.append(get_mosaic(owna, ownb, ownc, improve = True))

        # Se añade un título a la imagen
        imgs_title.append("ETSIIT Mosaic")

        # Se muestran las imágenes que se leen inicialmente
        show_images(imgs, imgs_title, cols = 1, gray = False)

        input(continue_text)

    #
    ############################################################################

    # Se vacían las listas con las imágenes y los nombres
    imgs.clear()
    imgs_title.clear()

    ############################################################################
    # Ejercicio 4
    #

    """
    Similar al punto anterior pero para N > 5
    """

    if 7 in ex:

        # Se añade el mosaico de las imágenes que se quieran
        imgs.append(get_mosaic(mos2, mos3, mos4, mos5, mos6, mos7, mos8, mos9, \
                                    mos10, mos11))

        # Se añade un título a la imagen
        imgs_title.append("ETSIIT Mosaic")

        # Se muestran las imágenes que se leen inicialmente
        show_images(imgs, imgs_title, cols = 1, gray = False)

        # Se vacían las listas con las imágenes y los nombres
        imgs.clear()
        imgs_title.clear()

        # Se añade el mosaico de las imágenes que se quieran
        imgs.append(get_mosaic(own1, own2, own3, own4, own5, own6, own7, own8, \
                            own9, own10, own11, own12, own13, improve = True))

        # Se añade un título a la imagen
        imgs_title.append("Granada Mosaic")

        # Se muestran las imágenes que se leen inicialmente
        show_images(imgs, imgs_title, cols = 1, gray = False)

        input(continue_text)

    #
    ############################################################################

#
################################################################################

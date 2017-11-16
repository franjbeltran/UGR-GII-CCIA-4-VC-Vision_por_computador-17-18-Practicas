# -*- coding: utf-8 -*-
"""

Francisco Javier Caracuel Beltrán

VC - Visión por Computador

4º - GII - CCIA - ETSIIT - UGR

Curso 2017/2018

Trabajo 2 - Dirección de puntos relevantes y Construcción de panoramas

"""

import cv2
import numpy as np
import copy
import math
from matplotlib import pyplot as plt

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

def show_images(imgs, names = list(), cols = num_cols, title = ""):
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

        # La imagen se recibe con flotantes, por lo que se hace el cambio a
        # enteros
        img = copy.deepcopy(img).astype(int)

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

def generate_gaussian_pyr_imgs(img, n = 4, sigma = 0, sigma_down = 1, \
                                                border = cv2.BORDER_DEFAULT):
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

    # Se añaden tantas imágenes como se haya indicado
    for i in range(n):

        imgs.append(cv2.pyrDown(imgs[i], borderType=border))

    return imgs

#
################################################################################

################################################################################
# Apartado 1
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

    #return int(6*sigma+1)
    return 3

def get_ksize(sigma = 1):
    """
    Devuelve el parámetro de apertura (ksize) que se va a utilizar para
    obtener los puntos Harris. El valor se fija al valor correspondiente al uso
    de máscaras gaussianas de sigma 1. El tamaño de la máscara Gaussiana es
    6*1+1.
    """

    #return int(6*sigma+1)
    return 3

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
    Devuelve una lista con los máximos locales definidos por env y otra lista
    con los valores correspondientes.
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
    max_points_values = []

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
                    max_points.append((row-env, col-env))
                    max_points_values.append(harris[row-env, col-env])

    # Se devuelven las coordenadas de los puntos máximos locales y su valor
    return max_points, max_points_values

def get_harris(img, sigma_block_size = 1.5, sigma_ksize = 1, k = 0.04, \
                threshold = 3, env = 3, n = 500):
    """
    Obtiene una lista potencial de los puntos Harris de una imagen (img).
    Los valores de los parámetros utilizados dependen del sigma que se
    recibe (sigma_block_size, sigma_ksize).
    Recibe (k), (threshol) y (env) utilizados en las funciones creadas para la
    obtención de los puntos.
    Se puede especificar el número máximo de puntos Harris (n) que se quieren
    mostrar, siendo elegidos los (n) de mayor valor.
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
    max_points, max_points_values = not_max_suppression_harris(harris, \
                                                                threshold, env)

    # Se ordenan los puntos por el valor Harris de mayor a menor, guardando solo
    # los n (500) puntos con valor más alto
    index_max_points_sorted = np.argsort(max_points_values)[::-1]
    index_max_points_sorted = index_max_points_sorted[0:n]

    # Se guardan los puntos que se quieren mostrar rodeados por el círculo
    points_to_show = [max_points[i] for i in index_max_points_sorted]

    # Se devuelven los puntos que se quieren mostrar
    return points_to_show

def show_circles(img, points, sigma = 1):
    """
    Dada una imagen (img), pinta sobre ella unos círculos cuyo centro se
    encuentra en unas coordenadas dadas por unos puntos (points).
    El radio del punto viene dado por sigma (sigma).
    """

    # Se hace un copiado de la imagen para no modificar la original
    img = copy.deepcopy(img)

    # En points se tienen las coordenadas de los puntos que se van a rodear con
    #un círculo
    for point in points:
        cv2.circle(img, center=(point[1], point[0]), radius=sigma, color=1, \
                    thickness=2)

    # Se devuelve la imagen con los puntos
    return img

#
########

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

    # Listas para la visualización de las imágenes
    imgs = []
    imgs_title = []

    # Se cargan las imágenes a mostrar, se aplica el esquema de color, se
    # convierten los valores a flotantes para trabajar bien con ellos y se le
    # asigna un título
    img1 = cv2.imread(path+'Tablero1.jpg')
    img1 = set_c_map(img1, cmap)
    #img1 = img1.astype(float)
    img1 = np.float32(img1)
    img_title1 = "Tablero1"

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

    if 1 in ex:

        # Se añade la imagen original a la lista de imágenes
        imgs.append(img1)

        # Se añade un título a la imagen
        imgs_title.append(img_title1)

        # Se obtienen las imágenes de la pirámide Gaussiana
        img_gauss = generate_gaussian_pyr_imgs(img1)

        # Se recorre cada imagen y se le calculan los puntos Harris
        for i, img in enumerate(img_gauss):
            imgs.append(show_circles(img, get_harris(img)))
            imgs_title.append("Harris Level "+str(i+1))

        # Se muestran las imágenes que se leen inicialmente
        show_images(imgs, imgs_title)

        input(continue_text)

    #
    ########

    # Se vacían las listas con las imágenes y los nombres
    imgs.clear()
    imgs_title.clear()

#
################################################################################

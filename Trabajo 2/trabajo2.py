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

def filter_2d(signal, kernel):
    """
    Se encarga de hacer convolución entre un vector y un kernel 1D que se
    recibe por parámetro.
    El resultado es un vector del mismo tamaño que el vector señal recibido con
    la convolución realizada.
    """

    # Para realizar la convolución a todo el vector se debe ampliar lo
    # suficiente por los extremos para poder aplicar el kernel a todas las
    # posiciones.

    # Se guarda la longitud inicial del vector señal y del kernel.
    length_signal = len(signal)
    length_kernel = len(kernel)

    # Se guarda el tamaño que debe crecer en cada extremo. No interesa
    # redondear, solo truncar para no añadir más espacio del necesario.
    grow = (int)(length_kernel/2)

    # Lo primero es ampliar el vector que se recibe para que el kernel pueda
    # ser pasado por todas las posiciones.

    # Se crea un array del tamaño que debe crecer signal.
    l_init = np.ndarray(grow)
    l_end = np.ndarray(grow)

    # Se reflejan los valores iniciales de signal en el array que se concatena
    # al inicio
    l_init[:] = signal[grow-1::-1]

    # Se reflejan los valores finales de signal en el array que se concatena al
    # final
    l_end[:] = signal[-1:-grow-1:-1]

    # Se concatenan las listas creadas para hacer el espejo
    new_signal = np.concatenate((l_init,signal, l_end))

    # Al modificarse los valores del vector de entrada, se debe hacer una copia
    # para no modificar la imagen original
    conv = copy.deepcopy(signal)

    # Se debe recorrer cada posición del vector señal. Se comprueba cada casilla
    # del vector que tiene los bordes reflejados, de manera que se multiplica
    # cada segmento del vector señal con el kernel.
    # Cuando se termina la multiplicación, se deben sumar todos los valores
    # calculados y esta suma será el nuevo valor de la posición por la que se
    # está iterando
    for i in range(length_signal):
        res = 0

        for j in range(length_kernel):
            res += kernel[j] * new_signal[i+j]

        conv[i] = np.array([res])

    # Se devuelve el vector convolucionado
    return conv

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
                    max_points.append([(row-env, col-env), \
                                            harris[row-env, col-env]])

    # Se devuelven las coordenadas de los puntos máximos locales y su valor
    return max_points

def get_harris(img, sigma_block_size = 1.5, sigma_ksize = 1, k = 0.04, \
                threshold = -10, env = 5, scale = -1):
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
        max_points[i].append(scale)

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
        cv2.circle(img, center=(point[0][1], point[0][0]), \
                    radius=point[2]*radius, color=color1, thickness=2)

    # Si se indica que se dibujen las orientaciones
    if orientations:

        # Se recorren todos los puntos
        for point in points:

            # Se transforma la orientación del punto en un ángulo dado
            angle = point[3]/np.pi*180

            # Se calcula el primer punto de la línea
            pt1 = (point[0][1], point[0][0])

            # Se calcula el segundo punto de la línea utilizando el ángulo y
            # se debe multiplicar por el radio utilizado al mostrar el punto
            pt2 = (int(point[0][1]+np.sin(angle)*point[2]*radius), \
                    int(point[0][0]+np.cos(angle)*point[2]*radius))

            # Se pinta la línea del punto actual
            cv2.line(img, pt1, pt2, color2)

    # Se devuelve la imagen con los puntos
    return img

#
########

########
# Sección B
#

def refine_harris_points(img, points, env = 5, zero_zone = -1, \
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
    new_points = [point[0] for point in points]

    # Se refinan los puntos (deben ser float)
    cv2.cornerSubPix(img, np.float32(new_points), (env, env), \
                        (zero_zone, zero_zone), criteria)

    # Se modifican las coordenadas con los puntos refinados
    for i, point in enumerate(new_points):
        points[i][0] = point

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
    new_points = np.array([point[0] for point in points]).T

    # Se calcula la arcotangente de cada punto que devuelve la orientación que
    # tendra dicho punto. Esa operación se realiza para todos los puntos que se
    # tienen en el array
    orientations = np.arctan2(img_x, img_y)[new_points[0], new_points[1]]

    for point, orientation in zip(points, orientations):
        point.append(orientation)

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
    keypoints = [cv2.KeyPoint(point[0][0], point[0][1], _size=point[2], \
                    _angle=point[3]) for point in keypoints]

    # Se calculan los descriptores
    return sift.compute(img, keypoints)

#
########

#
################################################################################

################################################################################
# Apartado 2
#



#
################################################################################

"""
- Refinamiento: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
-
-
-
"""

################################################################################
# Ejecuciones
#

# En caso de utilizar este fichero de manera externa para el uso de sus
# funciones no se ejecutará ningún código
if __name__ == "__main__":

    # Se insertan los ejercicios que se quieren ejecutar
    ex = list()

    # 1.a:
    #ex.append(1)

    # 1.b:
    #ex.append(2)

    # 1.c:
    #ex.append(3)

    # 1.d:
    #ex.append(4)

    # 2:
    ex.append(5)

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

        # Se obtienen las imágenes de la pirámide Gaussiana indicando que se
        # mantenga el tamaño de la imagen original en las imágenes de los
        # distintos niveles de la pirámide
        img_gauss = generate_gaussian_pyr_imgs(img1, resize = True)

        # Se crea la lista de puntos Harris que se van a mostrar
        points = []

        # Se recorre cada imagen y se le calculan los puntos Harris
        for i, img in enumerate(img_gauss):
            points = points + get_harris(img, scale = i+1)

        # Se escogen los 1500 mejores
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

        points = refine_harris_points(img1, points)

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

        # Se actualizan los puntos con las orientaciones que tienen
        points500 = get_orientations(img1, points500, sigma=5, own = False)

        # Se añade la imagen a la lista de imágenes
        imgs.append(show_circles(img1, points500, orientations = True))

        # Se añade un título a la imagen
        imgs_title.append(img_title1+": 500 puntos")

        # Se escogen los 1000 mejores
        points1000 = get_best_harris(points, 1000)

        # Se actualizan los puntos con las orientaciones que tienen
        points1000 = get_orientations(img1, points1000, sigma=5, own = False)

        # Se añade la imagen a la lista de imágenes
        imgs.append(show_circles(img1, points1000, orientations = True))

        # Se añade un título a la imagen
        imgs_title.append(img_title1+": 1000 puntos")

        # Se escogen los 1500 mejores
        points1500 = get_best_harris(points, 1500)

        # Se actualizan los puntos con las orientaciones que tienen
        points1500 = get_orientations(img1, points1500, sigma=5, own = False)

        # Se añade la imagen a la lista de imágenes
        imgs.append(show_circles(img1, points1500, orientations = True))

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
    # Apartado 2
    #

    """
    Una función que sea capaz de representar varias imágenes con sus títulos en
    una misma ventana. Usar esta función en todos los demás apartados.
    """

    if 5 in ex:

        # Se añade la imagen original a la lista de imágenes
        imgs.append(img1)

        # Se añade un título a la imagen
        imgs_title.append(img_title1)



        # Se muestran las imágenes que se leen inicialmente
        show_images(imgs, imgs_title, cols = 2)

        input(continue_text)

    #
    ############################################################################

    # Se vacían las listas con las imágenes y los nombres
    imgs.clear()
    imgs_title.clear()

#
################################################################################

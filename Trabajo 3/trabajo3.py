# -*- coding: utf-8 -*-
"""

Francisco Javier Caracuel Beltrán

VC - Visión por Computador

4º - GII - CCIA - ETSIIT - UGR

Curso 2017/2018

Trabajo 3 - Indexación y Recuperacion de Imágenes

"""

import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import auxFunc as aux
from scipy import cluster as cluster_sci
import glob
import math
import os
from scipy import spatial

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

# Se indica que se muestre la longitud completa de las matrices
np.set_printoptions(threshold=np.nan)

# Se inicializa SIFT
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

def show_images(imgs, names = list(), cols = num_cols, title = "", gray = True,
                    cvt_color = True):
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
        if cvt_color:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img)

    # Se visualiza la tabla con todas las imágenes
    plt.show()

def get_keypoints_descriptors(img, own = False, mask = None):
    """
    Dada una imagen, calcula sus keypoints y descriptores y los devuelve.
    ---
    Añadido en el Trabajo 3: el parámetro mask permite indicar una máscara para
    la imagen en la obtención de los puntos SIFT.
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
        keypoints, descriptors = sift.detectAndCompute(img, mask)

    # Se devuelven los keypoints y descriptores calculados
    return keypoints, descriptors

def get_matches_knn(img1, img2, k = 2, ratio = 0.75, n = 1, flag = 2, \
                        get_data = False, improve = False, mask1 = None,
                        mask2 = None):
    """
    Dadas dos imágenes (img1, img2), calcula los keypoints y los descriptores
    para obtener los matches de ambas imágenes con la técnica
    "Lowe-Average-2NN", utilizando los k vecinos más cercanos (por defecto, 2).
    Para obtener los mejores matches, se calcula utilizando un ratio (ratio).
    Se puede indicar el porcentaje del número de matches que se quieren mostrar
    (n).
    El flag permite indicar si se quieren que se muestren los keypoints y los
    matches (0) o solo los matches (2).
    Si se indica get_data = True, se devolverán los keypoints, descriptores y
    los matches.
    Si se indica el flag "improve" como True, elegirá los mejores matches.
    Devuelve una imagen que se compone de las dos imágenes con sus matches.
    ---
    Añadido en el Trabajo 3: el parámetro mask1 y mask2 permite indicar una
    máscara para las imágenes 1 y 2 respectivamente en la obtención de los
    puntos SIFT.
    """

    # Se obtienen los keypoints y los descriptores de las dos imágenes
    keypoints1, descriptors1 = get_keypoints_descriptors(img1, mask = mask1)
    keypoints2, descriptors2 = get_keypoints_descriptors(img2, mask = mask2)

    # Se crea el objeto BFMatcher de OpenCV
    bf = cv2.BFMatcher()

    # Se consiguen los puntos con los que hace match indicando los vecinos más
    # cercanos con los que se hace la comprobación
    matches = bf.knnMatch(descriptors1, descriptors2, k)

    # Se mostrará el número máximo de matches
    n = int(len(matches)*n)

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
    matcher = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches, \
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
# Ejercicios
#

########
# Ejercicio 1
#

def match_imgs(img1, img2):
    """
    Empareja dos imágenes recibidas por parámetro.
    Para la primera imagen aparece una ventana donde seleccionar la zona que se
    quiere emparejar con la segunda imagen.
    El resultado es una imagen donde se muestran las correspondencias con los
    puntos SIFT de la zona seleccionada de la primera imagen y su
    correspondiente zona de la segunda imagen.
    """

    # Se extraen los puntos de la zona que se quiere emparejar con la función
    # dada en el fichero auxFunc.py
    points = np.array(aux.extractRegion(img1))

    shape = img1.shape

    # Con las dimensiones de la primera imagen se crea una nueva
    mask = np.zeros(shape, dtype = np.uint8)

    # Se recorren todos los puntos de la primera imagen. Con cada punto se hace
    # una llamada a la función de OpenCV pointPolygonTest() para que compruebe
    # si ese punto pertenece al polígono formado.
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):

            # La función devuelve un valor positivo si el punto que se comprueba
            # está dentro del contorno, 0 si es el contorno y negativo si está
            # fuera de él. Se cambia el valor a 255 (blanco) si es contorno o
            # está dentro
            #if cv2.pointPolygonTest(points, (x, y), False) >= 0:
                #mask[x, y] = 255

            mask[x,y] = int(cv2.pointPolygonTest(points,(y,x),False)==1)

    # Los pasos anteriores eran necesarios para crear una máscara. Una vez
    # creada la máscara se pueden obtener los puntos SIFT del rango que indica
    # dicha máscara en la primera imagen.
    # Se utilizan las funciones creadas en el Trabajo 2 para obtener los
    # keypoints y los descriptores, así como para obtener una imagen con los
    # emparejamientos de ambas imágenes.
    # Se han modificado las funciones del Trabajo 2 para poder indicar la
    # máscara que se debe utilizar.

    # Se debe convertir la máscara para que SIFT pueda operar con ella
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Para mostrar el polígono seleccionado
    imgs = []
    imgs.append(mask)
    show_images(imgs, cvt_color = False)

    # Se llama a la función del Trabajo 2 para obtener las correspondencias
    img_matches = get_matches_knn(img1=img1, img2=img2, mask1=mask, n = 0.5)

    return img_matches

#
########

########
# Ejercicio 2
#

def get_areas(dictionary, descriptors_and_patches, index = None, n_best = 20,
                                                        get_cluster = True):
    """
    Se recibe un vocabulario (dictionary) con muchas palabras que contienen
    trozos de imágenes. Con la lista de descriptores y parches
    (descriptors_and_patches) se agrupan los parches en sus respectivos clusters
    y eligen los (n) mejores teniendo en cuenta su centroide que se encuentra en
    el vocabulario. Se calcula la varianza de cada cluster con los (n_best)
    mejores parches y se ordenan de menor varianza a mejor. Se devuelven los
    clusters que se estén en la posición que indica "index".
    Si "get_cluster" es verdadero, "index" se refiere al índice del clúster,
    si es falso, se refiere al "index" mejor clúster.
    """

    # Se carga el fichero con los descriptores y parches
    descriptors, patches = aux.loadAux(descriptors_and_patches, True)

    # Se carga el diccionario recibido con la función que se encuentra en
    # auxFunc.py. Se devuelve la precisión, etiquetas y el vocabulario
    accuracy, labels, dictionary = aux.loadDictionary(dictionary)

    # Se crea un diccionario auxiliar para hacer las operaciones intermedias
    d = dict()

    # Se recorre cada descriptor
    for i, descriptor in enumerate(descriptors):

        # Se comprueba el clúster al que pertenece el descriptor
        label = labels[i][0]

        # Se calcula la distancia euclídea entre el descriptor y el centroide
        # de su clúster
        distance = get_distance(dictionary[label], descriptor)

        # Si el clúster aparece en el diccionario auxiliar se añade la posición
        # que ocupa en la lista de descriptores y la distancia con su centroide.
        # Si no existe se crea el contenedor y se añade.
        if label in d:
            d[label][0].append(i)
            d[label][1].append(distance)
        else:
            d[label] = [[i], [distance]]

    # Se crean tres estructuras para poder operar con los datos.
    # Lista que contiene los (best) mejores parches de cada clúster.
    best_patches = []
    # Lista que contiene las varianzas de cada clúster
    variances = []
    # Lista que contiene las etiquetas de los clusters
    clusters = []

    # Se recorren los clusters guardados en el diccionario auxiliar
    for cluster in d:

        # Se ordenan los parches del clúster según su varianza (de menor a
        # mayor varianza) y se guardan en "s" los "n_best" mejores
        s = np.argsort(np.array(d[cluster][1]))[0:n_best]

        # Se crea una lista donde se añadirán los mejores parches del clúster
        best = []

        # Se crea una lista para guardar la varianza de cada parche elegido en
        # el clúster
        var = []

        # Se guardan los mejores parches y sus varianzas
        for i in s:
            best.append(patches[d[cluster][0][i]])
            var.append(d[cluster][1][i])

        # Se guardan los "best" mejores parches en la lista que contiene todos
        # los clusters
        best_patches.append(best)

        # Se guarda la varianza de los "best" mejores parches
        variances.append(np.var(best))

        # Se guarda la etiqueta del clúster
        clusters.append(cluster)

    # Se ordenan los clúster por su varianza de menor a mayor
    s = np.argsort(np.array(variances))

    """
    Para mostrar una lista con todas las imágenes y sus datos asociados:
    """
    """

    for i in reversed(range(len(clusters)-100, len(clusters))):

        print("Cluster: ",clusters[s[i]], ", Iteración: ", i, ", Varianza: ", \
                                                            variances[s[i]])
        show_images(best_patches[s[i]])

    """

    # Se crea la lista con los clusters que se quieren devolver
    imgs = []

    # Se guardan los índices de mejor a peor
    if not get_cluster:

        # Se recorren los índices que se quieren devolver
        for i in index:

            # Se guarda el clúster de "best" parches
            imgs.append((clusters[s[i]], best_patches[s[i]]))

    else:
    # Se guardan los clusters

        # Se recorren los índices que se quieren devolver
        for i in index:

            # Se guarda el clúster de "best" parches
            imgs.append((clusters[s[i]], best_patches[clusters.index(i)]))

    # Se devuelven los clusters
    return imgs

def get_distance(v1, v2):
    """
    Calcula la distancia euclídea entre dos vectores y devuelve el resultado
    """

    # Se guarda la longitud de los vectores (deben ser de igual tamaño)
    """length = len(v1)

    # Se inicializa la distancia
    distance = 0

    # Se recorren los vectores y se calcula su distancia
    for i in range(0, length):

        distance += abs(v1[i]-v2[i])"""

    distance = spatial.distance.euclidean(v1, v2)

    return distance

#
########

########
# Ejercicio 3
#

def load_imgs(path, ext = "png", not_find = ""):
    """
    Devuelve un vector con todas las imágenes que se encuentran en la ruta
    indicada.
    Se guardan los ficheros que tengan la extensión (ext).
    """

    # Contenedor donde se añadirán todas las imágenes que se encuentran en la
    # ruta obtenida por parámetro
    db = []

    # Se
    for filename in glob.glob(path+'*.'+ext):

        # Se añaden las imágenes que estén en el directorio y no sea las
        # recibidas en "not_find"
        if not os.path.basename(filename).startswith(not_find):
            img = cv2.imread(filename)
            img = np.uint8(img)

        db.append(img)

    return db

def get_same_imgs(img, imgs, dictionary, n = 5):
    """
    Función que recibe una imagen (img) y devuelve en un vector las (n)
    imágenes (imgs) con mayor coincidencia de entre todas las disponibles.
    imgs contiene todas las imágenes que forman parte de la base de datos.
    dictionary tendrá la ruta+nombre del fichero con el vocabulario.
    """

    # Se carga el diccionario recibido con la función que se encuentra en
    # auxFunc.py. Se devuelve la precisión, etiquetas y el vocabulario
    accuracy, labels, dictionary = aux.loadDictionary(dictionary)

    # Primero se obtienen los descriptores de la imagen
    keypoints, descriptors = get_keypoints_descriptors(img)

    # Se calcula el histograma de la imagen-pregunta
    histogram_query = get_histogram(dictionary, descriptors)

    # Se crea un contenedor donde se almacenará cada resultado al comparar el
    # histograma de la imagen-pregunta
    res_histograms = []

    # Se calcula el histograma de cada imagen que forma parte de la base de
    # datos y se compara el histograma con el de la imagen-pregunta
    for img_aux in imgs:

        # Se obtienen los descriptores de la imagen actual
        keypoints, descriptors = get_keypoints_descriptors(img_aux)

        # Se guarda el histograma correspondiente de cada imagen
        histogram = get_histogram(dictionary, descriptors)

        # Se compara el histograma de la imagen-pregunta con cada histograma de
        # las imágenes de la base de datos
        value_compare = compare_histograms(histogram_query, histogram)

        # Se añaden los resultados al contenedor
        res_histograms.append(value_compare)

    # Para ordenar los mejores resultados y obtener los índices, se utiliza
    # np.argsort(). sorted_res contiene una lista
    sorted_res = np.argsort(np.array(res_histograms))

    # Se quieren elegir las n mejores imágenes, por lo que se seleccionan las
    # n últimas (está ordenado de peor a mejor)
    best = sorted_res[:len(sorted_res)-n-1:-1]

    # Se recorren los índices y se devuelve una lista con las mejores imágenes
    best_imgs = []

    for i in best:
        best_imgs.append(imgs[i])

    return best_imgs


def get_histogram(dictionary, descriptors):
    """
    Calcula un histograma con el tamaño del número de posiciones del
    diccionario, teniendo en cada posición el número de veces que ha sido el
    mejor valor cada palabra de dicha posición con los descriptores recibidos.
    """

    # Se crea un histograma con todas las posiciones inicializadas a 0
    histogram = np.zeros(len(dictionary))

    # Función utilizada para calcular la función euclídea de cada descriptor
    # con el diccionario. Utilizada gracias a Ismael Sánchez García.
    # Devuelve dos vectores de tamaño igual al número de descriptores.
    # El primer vector contiene el índice de la mejor palabra que le corresponde
    # al descriptor que ocupa esa misma posición.
    # El segundo vector contiene la distancia entre el descriptor y la palabra
    # del diccionario
    words, distances = cluster_sci.vq.vq(descriptors, dictionary)

    # Se recorren los índices obtenidos y se suma 1 en su posición
    # correspondiente
    for word in words:
        histogram[word] += 1

    # Se devuelve el histograma
    return histogram

def compare_histograms(hist1, hist2):
    """
    Compara dos histogramas recibidos por parámetro "hist1", "hist2" y devuelve
    el valor entre ambos
    """

    # Se guarda la longitud del histograma 1 (los dos deben tener el mismo
    # tamaño)
    length = len(hist1)

    # Se inicializan los valores que se van a utilizar en la sumatoria
    # dq -> Sum i=1:V (dj(i)*q(i))
    dq = 0

    # d -> sqrt(Sum i=1:V (dj(i)^2))
    d = 0

    # q -> sqrt(Sum i=1:V (q(i)^2))
    q = 0

    # Se recorren los histogramas
    for i in range(0, length):

        # dq -> Sum i=1:V (dj(i)*q(i))
        dq += hist1[i]*hist2[i]

        # d -> sqrt(Sum i=1:V (dj(i)^2))
        d += hist1[i]*hist1[i]

        # q -> sqrt(Sum i=1:V (q(i)^2))
        q += hist2[i]*hist2[i]

    # Se devuelve el resultado
    return dq / (math.sqrt(d)*math.sqrt(q))

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

    # 1:
    ex.append(1)

    # 2:
    ex.append(2)

    # 3:
    ex.append(3)

    # Listas para la visualización de las imágenes
    imgs = []
    imgs_title = []

    # Se cargan las imágenes a mostrar, se aplica el esquema de color, se
    # convierten los valores a flotantes para trabajar bien con ellos y se le
    # asigna un título
    img1a = cv2.imread(path+'407.png')
    #img1a = set_c_map(img1a, cmap)
    img1a = np.uint8(img1a)
    img_title1a = "407.png"

    img1b = cv2.imread(path+'408.png')
    #img1b = set_c_map(img1b, cmap)
    img1b = np.uint8(img1b)
    img_title1b = "408.png"

    img2a = cv2.imread(path+'36.png')
    #img2a = set_c_map(img2a, cmap)
    img2a = np.uint8(img2a)
    img_title2a = "36.png"

    img2b = cv2.imread(path+'50.png')
    #img2b = set_c_map(img2b, cmap)
    img2b = np.uint8(img2b)
    img_title2b = "50.png"

    img3a = cv2.imread(path+'422.png')
    #img3a = set_c_map(img3a, cmap)
    img3a = np.uint8(img3a)
    img_title3a = "422.png"

    img3b = cv2.imread(path+'425.png')
    #img3b = set_c_map(img3b, cmap)
    img3b = np.uint8(img3b)
    img_title3b = "425.png"

    img4 = cv2.imread(path+'266.png')
    #img4 = set_c_map(img4, cmap)
    img4 = np.uint8(img4)
    img_title4 = "266.png"

    img5 = cv2.imread(path+'364.png')
    #img5 = set_c_map(img5, cmap)
    img5 = np.uint8(img5)
    img_title5 = "364.png"

    img6 = cv2.imread(path+'62.png')
    #img6 = set_c_map(img6, cmap)
    img6 = np.uint8(img6)
    img_title6 = "62.png"

    # Se guardan los nombres de los diccionarios
    dictionary500 = path+"kmeanscenters500.pkl"
    dictionary1000 = path+"kmeanscenters1000.pkl"
    dictionary5000 = path+"kmeanscenters5000.pkl"

    # Se guarda el nombre del fichero que contiene los descriptores y parches
    descriptors_and_patches = path+"descriptorsAndpatches.pkl"

    ############################################################################
    # Ejercicio 1
    #

    """
    Emparejamiento de descriptores
    """

    if 1 in ex:

        # Se añaden las dos imágenes que se van a utilizar
        imgs.append(img1a)
        imgs.append(img1b)

        imgs_title.append(img_title1a)
        imgs_title.append(img_title1b)

        # Se muestran las imágenes antes de mostrar su emparejamiento con los
        # puntos SIFT
        #show_images(imgs, imgs_title)

        # Se vacían las listas con las imágenes y los nombres
        imgs.clear()
        imgs_title.clear()

        # Se obtiene la imagen con el emparejamiento de los puntos SIFT
        img_res1 = match_imgs(img1a, img1b)

        img_title_res1 = "Pareja 1"

        imgs.append(img_res1)
        imgs_title.append(img_title_res1)

        show_images(imgs, imgs_title)

        input(continue_text)

        # Se vacían las listas con las imágenes y los nombres
        imgs.clear()
        imgs_title.clear()

        # Se obtiene la imagen con el emparejamiento de los puntos SIFT
        img_res2 = match_imgs(img2a, img2b)

        img_title_res2 = "Pareja 2"

        imgs.append(img_res2)
        imgs_title.append(img_title_res2)

        show_images(imgs, imgs_title)

        input(continue_text)

        # Se vacían las listas con las imágenes y los nombres
        imgs.clear()
        imgs_title.clear()

        # Se obtiene la imagen con el emparejamiento de los puntos SIFT
        img_res3 = match_imgs(img3a, img3b)

        img_title_res3 = "Pareja 3"

        imgs.append(img_res3)
        imgs_title.append(img_title_res3)

        show_images(imgs, imgs_title)

        input(continue_text)

    #
    ########

    # Se vacían las listas con las imágenes y los nombres
    imgs.clear()
    imgs_title.clear()

    #
    ############################################################################

    ############################################################################
    # Ejercicio 2
    #

    """
    Visualización del vocabulario
    """

    if 2 in ex:

        # Se indican las posiciones de los clusters que se quieren mostrar.
        # Serán dos buenos y dos malos
        index = (555, 43, 61, 687)

        # Se obtienen las regiones que se han indicado
        clusters = get_areas(dictionary1000, descriptors_and_patches, index)

        # Se muestran las regiones obtenidas
        for cluster in clusters:
            imgs_title.append("Clúster "+str(cluster[0]))
            show_images(cluster[1], cvt_color = False, cols = 5)
            imgs_title.clear()

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
    Recuperación de imágenes
    """

    if 3 in ex:

        # Se muestra la imagen-pregunta
        imgs.append(img4)
        imgs_title.append(img_title4)

        show_images(imgs, imgs_title)

        # Se obtienen las imágenes disponibles, excepto la que se va a enviar
        db = load_imgs(path, "png", "266")

        # Se devuelven las n (5) mejores imágenes
        best_imgs =  get_same_imgs(img4, db, dictionary5000, 5)

        # Se muestran las mejores imágenes
        show_images(best_imgs)

        # Se limpian los contenedores de las imágenes
        imgs.clear()
        imgs_title.clear()

        input(continue_text)

        # Se muestra la imagen-pregunta
        imgs.append(img5)
        imgs_title.append(img_title5)

        show_images(imgs, imgs_title)

        # Se obtienen las imágenes disponibles, excepto la que se va a enviar
        db = load_imgs(path, "png", "364")

        # Se devuelven las n (5) mejores imágenes
        best_imgs =  get_same_imgs(img5, db, dictionary5000, 5)

        # Se muestran las mejores imágenes
        show_images(best_imgs)

        # Se limpian los contenedores de las imágenes
        imgs.clear()
        imgs_title.clear()

        input(continue_text)

        # Se muestra la imagen-pregunta
        imgs.append(img6)
        imgs_title.append(img_title6)

        show_images(imgs, imgs_title)

        # Se obtienen las imágenes disponibles, excepto la que se va a enviar
        db = load_imgs(path, "png", "62")

        # Se devuelven las n (5) mejores imágenes
        best_imgs =  get_same_imgs(img6, db, dictionary5000, 5)

        # Se muestran las mejores imágenes
        show_images(best_imgs)

        input(continue_text)

    #
    ############################################################################

#
################################################################################

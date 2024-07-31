import cv2 as cv
import numpy as np
from sensors_class import Sensors    
import math
from sklearn.cluster import DBSCAN
import sys

def sobel(src):
    window_name = ('Sobel Demo - Simple Edge Detector')
    scale = 1
    delta = 0
    ddepth = cv.CV_16S

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + src)
        return -1
 
    
    grad_x = cv.Sobel(src, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
 
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(src, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    
 
    
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    
 
    
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    print(grad)

    edges_sobel = cv.Canny(grad,75,150)

    cv.imshow(window_name, grad)



def color_detection(image) -> tuple:
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Usar apenas o canal Hue para a detecção de cores
    hue_channel = hsv[:, :, 0]

    blur = cv.GaussianBlur(hsv, (5, 5), 0)

    # Limiares para a máscara vermelha
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Limiares para a máscara amarela
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    maskyellow = cv.inRange(blur,lower_yellow,upper_yellow)


    maskred1 = cv.inRange(blur,lower_red1,upper_red1)
    maskred2 = cv.inRange(blur,lower_red2,upper_red2)

    # Combina as duas máscaras vermelhas
    mask_red = cv.bitwise_or(maskred1, maskred2)

    

    # Aplicar a máscara na imagem original
    result_yellow = cv.bitwise_and(image, image, mask=maskyellow)
    result_red = cv.bitwise_and(image, image, mask=mask_red)
    
    return result_yellow, result_red


max_width = 500  # Largura máxima das linhas
max_height = 500  # Altura máxima das linhas

def main(argv):
    #Exige os argumentos
    if len(argv) < 1:
        print ('Not enough parameters')
        print ('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1

    image = cv.imread(argv[0])
    copy = image.copy()

    stretch_near = cv.resize(image, (1200, 720),  
               interpolation = cv.INTER_NEAREST) 
    
    stretch_copy = cv.resize(copy, (780, 540),  
               interpolation = cv.INTER_NEAREST)

    # sobel(stretch_copy)

    gray = cv.cvtColor(stretch_near, cv.COLOR_BGR2GRAY)
    blur  = cv.GaussianBlur(gray, (5, 5), 0)
    blur2 = cv.GaussianBlur(stretch_copy,(5,5),0)
    
    edges = cv.Canny(blur2,75,150)
    
    #Faz as máscaras vermelha e amarela
    mask_yellow,mask_red = color_detection(stretch_copy)
   
    ############################################################################################
    ######################################### AMARELO  #########################################
    ############################################################################################

    edges_yellow = cv.Canny(mask_yellow,75,150)


    ############################################################################################
    #######################################  VERMELHO   ########################################
    ############################################################################################
    edges_red = cv.Canny(mask_red,75,150)

    
    contornos, _ = cv.findContours(edges_red, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    

    # Coletar pontos dos contornos
    points = []
    for contour in contornos:
        for point in contour:
            points.append(point[0])

    points = np.array(points)

    # Ordenar os pontos por coordenada X e, em caso de empate, pela coordenada Y
    sorted_points = sorted(points, key=lambda point: (point[0], point[1]))

    # Converter a lista de pontos ordenados de volta para um array numpy com np.vstack
    sorted_points_np = np.vstack(sorted_points)

    # Exibir os pontos ordenados
    print("Pontos ordenados:")
    print(sorted_points_np)

    # for i in range(len(sorted_points_np)):

    #     concatenated_contours = np.vstack(contornos)

    #     # Ordenar os pontos
    #     sorted_points = sorted(concatenated_contours, key=lambda point: (point[0][0], point[0][1]))

    #     # Converter a lista de pontos ordenados de volta para um array numpy com np.vstack
    #     sorted_points_np = np.vstack(sorted_points)
    #     print(sorted_points_np)

    ############################################################################################

    # pontos_ordenados = []

    # distances = []
    # pontos_ordenados.append(concatenated_contours[0])
    # for i in range(len(concatenated_contours)):
    #     for j in range(i+1,len(concatenated_contours)):
    #         distance = np.linalg.norm(concatenated_contours[i] - concatenated_contours[j])
    #         print(f"Ponto na posição {i} - Pontos na posição {j} = {distance}")
    #         distances.append(distance)
    #     #Pega o índice da distância mais próxima
    #     mais_proximo = np.argmin(distances)
    #     print(mais_proximo)
    #     distances.clear()
        
        # pontos_ordenados.append(concatenated_contours[mais_proximo])
    
    ############################################################################################
    
    # for contour in contornos:
    #     a = 0.02 *cv.arcLength(contour,True)
    #     approx = cv.approxPolyDP(contour,a,True)
    #     print(approx)
    #     if len(approx) == 15:
    #         cv.drawContours(stretch_near,[approx],0,(0,255,0),2)
    
    cv.imshow("edges",edges)


    # cv.imshow("Canny edges",stretch_near)
    cv.imshow("Edges Yellow",edges_yellow)
    cv.imshow("Edges Red",edges_red)
    cv.drawContours(stretch_copy, [sorted_points_np], -1, (0, 255, 0), 3)
    cv.imshow("HOUGH", stretch_copy)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv[1:])
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
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 70, 50])
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
    """
    if len(argv) < 1:
        print ('Not enough parameters')
        print ('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1
    """
    video = cv.VideoCapture(argv[0])
    # image = cv.VideoCapture(argv[0])


    while True:
        status,frame = video.read()
        copy = frame.copy()

        stretch_near = cv.resize(frame, (1200, 720),  
                interpolation = cv.INTER_NEAREST) 
        
        stretch_copy = cv.resize(copy, (780, 540),  
                interpolation = cv.INTER_NEAREST)

        # sobel(stretch_copy)

        gray = cv.cvtColor(stretch_near, cv.COLOR_BGR2GRAY)
        blur  = cv.GaussianBlur(gray, (5, 5), 0)
        blur2 = cv.GaussianBlur(stretch_copy,(5,5),0)
        
        edges = cv.Canny(blur2,200,500)

        # Detectar linhas usando a Transformada de Hough
        linhas = cv.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)

        # Desenhar as linhas detectadas
        if linhas is not None:
            for linha in linhas:
                x1, y1, x2, y2 = linha[0]
                cv.line(stretch_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
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

        
        contornos, hierarquia = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        
        for i, contour in enumerate(contornos):
            # Identifica o objeto mais próximo com base no maior contorno (suposição: objeto maior está mais próximo)
            contorno_maior = max(contornos, key=cv.contourArea)
            print(f"Contorno {i}: {contour}")
            # Verifica a hierarquia: hierarquia[0][i] contém [next, previous, first_child, parent]
            # Se parent == -1, significa que este contorno é de um objeto separado
            if hierarquia[0][i][3] == -1:
                cv.drawContours(stretch_copy, [contour], -1, (0, 255, 0), 2)         
            # for point in contour:
            #     points.append(point[0])

       
    
        # for i in range(len(sorted_points_np)):

        #     concatenated_contours = np.vstack(contornos)

        #     # Ordenar os pontos
        #     sorted_points = sorted(concatenated_contours, key=lambda point: (point[0][0], point[0][1]))

        #     # Converter a lista de pontos ordenados de volta para um array numpy com np.vstack
        #     sorted_points_np = np.vstack(sorted_points)
        #     print(sorted_points_np)


        cv.imshow("edges",edges)


        # cv.imshow("Canny edges",stretch_near)
        cv.imshow("Edges Yellow",edges_yellow)
        cv.imshow("Edges Red",edges_red)
        cv.imshow("FRAME", stretch_copy)
        
        key = cv.waitKey(25)
        if key == 27:
            break

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv[1:])
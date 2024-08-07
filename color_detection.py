import cv2 as cv
import numpy as np
from sensors_class import Sensors    
import math
    

# Função para encontrar o maior segmento de linha
def find_longest_line(image):
    # Aplicar o filtro de Canny
    edges = cv.Canny(image, 50, 100, apertureSize=3)
    
    # Detectar linhas usando a Transformada de Hough
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return None, 0  # Retorna None se nenhuma linha for detectada
    
    # Inicializar variáveis para a linha mais longa
    longest_line = None
    max_length = 0
    
    # Iterar sobre todas as linhas detectadas
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        if length > max_length:
            max_length = length
            longest_line = (x1, y1, x2, y2)
    
    return longest_line, max_length




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

def hough(lines,edges,image,plot):
    cdstP = np.copy(edges)
    """
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(edges, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    """

    contornos, _ = cv.findContours(cdstP, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
   
    borders = []

    for contorno in contornos:
        approx = cv.approxPolyDP(contorno, 0.02 * cv.arcLength(contorno, True), True)
        if len(approx) == 1:  # Verifica se o contorno tem 4 vértices (retângulo)
            (x, y, w, h) = cv.boundingRect(approx)  #Pega x,y, altura e largura dos retângulos]
            cv.rectangle(plot, (x, y), (x + w, y + h), (0, 0, 255), 2)  
   
    linesP = cv.HoughLines(edges, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(plot, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
   
    cv.imshow("Linhas detectadas",plot)

def main():
    image = cv.imread('./images/red1.jpg')
    video = cv.VideoCapture(2)
    
    while True:
        status,frame = video.read()


        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur  = cv.GaussianBlur(gray, (5, 5), 0)


        cv.imshow("Imagem", frame)


        #edges_yellow = cv.Canny(mask_yellow,75,150)

        #edges_red = cv.Canny(mask_red,75,150)

        edges = cv.Canny(blur,75,150)

        lines = cv.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)
    
        hough(lines,edges,blur,frame)

        
        cv.imshow("Canny edges",edges)
        # cv.imshow("Edges Yellow",edges_yellow)
        # cv.imshow("Edges Red",edges_red)
     
        key = cv.waitKey(25)
        if key == 27:
            break

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
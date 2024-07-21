import cv2 as cv
import numpy as np
from sensors_class import Sensors
import math
import argparse
from pyzbar.pyzbar import decode
from PIL import Image 


######################################################################
# GLOBAIS
######################################################################

# ALTURA_REAL_CENTRO_RUA_CM = 4
# LARGURA_REAL_CENTRO_RUA_CM = 2.1
# LARGURA_RUA = 20

#####################################################################
#Fazer a detecção de  ângulos dos retângulos utilizando cv2.boxPoints() 

first_rectangle_vertices = None


def _qrcode(cam) -> bool:  #Processa imagem buscando um qrcode, retorna o conteúdo do qrcode.
    img = (cam)
    return decode(img) 


def qrcode_detection(cam):
    qr_detected = _qrcode(cam)
    if qr_detected:
        prox = None
        id = 0
        for i in range(len(qr_detected)):
            _, x, y, _ = frame_center_detection(cam)
            x_, y_, w, h = qr_detected[i].rect
            if not prox:
                prox = np.linalg.norm([y_ - y, x_ - x])
                id = i
            else:
                ndist = np.linalg.norm([y_ - y, x_ - x])
                if ndist < prox:
                    prox = ndist
                    id = i
        
        barCode = str(qr_detected[id].data.decode("utf-8"))
        pos = qr_detected[id].rect
        return barCode, pos
    return None, None

    
def show_qrcode(frame,pos):
    if pos:
        (left,top,width,height) = pos
        top_left_corner = (left,top)
        top_right_corner = (left+width,top)
        bottom_right_corner = (left+width,top+height)
        bottom_left_corner = (left,top+height)
        
        print(f"Corner esquerdo cima: {top_left_corner}")
        print(f"Corner direito baixo:{top_right_corner}")
        print(f"Corner direito baixo:{bottom_right_corner}")
        print(f"Corner direito baixo:{bottom_left_corner}")
        
        cv.line(frame,(top_left_corner),(top_right_corner),(255,0,0),5)
        cv.line(frame,(top_left_corner),(bottom_left_corner),(255,0,0),5)
        cv.line(frame,(bottom_left_corner),(bottom_right_corner),(255,0,0),5)
        cv.line(frame,(top_right_corner),(bottom_right_corner),(255,0,0),5)
        

    else:
        return None

    #cv.rectangle(frame,(384,0),(510,128),(0,255,0),3)

def frame_center_detection(frame):
    # Obter as dimensões da imagem
    altura, largura, _ = frame.shape
    

    center_x = int(largura/2)
    center_y = int(altura/2)
    
    center = (center_x,center_y)
    radius = 2
    color = (255,0,0)
    thickness = 2    
    
    draw_circle = cv.circle(frame,center,radius,color,thickness)
    
    return draw_circle,center_x,center_y,largura

def distance_of_points(frame,xa,ya,xb,yb):
    distance_center_fr = math.sqrt((xb-xa)**2+(yb-ya)**2)  #Distância do pixel central para o retângulo frontal (Mais distante)
    
    return distance_center_fr

def detect_rectangles(frame):        
    #Aplicando blur Gaussiano na imagem
    blur = cv.GaussianBlur(frame, (5, 5), 0)
    
    #hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    
    #Limiar de cor preta em bgr
    lower = np.array([0,0,0], dtype = "uint8")
    upper = np.array([59,59,59], dtype = "uint8")

    #Limiar de cor preta 2 em bgr
    lower2= np.array([0,0,0], dtype = "uint8")
    upper2 = np.array([179,0,100], dtype = "uint8")


    #A máscara não é essenciail porém elas ajudam a ter certeza de que a imagem será filtrada com as cores certas da rua
    mask = cv.inRange(blur, lower,upper)
    
    #Aplicando as máscaras com base nos limiares
    
    # mask2 = cv.inRange(blur,lower2,upper2)
    # mask = 255 - mask
    # #Somando as máscaras
    # mask = mask + mask2
    
    #Identificando as arestas presentes na imagem de acordo com as máscaras (Filtro de Canny)
    edges = cv.Canny(mask, 75, 150)
    cv.imshow("Edges", edges)
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    height, width = frame.shape[:2]
    center_x = width // 2
    max_width = 100  # Largura máxima dos retângulos
    max_height = 100  # Altura máxima dos retângulos

    rectangles = []
        
        
    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.02 * cv.arcLength(contour, True), True)
        
        if len(approx) == 4:  # Verifica se o contorno tem 4 vértices (retângulo)
            (x, y, w, h) = cv.boundingRect(approx)  #Pega x,y, altura e largura dos retângulos
            #box = calculating_box(approx)

            aspect_ratio = float(w) / h
            # Verifica se o retângulo está no centro e se é pequeno
            if 0.2 < aspect_ratio < 5.0 and w < max_width and h < max_height and h > w:
                #rect_center_x = x + w // 2
                #if abs(rect_center_x - center_x) < width * 0.2:  # Verifica se o retângulo está no centro
                rectangles.append((x, y, w, h))
                
                #print(f"Altura = {h} / Largura = {w}")
    # cv.imshow("Edges", edges)
    
    return rectangles

#Ângulo entre o frame central e o retângulo frontal
def angle_between_fc_fr(x1,y1,x2,y2): 
    deltax = x2-x1
    deltay = y2-y1
    
    if deltax==0:
        return 0
    
    #Calculando o coeficiente angular
    angular_coef = deltay/deltax
    #Calculando a arctg do coeficiente
    arctg = np.arctan(angular_coef)
    
    #Definindo o ângulo em graus
    angulo_graus = math.degrees(arctg)
    
    
    return angulo_graus

def calculating_box(approx):
    rect = cv.minAreaRect(approx)           #Desenha o retângulo de área mínima, com a angulação correta
    box = cv.boxPoints(rect)                #Pega os pontos dos vértices do primeiro e último retângulo
    return box



def pick_up_rectangles(rectangles):
    if not rectangles:
        return None, None
    
    # Ordena os retângulos com base na coordenada y
    rectangles.sort(key=lambda rect: rect[1])
    
    # Primeiro e último retângulo
    first_rectangle = rectangles[0]
    print(f"Primeiro retângulo: {first_rectangle}")
    last_rectangle = rectangles[-1]
    
    return first_rectangle, last_rectangle

def converting_px_meters(pixel,h,altura_centro):
    cm = (pixel*altura_centro)/h   # h=67  nesse caso (tamanho dos 4 cm de altura do centro da pista)
    return cm

def converting_centimeters_px(cm,h_qr,qr_code_dimension):
    pxs = (cm*qr_code_dimension)/h_qr   
    return pxs

#Calcula o ângulo baseado no vértice superior esquerdo e inferior direito
def calcular_angulo(vse,vid):
    x1,y1 = vse
    x2,y2 = vid

    #Calcular o vetor da diagonal
    dx = x2-x1
    dy = y2-y1

    # Calcular o ângulo da diagonal com o eixo horizontal
    angulo_diagonal = math.degrees(math.atan2(dy, dx))

    # Calcular o ângulo do retângulo em relação ao eixo horizontal
    angulo_retangulo = angulo_diagonal - 45

    return angulo_retangulo

def main():
    video = cv.VideoCapture(0)
    sensors = Sensors()
    

    # Cria um analisador de argumentos
    parser = argparse.ArgumentParser(description="Calcular a distância.")

    # Adiciona um argumento chamado "centerheight"
    parser.add_argument('--centerheight', type=float, required=True, help="Altura do centro da rua")
    # Analisa os argumentos fornecidos pelo usuário
    args = parser.parse_args()
    altura_real_centro_da_rua = args.centerheight
    print(altura_real_centro_da_rua)

    
    while True:
        status, frame = video.read()
        new_frame = frame.copy()
        

        
        if not status:
            video = cv.VideoCapture("test_video/street2.mp4")
            continue
        
        #Definindo as coordenadas do frame central
        (_,xb,yb,largura) = frame_center_detection(new_frame)     

        (_,posqr) = qrcode_detection(new_frame)

        show_qrcode(new_frame,posqr)

        rectangles = detect_rectangles(frame)
        first_rectangle, last_rectangle = pick_up_rectangles(rectangles)
       

        #Faz o primeiro e último retângulos e retorna a distância
        if first_rectangle:

            ######################################################################################################
            #PRINTS
            ######################################################################################################
            x, y, w, h = first_rectangle                                    #Atribui os valores ao primeiro retângulo
            center_x, center_y = x + w // 2, y + h // 2                     #Definindo o meio do primeiro retângulo
            
            distancia_meio_retangulo_frontal = distance_of_points(frame,center_x,center_y,xb,yb)    #Em pixels
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)     #DESENHANDO O RETÂNGULO
            meters_to_fly = round(converting_px_meters(distancia_meio_retangulo_frontal,h,altura_real_centro_da_rua), 2)
            print(h,altura_real_centro_da_rua)
            
            ######################################################################################################
            #MEDIÇÃO DOS VÉRTICES DO PRIMEIRO RETÂNGULO
            ######################################################################################################

            #Vértice superior esquerdo
            sup_esq = (x,y)
            inf_dir = (x+w,y+h)
            
            angulo_do_primeiro_retangulo = calcular_angulo(sup_esq,inf_dir)
            

            #CALCULANDO A DISTÂNCIA
            meters_rectangle = converting_px_meters(h,h,altura_real_centro_da_rua)
            cv.putText(frame,str(meters_rectangle),(x, y+h+15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255),1)
            xa, ya, wa, ha = last_rectangle
            center_x_last, center_y_last = xa + wa // 2, ya + ha // 2                     #Definindo o meio do primeiro retângulo
            cv.rectangle(frame, (xa, ya), (xa + wa, ya + ha), (0, 0, 255), 2)             
        
            ######################################################################################################
            #DISTÂNCIAS
            ######################################################################################################

            
            distance_between_rectangles_center = round(distance_of_points(frame,center_x,center_y,center_x_last,center_y_last),2)
            distance_between_rectangles_y = center_y_last-center_y          #Distância entre os retângulos em pixel
            converted_y_rectangles = round(converting_px_meters(distance_between_rectangles_y,h,altura_real_centro_da_rua),2)

            ######################################################################################################
            #PRINTS
            ######################################################################################################
            print(f"Ângulo do primeiro retângulo: {angulo_do_primeiro_retangulo}")
            print(f"Distância entre a primeira e a última marcação: {distance_between_rectangles_center} pixels")
            print(f"Distância entre os retângulos em Y : {distance_between_rectangles_y} pixels || {converted_y_rectangles} centímetros")

            
            print(f"Distância para deslocamento : {meters_to_fly} cm")
            angle_correction = round(angle_between_fc_fr(center_x,center_y,xb,yb),2)
            print(f"Ângulo de correção: {angle_correction}°\n============================================================================\n")
            
        
        sensors.setImage(frame)
        sensors.processImage()
        sensors.show()

        cv.imshow("Video", frame)
        cv.imshow("Copia", new_frame)
        
        key = cv.waitKey(25)
        if key == 27:
            break
        
    
    
    video.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
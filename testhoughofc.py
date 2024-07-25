"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np

def detect_rectangles(frame):        
    #Aplicando blur Gaussiano na imagem
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    
    #Identificando as arestas presentes na imagem de acordo com as máscaras (Filtro de Canny)
    edges = cv.Canny(blur, 50, 200)
    cv.imshow("Edges", edges)
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    height, width = frame.shape[:2]
    center_x = width // 2
    max_width = 50  # Largura máxima dos retângulos
    max_height = 50  # Altura máxima dos retângulos

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
                #for i in range(rectangles):
                
                #print(f"Altura = {h} / Largura = {w}")
    # cv.imshow("Edges", edges)
    
    return rectangles

def calculate_centers(rectangles):
    centers = []
    for x, y, w, h in rectangles:
        center_x = x + w // 2
        centers.append(center_x)
        print(f"Centros:{centers}")
    return centers

def calculate_tam(rectangles):
    tams = []
    for x, y, w, h in rectangles:
        altura = h
        tams.append(altura)
    return tams


def filter_rectangles_by_center(rectangles, centers,tamanho):
    if not centers:
        return []

    mean_center = np.mean(centers)
    mean_height = np.mean(tamanho)

    print(f"Média: {mean_center}")
    tolerance = 0.02 * mean_center  # 10% of the mean center value
    tolerance_height = 0.02 * mean_height
    print(f"Tolerância: {tolerance}")

    filtered_rectangles = []
    for x, y, w, h in rectangles:
        center_x = x + w // 2
        if abs(center_x - mean_center) <= tolerance and abs(h-mean_height) <= tolerance_height:
            filtered_rectangles.append((x, y, w, h))
    
    return filtered_rectangles


######################################################################################################################################
# Pegar os centros da rua, comparálos e fazer uma média, depois pegar os que tem o centro a 10% da média para ambos os lados e plotar
######################################################################################################################################
def main():
    video = cv.VideoCapture(2)
    # video = cv.VideoCapture(0)
    all_rectangles = []
    while True:
        status,frame = video.read()
        rectangles = detect_rectangles(frame)
        
        centers = calculate_centers(rectangles)
        tamanhos = calculate_tam(rectangles)
        filtered_rectangles = filter_rectangles_by_center(rectangles, centers,tamanhos)
        
        """
        cr = set(filtered_rectangles)
        ar = set(all_rectangles)
        all_rectangles = ar ^ cr
        """
        if not status:
            video = cv.VideoCapture(2)
            # video = cv.VideoCapture(0)
            continue

        for x,y,w,h in all_rectangles:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)     #DESENHANDO O RETÂNGULO

        cv.imshow("VIDEO",frame)
        key = cv.waitKey(25)
        if key == 27:
            break
        ## [exit]

if __name__ == "__main__":
    main()

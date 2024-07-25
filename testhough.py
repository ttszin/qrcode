"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np


def detect_rectangles(image_path):
    # Carregar a imagem
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Aplicar um filtro para reduzir ruídos
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Detectar bordas usando o Canny
    edges = cv.Canny(blurred, 50, 150)
    
    # Detectar linhas usando a Transformada de Hough
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Mostrar a imagem com as linhas detectadas
    cv.imshow('Detected Rectangles', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Exemplo de uso
detect_rectangles('caminho/para/sua/imagem.jpg')


def main():
    video = cv.VideoCapture(2)
    while True:
        
        status,frame = video.read()

        ## [edge_detection]
        # Edge detection
        dst = cv.Canny(frame, 50, 200, None, 3)
        ## [edge_detection]

        # Copy edges to the images that will display the results in BGR
        cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)

        ## [hough_lines]
        #  Standard Hough Line Transform
        lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
        ## [hough_lines]
        ## [draw_lines]
        # Draw the lines
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

                cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
        ## [draw_lines]

        ## [hough_lines_p]
        # Probabilistic Line Transform
        linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
        ## [hough_lines_p]



        ## [draw_lines_p]
        # Draw the lines
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
        ## [draw_lines_p]
        ## [imshow]
        # Show results
        cv.imshow("Source", frame)
        cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
        cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
        ## [imshow]
        ## [exit]
        # Wait and Exit
        key = cv.waitKey(25)
        if key == 27:
            break
        ## [exit]

if __name__ == "__main__":
    main()
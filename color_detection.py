import cv2 as cv
import numpy as np
from sensors_class import Sensors


def color_detection(image) -> tuple:
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
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



def main():
    # image = cv.imread('./images/red1.jpg')
    video = cv.VideoCapture(2)
    sensors = Sensors()
    while True:
        status,frame = video.read()
        

        mask_yellow,mask_red = color_detection(frame)

        cv.imshow("Imagem", frame)


        edges_yellow = cv.Canny(mask_yellow,75,150)

        edges_red = cv.Canny(mask_red,75,150)


        sensors.setImage(frame)
        sensors.processImage()
        sensors.show()
        cv.imshow("Edges Yellow",edges_yellow)
        cv.imshow("Edges Red",edges_red)
     
        key = cv.waitKey(25)
        if key == 27:
            break

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
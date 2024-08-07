import cv2


def main():
    #load the image
    video = cv2.VideoCapture(2)


    while True:
        # grayscale
        status,frame = video.read()
        result = frame.copy()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # adaptive threshold
        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,9)

        # Fill rectangular contours
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(thresh, [c], -1, (255,255,255), -1)

        # Morph open
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

        # Draw rectangles, the 'area_treshold' value was determined empirically
        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        area_treshold = 4000
        for c in cnts:
            if cv2.contourArea(c) > area_treshold :
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (36,255,12), 3)

        cv2.imshow('thresh', thresh)
        cv2.imshow('opening', opening)
        cv2.imshow('image', frame)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
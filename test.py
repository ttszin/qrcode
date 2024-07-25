import cv2
import numpy as np

# Function to draw lines
def draw_lines(img, lines, color=(0, 255, 0), thickness=2):
    if lines is not None:
        for line in lines:
            # Each line should be a 1D array or list with two elements
            if len(line) == 1 and len(line[0]) == 2:
                rho, theta = line[0]  # Unpacking line parameters
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            else:
                print(f"Unexpected line format: {line}")

# Function to find intersection points
def line_intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([rho1, rho2])
    try:
        x, y = np.linalg.solve(A, b)
        return int(np.round(x)), int(np.round(y))
    except np.linalg.LinAlgError:
        return None

def filter_lines(lines, min_length=100):
    filtered_lines = []
    for line in lines:
        if len(line) == 1 and len(line[0]) == 2:
            rho, theta = line[0]
            if rho > min_length:
                filtered_lines.append((rho, theta))
        else:
            print(f"Unexpected line format in filter_lines: {line}")
    return np.array(filtered_lines)

def main():
    video = cv2.VideoCapture(2)
    while True:
        status, frame = video.read()
        
        if not status:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # Hough Line Transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)  # Adjust threshold as needed

        if lines is not None:
            if len(lines) > 100:  # Limit to 100 lines for simplicity
                print(len(lines))
                lines = lines[:100]
                draw_lines(frame, lines)
            # Draw the lines on the image

            # Filter the lines
            lines = filter_lines(lines)
            
            
            # Find potential corners by checking intersections
            corners = []
            for i in range(0, len(lines)):
                for j in range(i + 1, len(lines)):
                    line1 = lines[i]
                    line2 = lines[j]
                    corner = line_intersection(line1, line2)
                    if corner is not None:
                        corners.append(corner)

            # Remove duplicate corners and convert to array
            corners = list(set(corners))
            
            # Assuming we have at least four corners, we can draw the rectangle
            if len(corners) >= 4:
                # Sort the corners to form a rectangle (optional step)
                corners = np.array(corners, dtype=np.float32)
                rect = cv2.minAreaRect(corners)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

        # Show the result
        cv2.imshow('Detected Rectangle', frame)
        
        key = cv2.waitKey(25)
        if key == 27:
            break

    video.release()
    cv2.destroyAllWindows()

main()

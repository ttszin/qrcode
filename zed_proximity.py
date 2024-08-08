import pyzed.sl as sl
import cv2
import numpy as np

# Configurar a câmera ZED
camera = sl.Camera()
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
init_params.coordinate_units = sl.UNIT.METER
camera.open(init_params)

# Configurar os parâmetros de runtime
runtime_params = sl.RuntimeParameters()

# Definir objetos de memória para capturar imagens e profundidade
image = sl.Mat()
depth = sl.Mat()

# Capturar a imagem e o mapa de profundidade
if camera.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
    camera.retrieve_image(image, sl.VIEW.LEFT)
    camera.retrieve_measure(depth, sl.MEASURE.DEPTH)

    # Converter para numpy arrays
    image_np = image.get_data()
    depth_np = depth.get_data()

    # Encontrar o ponto mais próximo
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(depth_np)

    # Destacar o ponto mais próximo na imagem RGB
    cv2.circle(image_np, min_loc, 10, (0, 0, 255), 2)

    # Exibir a imagem
    cv2.imshow("Imagem com ponto mais próximo", image_np)
    cv2.waitKey(0)

# Fechar a câmera
camera.close()

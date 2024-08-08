import cv2
import numpy as np

# Carrega a imagem principal e o template
img = cv2.imread('./template/obstaculoamarelo3.jpg', 0)

image_copy = img.copy()
stretch_img = cv2.resize(image_copy,(1200,720),
                interpolation = cv2.INTER_NEAREST)

# Carrega o template
template = cv2.imread('./template/caninho.png', cv2.IMREAD_GRAYSCALE)
stretch_template = cv2.resize(template,(1200,720),
                    interpolation = cv2.INTER_NEAREST)
# template2 = cv2.imread('cano2.png',0)
# template3 = cv2.imread('cano3.png',0)
# template4 = cv2.imread('cano4.png',0)

# cv2.bitwise_or(template,template2,template3,template4)

# Cria a máscara, que deve ter o mesmo tamanho que o template
# As áreas que não fazem parte da forma desejada devem ser 0 (preto)
# As áreas da forma desejada devem ser 255 (branco)
# Cria a máscara binária com threshold
_, mask = cv2.threshold(stretch_template, 1, 255, cv2.THRESH_BINARY)
stretch_mask = cv2.resize(mask, (1200, 720),  
                interpolation = cv2.INTER_NEAREST) 

# Aplica o template matching com a máscara
res = cv2.matchTemplate(stretch_img, stretch_template, cv2.TM_CCOEFF, mask=stretch_mask)

# Encontra o melhor local de correspondência
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# Obtém o ponto superior esquerdo da correspondência
top_left = max_loc
bottom_right = (top_left[0] + stretch_template.shape[1], top_left[1] + stretch_template.shape[0])

# Desenha um retângulo em torno da correspondência detectada
cv2.rectangle(image_copy, top_left, bottom_right, 255, 2)

# Mostra a imagem final
cv2.imshow('Detected', image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()


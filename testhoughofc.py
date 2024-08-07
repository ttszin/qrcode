import cv2
import numpy as np

def main():
    # Carrega a imagem
    imagem = cv2.imread('./imagens/red1.jpg')
    imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    # Detecta contornos na imagem em escala de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    bordas = cv2.Canny(imagem_cinza, 100, 200)
    contornos, hierarquia = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Identifica o objeto mais próximo com base no maior contorno (suposição: objeto maior está mais próximo)
    contorno_maior = max(contornos, key=cv2.contourArea)

    # Cria uma máscara em branco do mesmo tamanho da imagem
    mascara_objeto = np.zeros_like(imagem_cinza)

    # Desenha o contorno do maior objeto na máscara
    cv2.drawContours(mascara_objeto, [contorno_maior], -1, 255, thickness=cv2.FILLED)

    # Define os limites para a cor vermelha no espaço de cor HSV
    limiar_inferior1 = np.array([0, 70, 50])
    limiar_superior1 = np.array([10, 255, 255])

    limiar_inferior2 = np.array([170, 70, 50])
    limiar_superior2 = np.array([180, 255, 255])

    # Segmenta a cor vermelha na imagem
    mascara_vermelha1 = cv2.inRange(imagem_hsv, limiar_inferior1, limiar_superior1)
    mascara_vermelha2 = cv2.inRange(imagem_hsv, limiar_inferior2, limiar_superior2)
    mascara_vermelha = cv2.bitwise_or(mascara_vermelha1, mascara_vermelha2)

    # Aplica a máscara do objeto mais próximo na máscara vermelha
    mascara_final = cv2.bitwise_and(mascara_vermelha, mascara_objeto)

    # Calcula a proporção de pixels vermelhos dentro da área do objeto
    total_pixels = cv2.countNonZero(mascara_objeto)
    pixels_vermelhos = cv2.countNonZero(mascara_final)
    proporcao_vermelha = pixels_vermelhos / total_pixels if total_pixels > 0 else 0

    # Exibe os resultados
    print(f'Proporção de área vermelha no objeto mais próximo: {proporcao_vermelha:.2f}')

    cv2.imshow('Área Vermelha no Objeto Mais Próximo', mascara_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

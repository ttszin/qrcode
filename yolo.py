import torch
import cv2
def main():
    # Carregar o modelo YOLOv5 pré-treinado
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Você pode usar yolov5s, yolov5m, yolov5l, yolov5x

    # Carregar a imagem
    imagem = cv2.imread('./imagens/red1.jpg')

    # Realizar a detecção
    resultados = model(imagem)

    # Processar os resultados
    for det in resultados.xyxy[0]:
        x1, y1, x2, y2, conf, class_id = det
        # Desenhar a caixa delimitadora
        cv2.rectangle(imagem, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Adicionar label de classe
        cv2.putText(imagem, f'{resultados.names[int(class_id)]} {conf:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Exibir a imagem com as detecções
    cv2.imshow('Detecção com YOLOv5', imagem)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
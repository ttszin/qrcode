import cv2
import numpy as np

# Carregar os nomes das classes do COCO
classes = None
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Carregar o modelo YOLO
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Definir parâmetros de processamento
def get_outputs_names(net):
    # Obter os nomes das camadas de saída da rede
    layer_names = net.getLayerNames()
    return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def draw_pred(frame, class_id, conf, left, top, right, bottom):
    # Desenhar uma caixa delimitadora ao redor do objeto detectado
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    label = f'{classes[class_id]}: {conf:.2f}'
    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def post_process(frame, outs, conf_threshold=0.5, nms_threshold=0.4):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    class_ids = []
    confidences = []
    boxes = []

    # Analisar as saídas da rede e filtrar os objetos detectados
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Aplicar supressão de não-máximos para remover caixas sobrepostas
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left, top, width, height = box
        draw_pred(frame, class_ids[i], confidences[i], left, top, left + width, top + height)

# Carregar a imagem ou capturar vídeo da câmera
image_path = 'image.jpg'  # Substitua pelo caminho da sua imagem
frame = cv2.imread(image_path)

# Criar um blob da imagem
blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Fazer a detecção
outs = net.forw

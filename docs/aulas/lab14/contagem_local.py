#!/usr/bin/env python3
"""Deteccao e contagem em tempo real com YOLO e OpenCV."""

from pathlib import Path

import cv2
from ultralytics import YOLO


# Altere estas configuracoes para experimentar outros videos e classes.
VIDEO_PATH = "pessoa_estacao.mp4"
USE_WEBCAM = False
TARGET_CLASS = "person"
CONFIDENCE_THRESHOLD = 0.40
LINE_RATIO = 0.50
MODEL_NAME = "yolo26n.pt"


def get_class_id(model, target_class):
    for class_id, class_name in model.names.items():
        if class_name == target_class:
            return class_id
    available = ", ".join(model.names.values())
    raise ValueError(
        f"Classe '{target_class}' nao existe no COCO. Classes disponiveis: {available}"
    )


def draw_hud(frame, count, target_class, line_y, paused):
    height, width = frame.shape[:2]
    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 3)
    cv2.putText(
        frame,
        f"{target_class}: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        3,
    )
    message = "ESPACO: pausar | R: reiniciar contagem | Q ou ESC: sair"
    if paused:
        message = "PAUSADO | " + message
    cv2.putText(
        frame,
        message,
        (20, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def main():
    if not 0 < CONFIDENCE_THRESHOLD <= 1:
        raise ValueError("CONFIDENCE_THRESHOLD deve estar entre 0 e 1.")
    if not 0 < LINE_RATIO < 1:
        raise ValueError("LINE_RATIO deve estar entre 0 e 1.")
    if not USE_WEBCAM and not Path(VIDEO_PATH).exists():
        raise FileNotFoundError(f"Video nao encontrado: {VIDEO_PATH}")

    model = YOLO(MODEL_NAME)
    target_class_id = get_class_id(model, TARGET_CLASS)
    source = 0 if USE_WEBCAM else VIDEO_PATH
    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError("Nao foi possivel abrir a fonte de video.")

    previous_center_y = {}
    counted_ids = set()
    count = 0
    paused = False
    last_frame = None

    print("Janela aberta. Pressione Q ou ESC para sair.")
    while True:
        if not paused:
            success, frame = capture.read()
            if not success:
                print(f"Fim do video. Total contado: {count}")
                break

            line_y = int(frame.shape[0] * LINE_RATIO)
            tracked = model.track(
                frame,
                persist=True,
                classes=[target_class_id],
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )[0]
            annotated = tracked.plot()

            if tracked.boxes.id is not None:
                boxes = tracked.boxes.xyxy.cpu().numpy()
                track_ids = tracked.boxes.id.int().cpu().tolist()
                for box, track_id in zip(boxes, track_ids):
                    _, y1, _, y2 = box
                    center_y = (y1 + y2) / 2
                    previous_y = previous_center_y.get(track_id)
                    crossed_down = previous_y is not None and previous_y < line_y <= center_y
                    if crossed_down and track_id not in counted_ids:
                        count += 1
                        counted_ids.add(track_id)
                    previous_center_y[track_id] = center_y

            last_frame = annotated

        if last_frame is not None:
            display = last_frame.copy()
            line_y = int(display.shape[0] * LINE_RATIO)
            draw_hud(display, count, TARGET_CLASS, line_y, paused)
            cv2.imshow("Lab 14 - Deteccao e Contagem", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        if key in (ord("r"), ord("R")):
            previous_center_y.clear()
            counted_ids.clear()
            count = 0
            print("Contagem reiniciada.")

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import cv2
# ЗАВИСИМОСТИ ПРОЕКТА:
# pip install torch ultralytics supervision opencv-python
import time
import threading
import queue
import logging
import torch
from ultralytics import YOLO
import supervision as sv

logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ============================== НАСТРОЙКИ ==============================
SOURCE = 0  # 0 (или 1, 2...) — веб-камера; либо строка с путём к видеофайлу
CAM_REQUEST_WIDTH = 1280
CAM_REQUEST_HEIGHT = 720
MODEL_PATH = "yolo26n.pt"

IMG_SIZE = 640
CONF_THRESHOLD = 0.45
DETECT_EVERY_N_FRAMES = 1
LOOP_VIDEO = False

SAVE_OUTPUT = True
OUTPUT_PATH = "conveyor_vision_demo.mp4"
OUTPUT_FPS = 30  # фиксированный FPS сохранённого файла
DISPLAY_WINDOW = True  # показывать live-окно, пока идёт обработка

RESIZE_TO = (1920, 1080)
# =========================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Initializing model on {device.upper()}...")

if device == "cuda":
    torch.backends.cudnn.benchmark = True

USE_HALF = device == "cuda"

model = YOLO(MODEL_PATH).to(device)

dummy_input = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), device=device)
for _ in range(3):
    model(dummy_input, verbose=False, half=USE_HALF)
print("[INFO] Model warmed up and ready!")


class VideoStream:
    """Читает кадры в отдельном потоке, чтобы декодирование видео/камеры не блокировало инференс."""

    def __init__(
        self, source, loop=True, queue_size=2, request_width=None, request_height=None
    ):
        self.is_camera = isinstance(source, int)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            kind = "camera" if self.is_camera else "video file"
            raise RuntimeError(f"Could not open {kind}: {source}")

        if self.is_camera and request_width and request_height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, request_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, request_height)

        self.loop = loop and not self.is_camera
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _reader(self):
        while not self.stopped:
            if self.q.full():
                time.sleep(0.001)
                continue
            ok, frame = self.cap.read()
            if not ok:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.stopped = True
                self.q.put(None)
                break
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def stop(self):
        self.stopped = True
        self.cap.release()


try:
    stream = VideoStream(
        SOURCE,
        loop=LOOP_VIDEO,
        request_width=CAM_REQUEST_WIDTH,
        request_height=CAM_REQUEST_HEIGHT,
    ).start()
except RuntimeError as e:
    print(f"[ERROR] {e}")
    raise SystemExit(1)

w, h = RESIZE_TO if RESIZE_TO else (stream.width, stream.height)
print(f"[INFO] Source resolution: {stream.width}x{stream.height} -> working at {w}x{h}")
tracker = sv.ByteTrack()

corner_annotator = sv.BoxCornerAnnotator(
    thickness=3,
    corner_length=20,
    color=sv.ColorPalette.DEFAULT,
    color_lookup=sv.ColorLookup.CLASS,
)
label_annotator = sv.LabelAnnotator(
    text_scale=0.6,
    text_thickness=2,
    color=sv.ColorPalette.DEFAULT,
    color_lookup=sv.ColorLookup.CLASS,
)

writer = None
if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, OUTPUT_FPS, (w, h))
    print(f"[INFO] Saving annotated video to: {OUTPUT_PATH}")

if DISPLAY_WINDOW:
    cv2.namedWindow("CONVEYOR VISION PRO", cv2.WINDOW_NORMAL)

fps_ema = 0.0
EMA_ALPHA = 0.1
prev_time = time.time()

frame_idx = 0
last_detections = sv.Detections.empty()
last_labels = []

try:
    while True:
        frame = stream.read()
        if frame is None:
            break

        if RESIZE_TO:
            frame = cv2.resize(frame, RESIZE_TO)

        run_detection = frame_idx % DETECT_EVERY_N_FRAMES == 0

        if run_detection:
            results = model(
                frame, imgsz=IMG_SIZE, device=device, half=USE_HALF, verbose=False
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[detections.confidence > CONF_THRESHOLD]
            detections = tracker.update_with_detections(detections)

            last_labels = [
                f"ID#{tracker_id} {model.names[class_id].upper()} {confidence:.2f}"
                for confidence, class_id, tracker_id in zip(
                    detections.confidence, detections.class_id, detections.tracker_id
                )
            ]
            last_detections = detections

        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time
        inst_fps = (1.0 / dt) if dt > 0 else 0.0
        fps_ema = (
            inst_fps
            if fps_ema == 0
            else EMA_ALPHA * inst_fps + (1 - EMA_ALPHA) * fps_ema
        )

        annotated_frame = frame.copy()
        annotated_frame = corner_annotator.annotate(
            scene=annotated_frame, detections=last_detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=last_detections, labels=last_labels
        )

        class_counts = {}
        for class_id in last_detections.class_id:
            name = model.names[class_id].upper()
            class_counts[name] = class_counts.get(name, 0) + 1
        breakdown = (
            "  ".join(f"{name}:{count}" for name, count in sorted(class_counts.items()))
            or "—"
        )

        panel = annotated_frame.copy()
        cv2.rectangle(panel, (10, 10), (470, 180), (15, 15, 15), -1)
        annotated_frame = cv2.addWeighted(panel, 0.6, annotated_frame, 0.4, 0)
        cv2.rectangle(
            annotated_frame, (10, 10), (470, 180), (255, 255, 255), 1, cv2.LINE_AA
        )

        cv2.putText(
            annotated_frame,
            f"HARDWARE: {device.upper()}{' (FP16)' if USE_HALF else ''}",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated_frame,
            f"FPS: {fps_ema:.0f}",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated_frame,
            f"ON BELT NOW: {len(last_detections)}",
            (30, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated_frame,
            breakdown,
            (30, 163),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )

        if int(current_time * 2) % 2 == 0:
            cv2.circle(annotated_frame, (w - 110, 34), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            annotated_frame,
            "LIVE",
            (w - 90, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        if writer is not None:
            writer.write(annotated_frame)

        if DISPLAY_WINDOW:
            cv2.imshow("CONVEYOR VISION PRO", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
finally:
    stream.stop()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")

    try:
        import winsound

        winsound.MessageBeep()
    except Exception:
        pass
    try:
        import ctypes

        msg = "Обработка видео завершена!"
        if SAVE_OUTPUT:
            msg += f"\n\nСохранено в:\n{OUTPUT_PATH}"
        ctypes.windll.user32.MessageBoxW(0, msg, "Conveyor Vision Pro", 0x40)
    except Exception:
        pass
# Not-ai
Сортировочный аппарат для склала доставки типа самокат

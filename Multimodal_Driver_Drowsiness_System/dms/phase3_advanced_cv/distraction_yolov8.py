"""
Phase 3 - Advanced Computer Vision: Distraction Detection
============================================================
Uses YOLOv8 (Ultralytics) to detect distraction-related objects/behaviors:
phone use, eating, smoking. Designed to run alongside face_mesh_ear.py on
the Jetson, exported to TensorRT INT8 for real-time inference.

Requires: ultralytics, opencv-python
    pip install ultralytics opencv-python --break-system-packages

Note on the model: this script loads a base YOLOv8n checkpoint. For real
deployment you must fine-tune on a driver-distraction dataset (e.g.
State Farm Distracted Driver Detection, or your own Phase 7 collected
data) with classes like {phone, food, cigarette, hand_on_wheel}. The
`CLASS_TO_DISTRACTION` map below shows how COCO-pretrained detections can
be used as an interim proxy (phone/cell phone -> "phone", etc.) before
your fine-tuned weights are ready.
"""

import argparse
import sys
import time

sys.path.append("..")
from common.schemas import now  # noqa: E402

# Proxy mapping from COCO class names to Phase-3 distraction categories.
# Replace with your fine-tuned model's class list once trained (Phase 7).
CLASS_TO_DISTRACTION = {
    "cell phone": "phone",
    "cup": "eating",       # coarse proxy: drinking/eating
    "bottle": "eating",
    "sandwich": "eating",
    "banana": "eating",
}

CONF_THRESHOLD = 0.4


class DistractionDetector:
    def __init__(self, weights: str = "yolov8n.pt", device: str = "cpu"):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = device

    def infer(self, frame_bgr):
        """Returns (label, confidence) for the highest-confidence
        distraction-relevant detection in the frame, or ("none", 0.0)."""
        results = self.model.predict(frame_bgr, device=self.device, verbose=False,
                                      conf=CONF_THRESHOLD)
        best_label, best_conf = "none", 0.0
        for r in results:
            names = r.names
            for box in r.boxes:
                cls_name = names[int(box.cls[0])]
                conf = float(box.conf[0])
                mapped = CLASS_TO_DISTRACTION.get(cls_name)
                if mapped and conf > best_conf:
                    best_label, best_conf = mapped, conf
        return best_label, best_conf

    def export_tensorrt_int8(self, calib_data_yaml: str, out_dir: str = "."):
        """Optimize the model with TensorRT INT8 for Jetson deployment
        (Phase 3 deliverable requirement). Requires a calibration dataset
        yaml describing representative driving-cabin images."""
        self.model.export(format="engine", int8=True, data=calib_data_yaml,
                           device=self.device, project=out_dir)


def main():
    ap = argparse.ArgumentParser(description="Phase 3: YOLOv8 distraction detection")
    ap.add_argument("--source", default="0")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    import cv2
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    detector = DistractionDetector(weights=args.weights)

    print("[*] Phase 3 distraction detector running.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        label, conf = detector.infer(frame)
        ts = now()
        if not args.headless:
            color = (0, 0, 255) if label != "none" else (0, 200, 0)
            cv2.putText(frame, f"{label} ({conf:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow("Phase 3 - Distraction Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(f"t={ts:.2f} distraction={label} conf={conf:.2f}")
            time.sleep(0.03)

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

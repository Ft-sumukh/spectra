"""
COCO class names for YOLO object detection.

80 classes in the canonical COCO order used by every YOLOv8/YOLOv10 export.
Kept in its own module so the vision engine stays readable and so the labels
can be swapped without touching inference code.

Source: the standard Ultralytics COCO label ordering.
"""

COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

NUM_COCO_CLASSES = len(COCO_CLASSES)

if NUM_COCO_CLASSES != 80:  # pragma: no cover
    raise RuntimeError(
        f"COCO label table must hold exactly 80 entries, found {NUM_COCO_CLASSES}. "
        f"A wrong count silently mislabels every detection."
    )

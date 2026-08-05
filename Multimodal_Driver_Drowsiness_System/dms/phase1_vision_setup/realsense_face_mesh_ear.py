import cv2
import numpy as np
import pyrealsense2 as rs

from face_mesh_ear import DrowsinessEstimator

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

estimator = DrowsinessEstimator()
align = rs.align(rs.stream.color)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        vision = estimator.process(color_image)
        annotated_rgb = estimator.draw_overlay(color_image.copy(), vision)

        depth_display = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET,
        )

        cv2.putText(
            depth_display,
            "Depth",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        display = np.hstack((annotated_rgb, depth_display))
        cv2.imshow("D435 — Driver monitoring | Depth", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

    if estimator.face_mesh is not None:
        estimator.face_mesh.close()

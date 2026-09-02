import cv2
import numpy as np
import pyrealsense2 as rs
import supervision as sv
from trackers import McByteTracker
from face_mesh_ear import DrowsinessEstimator

def main():
    # 1. Initialize Intel RealSense D435 Camera
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable RGB stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    print("[*] Starting Intel RealSense D435...")
    profile = pipeline.start(config)
    
    # 2. Initialize McByte Tracker and Drowsiness Estimator
    print("[*] Initializing McByte Tracker...")
    tracker = McByteTracker()
    estimator = DrowsinessEstimator()
    
    try:
        while True:
            # Wait for frames from camera
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
                
            # Convert frame to numpy array for OpenCV compatibility
            color_image = np.asanyarray(color_frame.get_data())
            
            # 3. Process image with the MediaPipe estimator
            vision = estimator.process(color_image)
            
            if vision.face_found:
                h, w, _ = color_image.shape
                
                # Retrieve or calculate the face bounding box
                if hasattr(vision, 'face_box'):
                    face_box = vision.face_box  # Format: [xmin, ymin, xmax, ymax]
                elif hasattr(vision, 'landmarks') and vision.landmarks:
                    # Extract geometric boundary coordinates from face landmarks
                    xs = [lm[0] * w for lm in vision.landmarks]
                    ys = [lm[1] * h for lm in vision.landmarks]
                    face_box = [min(xs), min(ys), max(xs), max(ys)]
                else:
                    # Fallback default box in the center of the frame
                    face_box = [w//4, h//4, 3*w//4, 3*h//4]
                
                # Convert bounding box coordinates to supervision standard format
                detections = sv.Detections(
                    xyxy=np.array([face_box], dtype=np.float32),
                    confidence=np.array([0.99], dtype=np.float32),
                    class_id=np.array([0], dtype=np.int32)
                )
                
                # 4. Update McByte Tracker (the raw frame is required to propagate masks)
                tracked_detections = tracker.update(detections=detections, frame=color_image)
                
                # McByte assigns a stable and persistent tracking ID to the driver
                if len(tracked_detections) > 0 and tracked_detections.tracker_id is not None:
                    driver_id = tracked_detections.tracker_id[0]
                    
                    # Draw bounding box and display the stable driver ID
                    box = tracked_detections.xyxy[0].astype(int)
                    cv2.rectangle(color_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(color_image, f"Driver ID: {driver_id}", (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Overlay real-time fatigue and drowsiness metrics on screen
                cv2.putText(color_image, f"EAR: {vision.ear:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(color_image, f"PERCLOS: {vision.perclos:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(color_image, f"Yawn: {vision.yawn_score:.2f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(color_image, "No Face Found", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Render the final output image window
            cv2.imshow("RealSense D435 + McByte Tracking", color_image)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Graceful shutdown of RealSense camera stream and window cleanups
        print("[*] Stopping RealSense pipeline...")
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

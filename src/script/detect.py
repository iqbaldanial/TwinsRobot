#!/usr/bin/env python

import rospy
import numpy as np
import cv2
import torch
import time
import os
import threading
from gtts import gTTS
from collections import Counter
from sensor_msgs.msg import Image
# from test_grocery.msg import Boundingbox
from std_msgs.msg import String

from cv_bridge import CvBridge, CvBridgeError
from airina_fyp.srv import StartDetection, StartDetectionResponse, LatestDetection, LatestDetectionResponse
import ultralytics
import math

class ItemDetection:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('start_detection')
        
        # Load the trained YOLO model - using the same model as ObjectPickerGUI
        self.model = ultralytics.YOLO('/home/mustar/catkin_ws/src/airina_fyp/DatasetModel3/runs/detect/train/weights/best.pt')
        self.bridge = CvBridge()
        image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        self.sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1)
        # Publisher for annotated images to be displayed in the dashboard
        self.pub_annotated = rospy.Publisher('/detection/annotated_image', Image, queue_size=1)
        self.CONFIDENCE_THRESHOLD = 0.7  # Using same confidence as ObjectPickerGUI
        self.service = rospy.Service('startDetect', StartDetection, self.handle_detection_request)
        self.service = rospy.Service('latestDetect', LatestDetection, self.get_latest_detections)
        self.cv_image = None 
        self.GRIPPER_REGION = (-1, -1, -1, -1)
        self.display_image = None  # Initialize display image
        self.latest_frame = None
        self.lock = threading.Lock()
        
        self.CLASS = ["crumpled paper", "plastic bottle", "drink carton", "dustbin"]
        self.CLASS_OBJECT = ["crumpled paper", "plastic bottle", "drink carton"]
        self.CLASS_TARGET = ["dustbin"]

        self.OBJECT_NAMES = {0: "crumpled paper", 1: "drink carton", 2: "dustbin", 3: "plastic bottle"}

        # announcement control
        self.last_announced_detection = None
        self.announcement_cooldown = 3.0  # seconds between announcements
        self.last_announcement_time = 0

        # Start detection and display threads
        threading.Thread(target=self.detection_loop, daemon=True).start()

    def image_callback(self, msg_color):
        """Only store latest frame, no detection here."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg_color, "bgr8")
            frame = np.flip(frame, axis=1)
            with self.lock:
                self.latest_frame = frame
        except CvBridgeError as e:
            rospy.logwarn(str(e))

    def detection_loop(self):
        """Run YOLO detection in a separate thread."""
        rate = rospy.Rate(10)  # Run detection at max 10 FPS
        while not rospy.is_shutdown():
            frame_copy = None
            with self.lock:
                if self.latest_frame is not None:
                    frame_copy = self.latest_frame.copy()

            if frame_copy is not None:
                self.update_display_image(frame_copy)

            rate.sleep()
    
    def update_display_image(self, frame):
        """Run YOLO and update display_image."""
        results = self.model(frame, conf=self.CONFIDENCE_THRESHOLD, show=False)

        if results[0].boxes:
            for box in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = box[:6]
                cls = int(cls)
                if cls in self.OBJECT_NAMES:
                    if self.is_in_gripper_region(x1, y1, x2, y2):
                        color = (0, 0, 255)
                        status = " (IN GRIPPER)"
                    else:
                        color = (0, 255, 0)
                        status = ""
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{self.OBJECT_NAMES[cls]} ({conf:.2f}){status}",
                                (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        gx1, gy1, gx2, gy2 = self.GRIPPER_REGION
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
        cv2.putText(frame, "GRIPPER REGION", (gx1, gy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        try:
            img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.pub_annotated.publish(img_msg)
        except Exception as e:
            rospy.logwarn(f"Failed to publish annotated image: {e}")

        with self.lock:
            self.display_image = frame
            self.cv_image = frame

    def is_in_gripper_region(self, bx1, by1, bx2, by2):
        # Define a fixed gripper region in image coordinates (tune values)
        gx1, gy1, gx2, gy2 = self.GRIPPER_REGION
        # Return True if the bounding box overlaps with the gripper region
        return not (bx2 < gx1 or bx1 > gx2 or by2 < gy1 or by1 > gy2)


    def detect_objects(self, frame):
        """
        Detect objects using the same logic as ObjectPickerGUI
        Returns list of detected objects with their bounding boxes
        """
        detected_objects = []
        
        # Run YOLO detection - same parameters as ObjectPickerGUI
        results = self.model(frame, conf=self.CONFIDENCE_THRESHOLD, show=False)
        print("Detecting")
        # Get image dimensions for coordinate correction
        img_height, img_width = frame.shape[:2]
        
        if results[0].boxes:
            for box in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = box[:6]
                cls = int(cls)      
                
                # Only process objects that are in our defined classes
                if cls in self.OBJECT_NAMES:
                    # Since the image was flipped horizontally, we need to correct the x-coordinates
                    # Convert flipped coordinates back to original coordinates
                    corrected_x1 = img_width - int(x2)  # x2 becomes x1 after flip correction
                    corrected_x2 = img_width - int(x1)  # x1 becomes x2 after flip correction
                    corrected_y1 = int(y1)  # y coordinates don't change with horizontal flip
                    corrected_y2 = int(y2)
                    
                    if not self.is_in_gripper_region(x1, y1, x2, y2):
                        print(f"{self.OBJECT_NAMES[cls]} not in gripper region")
                        detected_objects.append({
                            'x1': corrected_x1, 'y1': corrected_y1, 'x2': corrected_x2, 'y2': corrected_y2,
                            'confidence': float(conf), 'class_id': cls,
                            'class_name': self.OBJECT_NAMES[cls]
                        })
                    
                        # Draw bounding box on frame (using original flipped coordinates for visualization)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"{self.OBJECT_NAMES[cls]} ({conf:.2f})",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )
        # Draw gripper region for reference
        gx1, gy1, gx2, gy2 = self.GRIPPER_REGION
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
        self.cv_image = frame.copy()
        
        return detected_objects

    def get_latest_detections(self, req):
        with self.lock:
            if self.latest_frame is None:
                return LatestDetectionResponse(
                    xmin=[], xmax=[], ymin=[], ymax=[],
                    confidence=[], class_name=[],
                    success=False,
                    message="No camera image available"
                )
            frame_copy = self.latest_frame.copy()

        detections = self.detect_objects(frame_copy)

        if not detections:
            return LatestDetectionResponse(
                xmin=[], xmax=[], ymin=[], ymax=[],
                confidence=[], class_name=[],
                success=False,
                message="No objects detected"
            )

        return LatestDetectionResponse(
            xmin=[obj['x1'] for obj in detections],
            xmax=[obj['x2'] for obj in detections],
            ymin=[obj['y1'] for obj in detections],
            ymax=[obj['y2'] for obj in detections],
            confidence=[obj['confidence'] for obj in detections],
            class_name=[obj['class_name'] for obj in detections],
            success=True,
            message=f"Detected {len(detections)} objects"
        )


    def handle_detection_request(self, req):
        mode = req.mode
        target_class = req.class_name
        rospy.loginfo(f"Received detection request: mode={mode}, target={target_class}")

        max_wait_time = 5.0
        wait_start = rospy.Time.now()

        while self.latest_frame is None and (rospy.Time.now() - wait_start).to_sec() < max_wait_time:
            rospy.sleep(0.1)

        if self.latest_frame is None:
            return StartDetectionResponse(0, 0, 0, 0, "", False, "No camera image available")

        try:
            # --- This is the new looping logic for 'pickup' ---
            if mode == 'pickup':
                rospy.loginfo(f"Starting continuous search for '{target_class}'...")
                self.GRIPPER_REGION = (0, 0, 0, 0) # Clear gripper region
                
                SEARCH_TIMEOUT_SECONDS = 20.0  # Give up after 20 seconds
                search_start_time = rospy.Time.now()

                while not rospy.is_shutdown():
                    # 1. Check for search timeout
                    if (rospy.Time.now() - search_start_time).to_sec() > SEARCH_TIMEOUT_SECONDS:
                        rospy.logwarn("Search timed out. Object not found.")
                        return StartDetectionResponse(0, 0, 0, 0, "", False, f"I looked for {SEARCH_TIMEOUT_SECONDS} seconds but could not find the {target_class}.")

                    # 2. Get a fresh camera frame
                    frame_copy = None
                    with self.lock:
                        if self.latest_frame is not None:
                            frame_copy = self.latest_frame.copy()
                    
                    if frame_copy is None:
                        rospy.sleep(0.1)
                        continue
                    
                    # 3. Run detection on the fresh frame
                    detected_objects = self.detect_objects(frame_copy)
                    best_object = None

                    # 4. Check if our *specific target* is in the list
                    if detected_objects:
                        target_detections = [obj for obj in detected_objects if obj['class_name'] == target_class]
                        if target_detections:
                            # Found it! Get the one with the highest confidence
                            best_object = max(target_detections, key=lambda obj: obj['confidence'])
                    
                    # 5. If we found it, return a success response
                    if best_object:
                        rospy.loginfo(f"Target found! Detected {best_object['class_name']} with confidence {best_object['confidence']:.2f}")
                        
                        # (Optional: Save reference image - same as your old code)
                        ref_path_pick = "/home/mustar/catkin_ws/src/airina_fyp/src/detection/pick"
                        cv_img = np.flip(self.cv_image, axis=1)
                        ref_img = cv_img[int(best_object['y1']):int(best_object['y2']), int(best_object['x1']):int(best_object['x2'])].copy()
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(f"{ref_path_pick}/ref_img_{timestamp}.png", ref_img)
                        rospy.loginfo(f"Reference image saved at {ref_path_pick}")

                        # Return the successful response
                        return StartDetectionResponse(
                            xmin=best_object['x1'],
                            xmax=best_object['x2'],
                            ymin=best_object['y1'],
                            ymax=best_object['y2'],
                            class_name=best_object['class_name'],
                            success=True,
                            message=f"I have found the {best_object['class_name']}."
                        )

                    # 6. If not found, sleep for a moment and loop again
                    rospy.sleep(0.5) # Re-check 2 times per second

            # --- This is your original 'place' logic, left unchanged ---
            elif mode == 'place':
                self.GRIPPER_REGION = (274, 388, 372, 478) 
                detected_objects = self.detect_objects(self.latest_frame.copy())
                rospy.loginfo(f"Detection completed, found {len(detected_objects)} objects")
                
                if not detected_objects:
                    rospy.logwarn("No detected objects for 'place' mode")
                    return StartDetectionResponse(0, 0, 0, 0, "", False, "No detected objects")
                
                # (This logic is from your file, assuming it's for finding the dustbin)
                best_object = max(detected_objects, key=lambda obj: obj['confidence'])
                ref_path_place = "/home/mustar/catkin_ws/src/airina_fyp/src/detection/place"
                cv_img = np.flip(self.cv_image, axis=1)
                ref_img = cv_img[int(best_object['y1']):int(best_object['y2']), int(best_object['x1']):int(best_object['x2'])].copy()
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"{ref_path_place}/ref_img_{timestamp}.png", ref_img)
                rospy.loginfo(f"Reference image saved at {ref_path_place}")

                if target_class == "":
                    first_obj = detected_objects[0]
                    return StartDetectionResponse(
                        xmin=first_obj['x1'], xmax=first_obj['x2'], ymin=first_obj['y1'], ymax=first_obj['y2'],
                        class_name=first_obj['class_name'], success=True,
                        message=""
                    )
                else:
                    placement_response = self.find_similar_object(detected_objects, target_class)                   
                    return placement_response
            
            else:
                 rospy.logwarn(f"Unknown mode: {mode}")
                 return StartDetectionResponse(0, 0, 0, 0, "", False, "Unknown mode")

        except Exception as e:
            rospy.logerr(f"Error in handle_detection_request: {e}")
            return StartDetectionResponse(0, 0, 0, 0, "", False, str(e))

    def find_similar_object(self, detected_objects, target_class):
        print("Finding object with similarity check...")

        best_match = max((obj for obj in detected_objects if obj['class'] == target_class), key=lambda obj: obj['confidence'])

        if not best_match:
            print("No valid match after similarity check")
            return StartDetectionResponse(
                xmin=0, xmax=0, ymin=0, ymax=0,
                class_name=target_class,
                success=False,
                message=""
            )
        else:
            return StartDetectionResponse(
                xmin=best_match['x1'], xmax=best_match['x2'],
                ymin=best_match['y1'], ymax=best_match['y2'],
                class_name=best_match['class_name'],
                success=True,
                message=""
            )

    def announce_detections(self, detected_objects):
            
        if not detected_objects:
            self.speak("No objects detected")
            return

        # Count occurrences of each class name
        counts = Counter(obj['class_name'] for obj in detected_objects)

        # Build speech text
        parts = []
        for name, count in counts.items():
            if count == 1:
                parts.append(f"1 {name}")
            else:
                parts.append(f"{count} {name}s")  # plural form (basic)
        sentence = "I see " + " and ".join(parts)

        # Speak it
        return sentence
        

if __name__ == "__main__":
    try:
        item = ItemDetection()
        # Display loop in main thread
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            img = None
            with item.lock:
                if item.display_image is not None:
                    img = item.display_image.copy()

            if img is not None:
                cv2.imshow("YOLO Detections", img)
                cv2.waitKey(1)
            rate.sleep()

        cv2.destroyAllWindows()
        rospy.spin()
        # Make sure to properly handle shutdown
        cv2.destroyAllWindows()
    except rospy.ROSInterruptException:
        rospy.loginfo("Error with yolondistance.py")
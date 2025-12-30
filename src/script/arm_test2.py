#!/usr/bin/env python

import rospy
import math
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from cv_bridge import CvBridge, CvBridgeError
import cv2
import tf2_ros
import geometry_msgs
import tf2_geometry_msgs

# Import your custom service message types
from airina_fyp.srv import ArmHeadGripper, ArmHeadGripperResponse

# Robot specific constants
CAMERA_FRAME_ID = "camera_link"
BASE_FRAME_ID = "base_link"
ARUCO_POSE_TOPIC = "/aruco_single/pose"

class ArmManipulationService:
    def __init__(self):
        rospy.init_node("head_arm_hand", anonymous=True)

        self.last_pick_location = None

        # --- TF2 Setup for Coordinate Transforms ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # --- ArUco Marker Setup ---
        self.latest_aruco_pose_stamped = None
        self.aruco_pose_sub = rospy.Subscriber(
            ARUCO_POSE_TOPIC,
            PoseStamped, 
            self.aruco_pose_callback, 
            queue_size=1
        )
        
        # Camera and sensor setup
        self.bridge = CvBridge()
        self.camera_sub = rospy.Subscriber('/camera/depth_registered/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)

        # Publishers
        self.pub_arm1 = rospy.Publisher('/arm1_joint/command', Float64, queue_size=10)
        self.pub_arm2 = rospy.Publisher('/arm2_joint/command', Float64, queue_size=10)
        self.pub_arm3 = rospy.Publisher('/arm3_joint/command', Float64, queue_size=10)
        self.pub_arm4 = rospy.Publisher('/arm4_joint/command', Float64, queue_size=10)
        self.pub_gripper = rospy.Publisher('/gripper_joint/command', Float64, queue_size=10)
        self.pub_base = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Robot parameters
        self.camera_height = 45.0
        self.camera_offset = 25.0
        self.camera_angle = 30.0
        self.L2, self.L3, self.L4 = 10.16, 10.16, 6.0

        # Current sensor data
        self.frame = None
        self.depth_frame = None
        self.lidar_data = None
        
        # Navigation parameters
        self.center_threshold = 150
        self.distance_threshold = 70.0  # This is the original distance_threshold
        self.placement_distance_threshold = 60.0
        self.placement_center_radius = 100
        self.obstacle_distance_threshold = 0.5
        self.front_angle_range = 60
        
        # Movement parameters
        self.rotation_step = 0.3
        self.movement_step = 0.15
        self.step_duration = 0.05

        # Service server
        self.service = rospy.Service('arm_manipulation', ArmHeadGripper, self.handle_arm_manipulation)
        
        rospy.loginfo("Arm Manipulation Service initialized and ready")

    # --- ArUco Handlers ---
    def aruco_pose_callback(self, msg):
        """Stores the latest ArUco marker pose relative to the camera."""
        self.latest_aruco_pose_stamped = msg

    def get_robot_pose_from_aruco(self, target_id=1):
        """Transforms the ArUco marker pose from camera frame to robot base frame."""
        if self.latest_aruco_pose_stamped is None:
            rospy.logwarn("No ArUco marker pose received yet.")
            return None, None

        pose_in_camera_frame = self.latest_aruco_pose_stamped
        
        try:
            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME_ID, 
                pose_in_camera_frame.header.frame_id, 
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            
            pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_in_camera_frame, transform)
            
            robot_x = pose_transformed.pose.position.x * 100.0 # Convert meters to cm
            robot_z = pose_transformed.pose.position.z * 100.0 # Convert meters to cm
            
            placement_offset_cm = 5.0 # Vertical safety offset
            
            rospy.loginfo(f"Transformed ArUco target (base_link): X={robot_x:.3f}cm, Z={robot_z:.3f}cm")
            
            # Return X and Z, applying a safety offset for placement *above* the marker
            return robot_x, robot_z + placement_offset_cm
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"TF2 transform failed: {e}")
            return None, None
    
    # --- Service Handler ---
    def handle_arm_manipulation(self, req):
        response = ArmHeadGripperResponse()

        try:
            rospy.loginfo(f"Received arm manipulation request: mode={req.mode}, class={req.class_name}")

            if req.mode.lower() == "pick":
                success = self.execute_pick_sequence(req.xmin, req.xmax, req.ymin, req.ymax, req.class_name)

            elif req.mode.lower() == "place":
                success = self.execute_place_sequence(req.xmin, req.xmax, req.ymin, req.ymax, req.class_name)

            else:
                rospy.logwarn(f"Unknown mode: {req.mode}")
                success = False

            response.success = success
            response.message = "Operation completed successfully" if success else "Operation failed"

        except Exception as e:
            rospy.logerr(f"Error in arm manipulation service: {e}")
            response.success = False
            response.message = f"Error: {str(e)}"

        return response

    # --- Pick Sequence ---
    def execute_pick_sequence(self, xmin, xmax, ymin, ymax, class_name):
        """
        Execute the complete pick sequence for a STATIC arm.
        """
        rospy.loginfo("Starting STATIC pick sequence...")

        if class_name == "return_to_origin":
            rospy.logwarn("Target not found. Returning object to original pickup location.")
            if self.last_pick_location:
                rx, rz = self.last_pick_location
                # We add +5.0 cm to Z to ensure we place it gently ON the table/floor, not IN it.
                return self.place_object_at_target(rx, rz + 5.0, 90)
            else:
                rospy.logerr("No previous pick location saved!")
                return False
        
        # 1. Calculate bounding box properties
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        box_width = abs(xmax - xmin)
        box_height = abs(ymax - ymin)
        
        # 2. Get object distance using our new working depth topic
        distance = self.get_robust_depth(center_x, center_y, box_width, box_height)
        if distance is None: 
            rospy.logwarn("Could not get valid depth measurement. Aborting pick.")
            return False
            
        rospy.loginfo(f"Object found at distance: {distance:.1f} cm")

        # 3. Check if object is reachable
        if distance > self.distance_threshold:
            rospy.logwarn(f"Object is too far away ({distance:.1f}cm). "
                          f"Please move it closer (within {self.distance_threshold}cm).")
            return False
            
        rospy.loginfo("Object is reachable.")

        # 4. Transform pixel coordinates to robot (x, z) coordinates
        try:
            robot_x, robot_y, robot_z = self.transform_to_robot_frame_depth(distance, center_x, center_y)

            # Small forward offset so the arm reaches slightly past the detected point.
            # This helps compensate for any systematic under-estimation of distance for table objects,
            # without affecting the place behaviour (which uses its own coordinates).
            PICK_FORWARD_OFFSET_CM = 5.0
            robot_x += PICK_FORWARD_OFFSET_CM

            rospy.loginfo(
                f"Calculated robot target (with pick offset): X={robot_x:.2f}cm, Z={robot_z:.2f}cm "
                f"(+{PICK_FORWARD_OFFSET_CM:.1f}cm forward)"
            )

            self.last_pick_location = (robot_x, robot_z)
        except Exception as e:
            rospy.logerr(f"Failed to transform coordinates: {e}")
            return False

        # 5. Check if the arm can physically reach those (x, z) coordinates
        theta2, theta3, theta4 = self.calculate_inverse_kinematics(robot_x, robot_z, 90)
        if theta2 is None:
            rospy.logwarn(f"Cannot calculate Inverse Kinematics for target X={robot_x}, Z={robot_z}. "
                          "Target is unreachable by the arm.")
            return False

        rospy.loginfo("IK solution found. Arm can reach the object.")

        # 6. Align Gripper (Arm1 Joint)
        self.align_gripper_with_object(center_x)

        # 7. Execute the final pickup
        success = self.pickup_object(robot_x, robot_z, 90)
        
        rospy.loginfo(f"Static pick sequence {'completed successfully' if success else 'failed'}")
        return success

    # --- Sensor Callbacks ---
    def image_callback(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logwarn(f"Image conversion failed: {e}")

    def depth_callback(self, msg):
        try:
            if msg.encoding == "16UC1":
                self.depth_frame = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            elif msg.encoding == "32FC1":
                self.depth_frame = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            else:
                self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logwarn(f"Depth image conversion failed: {e}")

    def lidar_callback(self, msg):
        self.lidar_data = msg

    # --- Depth and Distance ---
    def get_depth_at_pixel(self, x, y):
        if self.depth_frame is None:
            return None
        
        height, width = self.depth_frame.shape
        x = int(max(0, min(x, width - 1)))
        y = int(max(0, min(y, height - 1)))
        
        depth_value = self.depth_frame[y, x]
        
        if self.depth_frame.dtype == np.uint16:
            if depth_value == 0:
                return None
            return depth_value / 10.0
        elif self.depth_frame.dtype == np.float32:
            if np.isnan(depth_value) or depth_value == 0:
                return None
            return depth_value * 100.0
        else:
            return None

    def get_robust_depth(self, center_x, center_y, box_width, box_height):
        if self.depth_frame is None:
            return None
        
        sample_radius = min(box_width, box_height) * 0.2
        depth_values = []
        
        sample_points = [
            (center_x, center_y),
            (center_x - sample_radius/2, center_y),
            (center_x + sample_radius/2, center_y),
            (center_x, center_y - sample_radius/2),
            (center_x, center_y + sample_radius/2),
        ]
        
        for x, y in sample_points:
            depth = self.get_depth_at_pixel(x, y)
            if depth is not None and depth > 0:
                depth_values.append(depth)
        
        if not depth_values:
            return None
        
        return np.median(depth_values)

    # --- Navigation Helpers (Copied from original file) ---
    def check_object_centered(self, center_x):
        frame_width = self.frame.shape[1] if self.frame is not None else 640
        camera_center_x = frame_width / 2
        offset = abs(center_x - camera_center_x)
        return offset < self.center_threshold

    # --- Coordinate Transform ---
    def transform_to_robot_frame_depth(self, distance_cm, center_x, center_y):
        distance = distance_cm
        horizontal_distance = math.cos(math.radians(self.camera_angle)) * distance
        vertical_offset = self.camera_height - math.sin(math.radians(self.camera_angle)) * distance
        
        robot_x = horizontal_distance - self.camera_offset
        robot_z = vertical_offset
        
        frame_width = self.frame.shape[1] if self.frame is not None else 640
        robot_y = (center_x - frame_width / 2) * 0.01
        
        return robot_x, robot_y, robot_z

    # --- Inverse Kinematics ---
    def calculate_inverse_kinematics(self, x, z, alpha_deg):
        alpha = math.radians(alpha_deg)
        m = z - self.L4 * math.cos(alpha)
        n = x - self.L4 * math.sin(alpha)

        if math.sqrt(m**2 + n**2) > (self.L2 + self.L3 + self.L4 + 200): 
            rospy.logwarn("IK Target out of range")
            return None, None, None

        cos_theta3 = (m**2 + n**2 - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
        cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
        
        theta3 = math.acos(cos_theta3)
        theta12 = math.atan2(n, m)
        beta = math.atan2(self.L3 * math.sin(theta3), self.L2 + self.L3 * math.cos(theta3))
        
        theta2 = theta12 - beta
        theta4 = alpha - (theta2 + theta3)
        
        return math.degrees(theta2), math.degrees(theta3), math.degrees(theta4)

    # --- Arm Control ---
    def move_to_ready_position(self):
        rospy.loginfo("Moving to ready position")
        self.pub_arm1.publish(Float64(0))
        self.pub_arm2.publish(Float64(-1.65806))
        self.pub_arm3.publish(Float64(2))
        rospy.sleep(1)
        self.pub_arm4.publish(Float64(1.09956))
        self.pub_gripper.publish(Float64(0.5)) # Close/neutral gripper
        rospy.sleep(5)

    def move_to_place_position(self):
        """Moves the arm to the HARDCODED place-down position."""
        rospy.loginfo("Moving the arm to the HARDCODED place-down position.")
        self.pub_arm1.publish(Float64(0))
        self.pub_arm2.publish(Float64(math.radians(26.77)))
        self.pub_arm3.publish(Float64(math.radians(0.00)))
        self.pub_arm4.publish(Float64(math.radians(63.23)))

        rospy.sleep(5)

        # Open the gripper to place down the object
        rospy.loginfo("Placing object down by opening the gripper.")
        self.pub_gripper.publish(Float64(-0.5))  # Open gripper
        rospy.sleep(1)

        # Return to the ready position after placing
        self.move_to_ready_position()
        return True # Return success

    # --- MODIFIED ALIGN FUNCTION ---
    def align_gripper_with_object(self, obj_center_x):
        """
        Rotates Arm 1 to point directly at the object.
        Updated to remove hardcoded LEFT_OFFSET and use dynamic image width.
        """
        # 1. Get dynamic screen center (don't assume 320)
        if self.frame is not None:
            width = self.frame.shape[1]
        else:
            width = 640 # Default fallback
            
        screen_center = width / 2
        
        # 2. Tuning Parameters
        # HFOV for Astra/Realsense is usually around 60 degrees (~1.047 radians)
        # Convert HFOV to radians and calculate gain based on image width
        HFOV_RADIANS = 1.047  # ~60 degrees
        FOV_GAIN = HFOV_RADIANS  # This gives us the full FOV range
        
        # 3. Calculate Error
        # Positive error (Object on Left of center) -> Positive Rotation (Left)
        # Negative error (Object on Right of center) -> Negative Rotation (Right)
        error_pixels = screen_center - obj_center_x
        
        # 4. Calculate Angle - normalize error to [-1, 1] range, then scale by FOV
        # This gives us the angle in radians proportional to the FOV
        normalized_error = error_pixels / screen_center  # Range: [-1, 1]
        angle_to_rotate = normalized_error * FOV_GAIN
        
        # 5. Mechanical Offset - Compensate for systematic bias
        # If gripper consistently ends up too far RIGHT, add positive offset (rotate more LEFT)
        # If gripper consistently ends up too far LEFT, add negative offset (rotate more RIGHT)
        # Tuned to compensate for rightward bias
        MECHANICAL_OFFSET = 0.08  # Positive = rotate more left to compensate for rightward bias
        
        final_angle = angle_to_rotate + MECHANICAL_OFFSET

        rospy.loginfo(f"Aligning: Object X={obj_center_x:.1f}, Center={screen_center:.1f}, "
                     f"Error={error_pixels:.1f}px, Normalized={normalized_error:.3f}, "
                     f"Base Angle={angle_to_rotate:.3f}rad, Offset={MECHANICAL_OFFSET:.3f}rad, "
                     f"Final Angle={final_angle:.3f}rad ({math.degrees(final_angle):.2f}°)")
        
        self.pub_arm1.publish(Float64(final_angle))
        rospy.sleep(1.5) # Wait for servo to move (increased for better settling)

    def pickup_object(self, x, z, alpha_deg):
        rospy.loginfo("Executing pickup sequence")
        theta2, theta3, theta4 = self.calculate_inverse_kinematics(x, z, alpha_deg)
        
        if theta2 is None or theta3 is None or theta4 is None:
            rospy.logwarn("Failed to calculate joint angles for pickup")
            return False
        
        rospy.loginfo(f"Pickup angles - Theta2: {theta2:.2f}°, Theta3: {theta3:.2f}°, Theta4: {theta4:.2f}°")
        
        # Open gripper
        self.pub_gripper.publish(Float64(-0.3))
        rospy.sleep(2)
        
        # Move to object
        self.pub_arm2.publish(Float64(math.radians(theta2)))
        self.pub_arm3.publish(Float64(math.radians(theta3)))
        self.pub_arm4.publish(Float64(math.radians(theta4)))
        rospy.sleep(5)
        
        # Close gripper
        self.pub_gripper.publish(Float64(1.0))
        rospy.sleep(1)
        
        # Return to ready position
        self.move_to_ready_position()

        return True

    # --- NEW ArUco Place Function ---
    def place_object_smart_dispatch(self):
        rospy.loginfo("Attempting smart placement using ArUco...")
        TARGET_ARUCO_ID = 1
        
        robot_x, robot_z = self.get_robot_pose_from_aruco(TARGET_ARUCO_ID)
        
        if robot_x is None or robot_z is None:
            rospy.logwarn(f"Failed to get ArUco marker pose for ID {TARGET_ARUCO_ID}. Using fallback.")
            return self.move_to_place_position()
            
        rospy.loginfo(f"ArUco target {TARGET_ARUCO_ID} acquired. Moving to robot coordinates: (X: {robot_x:.2f}cm, Z: {robot_z:.2f}cm)")

        theta2, theta3, theta4 = self.calculate_inverse_kinematics(robot_x, robot_z, 90)
        
        if theta2 is None:
            rospy.logwarn(f"Cannot calculate IK for placement. Using fallback.")
            return self.move_to_place_position()

        # Execute
        rospy.loginfo(f"Placing at angles: T2={theta2:.2f}, T3={theta3:.2f}, T4={theta4:.2f}")
        
        self.pub_arm1.publish(Float64(0)) 
        self.pub_arm2.publish(Float64(math.radians(theta2)))
        self.pub_arm3.publish(Float64(math.radians(theta3)))
        self.pub_arm4.publish(Float64(math.radians(theta4)))
        rospy.sleep(5) 
        
        rospy.loginfo("Opening gripper to place object.")
        self.pub_gripper.publish(Float64(-0.5)) 
        rospy.sleep(1)
        
        self.move_to_ready_position()
        return True
    
    def execute_place_sequence(self, xmin, xmax, ymin, ymax, class_name):
        """
        Execute the complete place sequence for a STATIC arm.
        """
        rospy.loginfo("Starting STATIC place sequence...")

        if xmin == 0 and xmax == 0:
            rospy.logwarn("No target coordinates. Using hardcoded place position.")
            return self.move_to_place_position() 

        # Calculate bounding box properties
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        box_width = abs(xmax - xmin)
        box_height = abs(ymax - ymin)

        # Get target distance 
        distance = self.get_robust_depth(center_x, center_y, box_width, box_height)
        if distance is None:
            rospy.logwarn("Could not get valid depth for dustbin. Using hardcoded place.")
            return self.move_to_place_position()

        rospy.loginfo(f"Target '{class_name}' found at distance: {distance:.1f} cm")

        # Transform pixel coordinates to robot (x, z) coordinates
        try:
            robot_x, robot_y, robot_z = self.transform_to_robot_frame_depth(distance, center_x, center_y)
            rospy.loginfo(f"Calculated robot target: X={robot_x:.2f}, Z={robot_z:.2f}")
        except Exception as e:
            rospy.logerr(f"Failed to transform coordinates: {e}")
            return self.move_to_place_position()

        # We add a 15cm Z-offset to place *above* the target
        robot_z_place = robot_z + 15.0 
        theta2, theta3, theta4 = self.calculate_inverse_kinematics(robot_x, robot_z_place, 90)
        if theta2 is None:
            rospy.logwarn(f"Cannot calculate IK for target X={robot_x}, Z={robot_z_place}.")
            return self.move_to_place_position()

        rospy.loginfo("IK solution found. Arm can reach the target.")

        # Align Gripper (Arm1 Joint)
        self.align_gripper_with_object(center_x)

        # Execute the final placement
        success = self.place_object_at_target(robot_x, robot_z_place, 90)

        rospy.loginfo(f"Static place sequence {'completed successfully' if success else 'failed'}")
        return success
    
    def place_object_at_target(self, x, z, alpha_deg):
        rospy.loginfo("Executing placement at target")
        theta2, theta3, theta4 = self.calculate_inverse_kinematics(x, z, alpha_deg)

        if theta2 is None:
            rospy.logwarn("Failed to calculate joint angles for placement")
            return False

        rospy.loginfo(f"Placement angles - T2: {theta2:.2f}°, T3: {theta3:.2f}°, T4: {theta4:.2f}°")

        # Move to object (with object in gripper)
        self.pub_arm2.publish(Float64(math.radians(theta2)))
        self.pub_arm3.publish(Float64(math.radians(theta3)))
        self.pub_arm4.publish(Float64(math.radians(theta4)))
        rospy.sleep(5)

        # Open gripper to drop object
        self.pub_gripper.publish(Float64(-0.5)) # Open gripper
        rospy.sleep(2)

        # Return to ready position
        self.move_to_ready_position()
        return True

if __name__ == '__main__':
    try:
        service = ArmManipulationService()
        rospy.loginfo("Arm Manipulation Service started successfully")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Arm Manipulation Service terminated")

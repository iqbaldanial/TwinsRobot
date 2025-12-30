#!/usr/bin/env python3
# filepath: /path/to/airina_fyp/src/example_main.py

import rospy
from std_msgs.msg import Float64
from airina_fyp.srv import Navigate, StartDetection, StartDetectionResponse
from airina_fyp.srv import ArmHeadGripper, ArmHeadGripperResponse
from airina_fyp.srv import StartTask, StartTaskResponse
import time

# Note: Removed gTTS, os, and speech_recognition imports as they are no longer needed for audio output

class Run:
    def __init__(self):
        rospy.init_node('start')

        # --- UPDATED CLASSES FOR TRASH CLEANUP ---
        self.CLASS_ROBOT_PICK = ["crumpled paper", "plastic bottle", "drink carton"]
        self.CLASS_DESTINATION = ["dustbin", "trash can", "bin"] 
        
        # State Variable: Keeps track of what the robot is holding
        self.item_picked_up = None 
        
        self.pub_head = rospy.Publisher('/head_joint/command', Float64, queue_size=10)

        # 1. HOST THE HIGH-LEVEL SERVICE
        rospy.Service('/initiate_cleanup_task', StartTask, self.handle_cleanup_request)
        rospy.loginfo("Arm Cleanup Task Service is Ready and waiting for commands.")
    
    def text2audio(self, text):
        """
        MODIFIED: Silent version. 
        Instead of speaking, this now just logs the status to the terminal.
        The actual voice interaction is handled by main.py.
        """
        rospy.loginfo(f"ACTION STATUS: {text}")

    def control_head(self, action):
        head_joint_values = {
            "down": 0.51,
            "rest": 0.0,
            "up": -0.51
        }
        if action in head_joint_values:
            self.pub_head.publish(Float64(head_joint_values[action]))
        else:
            rospy.logerr("Invalid action for head.")

    def start_detection(self, mode, class_name):
        """Calls the 'startDetect' service."""
        rospy.wait_for_service('startDetect')
        try:
            get_detection_service = rospy.ServiceProxy('startDetect', StartDetection)
            rospy.loginfo(f"Calling detection service: mode={mode}, class={class_name}")
            self.detection_response = get_detection_service(mode, class_name, True) 
            return self.detection_response
        except rospy.ServiceException as e:
            print("Detection service call failed:", e)
            return StartDetectionResponse(success=False, message=str(e))

    def arm_manipulation(self, mode, xmin, ymin, xmax, ymax, class_name):
        """Calls the 'arm_manipulation' service."""
        rospy.wait_for_service('arm_manipulation')
        print(f"Calling arm service: mode={mode}")
        try:
            get_arm_service = rospy.ServiceProxy('arm_manipulation', ArmHeadGripper)
            response = get_arm_service(mode, xmin, ymin, xmax, ymax, class_name) 
            return response
        except rospy.ServiceException as e:
            print("Arm manipulation service call failed:", e)
            return ArmHeadGripperResponse(success=False, message=str(e))

    # --- MAIN SERVICE HANDLER ---
    def handle_cleanup_request(self, req):
        """
        Decides whether to PICK or PLACE based on user input and robot state.
        """
        target = req.target_item_name.lower().strip()
        rospy.loginfo(f"Received request for: {target}")

        # ---------------------------------------------------------
        # CONDITION 1: PLACE COMMAND (Context Aware)
        # ---------------------------------------------------------
        
        is_generic_place = target in ["place", "it", "that", "item", "object", "current item"]
        is_destination = target in self.CLASS_DESTINATION
        is_holding_target = (self.item_picked_up is not None and target == self.item_picked_up)

        if is_destination or is_generic_place or is_holding_target:
            
            # Check if we are actually holding something
            if self.item_picked_up is None:
                msg = "I am not holding anything to place. Please tell me to pick up an item first."
                self.text2audio(msg) # Logs to terminal only
                return StartTaskResponse(success=False, verbal_response=msg)
            
            # Execute Place
            rospy.loginfo(f"Interpreted '{target}' as a PLACE command for '{self.item_picked_up}'")
            success = self.execute_place_logic()
            
            if success:
                msg = f"I have placed the {self.item_picked_up} in the dustbin. Ready for next command."
                self.item_picked_up = None # Reset state
            else:
                msg = "I failed to place the item."
            
            return StartTaskResponse(success=success, verbal_response=msg)

        # ---------------------------------------------------------
        # CONDITION 2: PICK COMMAND
        # ---------------------------------------------------------
        elif target in self.CLASS_ROBOT_PICK:
            
            # Check if we are already holding something
            if self.item_picked_up is not None:
                # This error only triggers if they name a DIFFERENT item than what we hold
                msg = f"I am already holding a {self.item_picked_up}. Please tell me to place it first."
                self.text2audio(msg)
                return StartTaskResponse(success=False, verbal_response=msg)

            # Execute Pick
            success = self.execute_pick_logic(target)
            if success:
                msg = f"I have picked up the {target}."
            else:
                msg = f"I could not pick up the {target}."

            return StartTaskResponse(success=success, verbal_response=msg)

        # ---------------------------------------------------------
        # CONDITION 3: UNKNOWN COMMAND
        # ---------------------------------------------------------
        else:
            msg = f"I don't know how to handle {target}. I can pick up bottles, paper, or cartons."
            self.text2audio(msg)
            return StartTaskResponse(success=False, verbal_response=msg)

    # --- SPLIT LOGIC PART 1: PICK ONLY ---
    def execute_pick_logic(self, target_item_name):
        
        self.text2audio(f"Okay, finding {target_item_name}.")
        
        # 1. Look down
        self.control_head('down')
        rospy.sleep(1.0)
        
        # 2. Detect
        try:
            detection_response = self.start_detection('pickup', target_item_name) 
            
            if not detection_response.success:
                self.text2audio(f"I couldn't find the {target_item_name}.")
                return False
            
            xmin, xmax, ymin, ymax = (detection_response.xmin, 
                                      detection_response.xmax, 
                                      detection_response.ymin, 
                                      detection_response.ymax)
            
            self.text2audio(f"Found it. Picking up.")
            
        except rospy.ServiceException as e:
            rospy.logerr(f"Detection service call failed: {e}")
            return False

        # 3. Arm Pick
        try:
            arm_response = self.arm_manipulation(
                mode='pick', 
                xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, 
                class_name=detection_response.class_name
            )
            
            if not arm_response.success:
                self.text2audio(f"Pickup failed. Returning to rest.")
                return False
            
            # SUCCESS: Update State
            self.item_picked_up = detection_response.class_name
            self.text2audio("Pickup successful. Holding position.")
            return True
            
        except rospy.ServiceException as e:
            rospy.logerr(f"Arm service call failed: {e}")
            return False

    # --- SPLIT LOGIC PART 2: PLACE ONLY ---
    def execute_place_logic(self):
        self.text2audio("Moving to placement phase.")
        
        # 1. Look down to find dustbin
        self.control_head('down') 
        rospy.sleep(1.0)
        self.text2audio("Looking for the dustbin.")

        # 2. Detect Dustbin
        dustbin_found = False
        dustbin_xmin, dustbin_ymin, dustbin_xmax, dustbin_ymax = 0, 0, 0, 0
        
        try:
            # We use 'pickup' mode logic just to find the object coordinates
            detection_response = self.start_detection('pickup', 'dustbin') 
            
            if detection_response.success:
                self.text2audio("Dustbin found.")
                dustbin_found = True
                dustbin_xmin = detection_response.xmin
                dustbin_ymin = detection_response.ymin
                dustbin_xmax = detection_response.xmax
                dustbin_ymax = detection_response.ymax
            else:
                self.text2audio("I cannot see the dustbin. Returning item to original position.")
                dustbin_found = False

        except rospy.ServiceException as e:
            rospy.logerr(f"Detection service call failed: {e}")
            return False

        rospy.sleep(1.0)

        # 3. Arm Place (Conditional)
        try:
            if dustbin_found:
                # Normal placement into dustbin
                arm_response = self.arm_manipulation(
                    mode='place', 
                    xmin=dustbin_xmin, 
                    ymin=dustbin_ymin, 
                    xmax=dustbin_xmax, 
                    ymax=dustbin_ymax, 
                    class_name='dustbin'
                )
            else:
                # --- NEW: RETURN TO ORIGIN ---
                arm_response = self.arm_manipulation(
                    mode='place', 
                    xmin=0, ymin=0, xmax=0, ymax=0, # Coordinates ignored by arm script in this mode
                    class_name='return_to_origin'
                )
                # -----------------------------

            if not arm_response.success:
                self.text2audio(f"Placement action failed.")
                return False

            return True

        except rospy.ServiceException as e:
            rospy.logerr(f"Arm service call failed: {e}")
            return False

      
        
if __name__=="__main__":
    try:
        robot_run = Run()
        rospy.spin() 
        
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted.")
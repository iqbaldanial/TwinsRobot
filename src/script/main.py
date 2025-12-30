#!/usr/bin/env python3

import rospy
from airina_fyp.srv import Talk, ListenToQuestion, StartTask
from std_msgs.msg import String

import time
import sys
import os
import threading
import json
import re  # Added for better text matching

from deepgram_utils import get_deepgram_utils
from general_llm import GeneralLLM 

pub_status = rospy.Publisher('/bumblebee/gui/status', String, queue_size=1)
pub_user = rospy.Publisher('/bumblebee/gui/user_text', String, queue_size=1)
pub_bot = rospy.Publisher('/bumblebee/gui/bot_text', String, queue_size=1)

global_last_target_item = None

# Update helper functions
def update_ui_status(text):
    pub_status.publish(String(text))
def update_ui_user(text):
    pub_user.publish(String(text))
def update_ui_bot(text):
    pub_bot.publish(String(text))

WAKE_WORD = "bumblebee" 
WAKE_WORD_TIMEOUT = 3
ACTIVATION_LISTEN_TIMEOUT = 4 
CONVERSATION_TIMEOUT = 7    
MAX_INITIAL_ATTEMPTS = 2    

# --- UPDATED SERVICE CALLER ---
def call_arm_task_service(target_item_name):
    """Calls the service that triggers the arm's cleanup task."""
    global global_last_target_item
    rospy.loginfo(f"Arm Task Request: Calling /initiate_cleanup_task for '{target_item_name}'")
    
    try:
        rospy.wait_for_service('/initiate_cleanup_task', timeout=10.0)
        start_task_service = rospy.ServiceProxy('/initiate_cleanup_task', StartTask)
        
        # Call the service (This blocks until the robot finishes the action!)
        response = start_task_service(target_item_name=target_item_name)
        
        if response.success:
            # --- STATE UPDATE ON SUCCESS ---
            # If the task was a PICKUP, store the item name.
            if target_item_name not in ["place", "dustbin"]:
                global_last_target_item = target_item_name
                rospy.loginfo(f"CONTEXT: Last target set to: {global_last_target_item}")
            # If the task was a PLACE, clear the item name.
            elif target_item_name == ["place", "dustbin"]:
                 global_last_target_item = None
                 rospy.loginfo("CONTEXT: Item placed. Last target cleared.")
            # -------------------------------
            
            return True, response.verbal_response
        else:
            # If failed, clear context to prevent erroneous follow-up actions.
            global_last_target_item = None
            return False, response.verbal_response

    except rospy.ServiceException as e:
        rospy.logerr(f"StartTask service call failed: {e}")
        return False, "Sorry, I lost connection to the arm control system."
    except Exception as e:
        rospy.logerr(f"Error during arm service call: {e}")
        return False, "An internal error occurred."


def deepgram_speak(text):
    update_ui_status("SPEAKING") 
    update_ui_bot(text)
    utils = get_deepgram_utils()
    if utils:
        utils.text2audio(text)
    else:
        rospy.logerr("Speech service unavailable.")

    update_ui_status("IDLE")

def convert_voice_to_text_deepgram(timeout=5):
    update_ui_status("LISTENING...")
    utils = get_deepgram_utils()
    if utils:
        text = utils.audio2text(timeout=timeout, listen_phrase="")
        
        # --- MAKE SURE THIS IS HERE ---
        if text and text.strip():  # Check if text is not empty
            rospy.loginfo(f"GUI Publishing User Text: {text}") # Debug log
            update_ui_user(text)   # Send to UI
        # ------------------------------
        
        return text
    return None

# --- UPDATED COMMAND PROCESSOR (The Orchestrator) ---
def process_command(llm_instance, user_input):
    
    update_ui_status("PROCESSING...")
    global global_last_target_item

    # 1. Get LLM response
    response_message = llm_instance.process_command(user_input)
    
    # 2. Check for Tool Call (Action Path)
    if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
        
        # We prepare a list of actions to execute
        actions_to_execute = []

        # A. Parse the LLM's extraction
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "initiate_arm_task":
                try:
                    function_args = json.loads(tool_call.function.arguments)
                    target_item = function_args.get("object", "unknown")

                    # --- FIX: Handle "It", "Unknown", or Ambiguous terms ---
                    ambiguous_terms = ["it", "that", "this", "unknown", "item", "object", "context_item"]
                    
                    # Check if the extracted item is vague
                    if target_item.lower() in ambiguous_terms:
                        
                        # 1. Check if user sentence contains PLACEMENT keywords
                        # If the LLM returned "unknown" but the user said "Place/Put", 
                        # we force the target to be "place".
                        user_text_lower = user_input.lower()
                        has_place_keyword = any(w in user_text_lower for w in ["place", "put", "throw", "drop", "bin"])
                        
                        if has_place_keyword:
                            target_item = "place"
                            rospy.loginfo("Ambiguity Resolved: Mapped 'unknown/it' to 'place' command based on sentence context.")
                        
                        # 2. Check Context (Fallback if it wasn't a place command)
                        elif target_item.lower() == "context_item" and global_last_target_item:
                             target_item = global_last_target_item
                             rospy.loginfo(f"CONTEXT: Resolved to '{target_item}'")

                    actions_to_execute.append(target_item)
                    
                except json.JSONDecodeError:
                    rospy.logerr("Error decoding tool function arguments.")

                #     # --- FIX 1: Handle "It" / Ambiguity ---
                #     # If LLM returns "it", "unknown", or "that", map it to "place"
                #     if target_item.lower() == "context_item":
                #         if global_last_target_item:
                #             target_item = global_last_target_item
                #             rospy.loginfo(f"CONTEXT: Ambiguous target resolved to '{target_item}'.")
                #         else:
                #             # Context failed: Speak the failure and ABORT the command
                #             deepgram_speak("I'm not sure which item you mean, as I haven't picked anything up.")
                #             rospy.logerr("CONTEXT FAILURE: Global item not set for ambiguous command.")
                #             return # ABORT command processing entirely
                    
                #     actions_to_execute.append(target_item)
                    
                # except json.JSONDecodeError:
                #     rospy.logerr("Error decoding tool function arguments.")

        # B. --- FIX 2: Heuristic Sequence Detection ---
        # Sometimes LLM only extracts "bottle" but misses the "place" part in a compound sentence.
        # If user said "Pick X AND place it", force a sequence.
        
        user_text_lower = user_input.lower()
        has_pick_keyword = any(w in user_text_lower for w in ["pick", "grab", "get", "take"])
        has_place_keyword = any(w in user_text_lower for w in ["place", "put", "throw", "drop", "bin"])
        
        # If we have a target (e.g., "bottle") AND the user also said "place", 
        # but "place" isn't in our action list yet...
        if actions_to_execute:
            first_action = actions_to_execute[0]

            should_sequence = (first_action != "place") and has_place_keyword and has_pick_keyword
            
            if should_sequence:
                rospy.loginfo("Detected sequential command (Pick -> Place). queuing 'place' action.")
                if "place" not in actions_to_execute:
                    actions_to_execute.append("place")
            

        # C. Execute the Actions Sequentially
        for action_target in actions_to_execute:
            
            # Speak feedback only if it's the second action (to avoid double talking)
            if action_target == "place" and len(actions_to_execute) > 1:
                deepgram_speak("Now placing it.")
            
            # Execute logic (Blocking Call)
            success, verbal_response = call_arm_task_service(target_item_name=action_target)
            
            # Speak the result
            deepgram_speak(verbal_response)
            rospy.loginfo(f"Arm Task Response: {verbal_response}")
            
            # If the first action failed (e.g., couldn't pick up), Stop! Do not try to place.
            if not success:
                rospy.logwarn("Sequence aborted due to failure in previous step.")
                break
            
            # Small pause between actions
            time.sleep(1.0)
            
        return
    
    # 3. Check for Conversational Response (Conversational Path)
    if hasattr(response_message, 'content') and response_message.content:
        deepgram_speak(response_message.content)
        rospy.loginfo(f"LLM Conversation Response: {response_message.content}")
        return

    rospy.logwarn("LLM response was empty or unhandled.")
    deepgram_speak("I'm sorry, I couldn't process that command clearly.")

# --- DEEPGRAM WAKE WORD DETECTION LOOP ---
def listen_for_wake_word():
    """Listens for the wake word in a single, short attempt."""
    
    rospy.loginfo(f">>> Waiting for wake word: '{WAKE_WORD}' <<<")
    
    text = convert_voice_to_text_deepgram(timeout=WAKE_WORD_TIMEOUT) 
    
    if text and WAKE_WORD in text.lower():
        rospy.loginfo("Wake word recognized.")
        return True
    
    return False

def main_conversational_loop():
    rospy.init_node('bumblebee_conversational_node', anonymous=True)
    
    # 1. Initialization
    speech_utils = get_deepgram_utils()
    if not speech_utils:
        rospy.logerr("Bumblebee cannot start without Deepgram utilities.")
        return
    
    try:
        llm_instance = GeneralLLM()
    except Exception as e:
        rospy.logerr(f"LLM failed to initialize: {e}")
        deepgram_speak("I am unable to connect to my core processing unit.")
        return

    # Initial terminal prompt
    print("\n--------------------------------------------------------------")
    rospy.loginfo("Bumblebee is ready. Say 'Bumblebee' to begin.")
    print("--------------------------------------------------------------\n")
    deepgram_speak("Say hey bumblebee to begin")

    # 2. Main Conversational Loop (Outer loop waits for wake word)
    while not rospy.is_shutdown():
        # Step 1: SILENTLY LISTEN FOR WAKE WORD
        if listen_for_wake_word():
            rospy.loginfo("Wake word detected! Entering command mode.")
            deepgram_speak("I'm here.") # Shortened greeting for speed
            
            # --- PHASE 2: Continuous Conversation Loop ---
            initial_command_successful = False
            
            # Allow for initial command attempts after wake word
            for attempt in range(MAX_INITIAL_ATTEMPTS):
                rospy.loginfo(f"Listening for initial command (Attempt {attempt + 1})...")
                
                # Use a specific activation timeout
                user_input = convert_voice_to_text_deepgram(timeout=ACTIVATION_LISTEN_TIMEOUT) 
                
                if user_input and user_input.strip():
                    # --- SUCCESSFUL INITIAL COMMAND ---
                    rospy.loginfo(f"Initial command received: {user_input}")
                    process_command(llm_instance, user_input)
                    initial_command_successful = True
                    break # Break out of the initial attempts loop
                else:
                    # --- FAILED INITIAL COMMAND ---
                    if attempt < MAX_INITIAL_ATTEMPTS - 1:
                        rospy.logwarn("Transcription failed. Reprompting user once.")
                        deepgram_speak("Sorry, I didn't catch that.")
                    else:
                        rospy.logerr("Initial command failed twice. Returning to silent wake word listener.")
                        break 
                        

            # --- PHASE 3: Continuous Conversation Loop ---
            if initial_command_successful:
                conversation_active = True
                while conversation_active and not rospy.is_shutdown():
                    rospy.loginfo(f"Listening for follow-up command (Timeout: {CONVERSATION_TIMEOUT}s)...")
                    
                    follow_up_input = convert_voice_to_text_deepgram(timeout=CONVERSATION_TIMEOUT)
                    
                    if follow_up_input and follow_up_input.strip():
                        rospy.loginfo(f"Follow-up command received: {follow_up_input}")
                        process_command(llm_instance, follow_up_input)
                    else:
                        rospy.loginfo("Silence detected. Exiting continuous conversation mode.")
                        conversation_active = False 
        
        rospy.sleep(0.1)


if __name__ == "__main__":
    try:
        main_conversational_loop()
    except rospy.ROSInterruptException:
        rospy.loginfo("Conversational Node: Shutting down...")
    except Exception as e:
        rospy.logerr(f"Conversational Node: Error occurred: {e}")
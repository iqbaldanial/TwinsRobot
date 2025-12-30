import streamlit as st
import rospy
import threading
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
import numpy as np
import time

# --- 1. ROS INTERFACE (Runs in Background) ---
class ROSInterface:
    def __init__(self):
        # Data holders (Thread-safe-ish for simple reads)
        self.latest_image = None
        self.status = "IDLE"
        self.user_text = "..."
        self.bot_text = "..."
        self.bridge = CvBridge()
        self.last_image_time = None
        
        # Initialize Node only once
        if not rospy.core.is_initialized():
            rospy.init_node('streamlit_dashboard', anonymous=True, disable_signals=True)

        # Subscribers - Try annotated image first, fallback to raw camera feed
        rospy.Subscriber("/detection/annotated_image", Image, self.image_cb, queue_size=1)
        rospy.Subscriber("/camera/color/image_raw", Image, self.raw_image_cb, queue_size=1)
        rospy.Subscriber("/bumblebee/gui/status", String, self.status_cb)
        rospy.Subscriber("/bumblebee/gui/user_text", String, self.user_cb)
        rospy.Subscriber("/bumblebee/gui/bot_text", String, self.bot_cb)

    def image_cb(self, msg):
        """Callback for annotated image (preferred)"""
        try:
            # Force conversion from BGR (ROS default) to RGB (Streamlit default)
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            self.last_image_time = rospy.Time.now()
        except Exception as e:
            # If "bgr8" fails, try "passthrough"
            try:
                cv_img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
                # Check if it's already RGB or needs conversion
                if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
                    self.latest_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                else:
                    self.latest_image = cv_img
                self.last_image_time = rospy.Time.now()
            except Exception:
                pass
    
    def raw_image_cb(self, msg):
        """Fallback callback for raw camera feed - only use if annotated image not available"""
        # Only update if we haven't received an annotated image recently (within 1 second)
        if self.last_image_time is None or (rospy.Time.now() - self.last_image_time).to_sec() > 1.0:
            try:
                cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.latest_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                try:
                    cv_img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
                    if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
                        self.latest_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    else:
                        self.latest_image = cv_img
                except Exception:
                    pass

    def status_cb(self, msg):
        self.status = msg.data

    def user_cb(self, msg):
        self.user_text = msg.data

    def bot_cb(self, msg):
        self.bot_text = msg.data

# --- 2. STREAMLIT CACHING (Singleton Pattern) ---
# This ensures we don't create multiple ROS nodes when Streamlit refreshes
@st.cache_resource
def get_ros_interface():
    return ROSInterface()

# --- 3. MAIN UI LAYOUT ---
def main():
    st.set_page_config(
        page_title="Bumblebee Dashboard", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Enhanced Custom CSS for beautiful, accessible UI
    st.markdown("""
        <style>
        /* Hide Streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Main container styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Status indicator styling with animations */
        .status-container {
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            text-align: center;
            font-size: 1.2rem;
            font-weight: 600;
            letter-spacing: 1px;
        }
        
        .status-recording {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            color: white;
            animation: pulse 2s infinite;
            border: 3px solid #ff4757;
        }
        
        .status-processing {
            background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
            color: #2c2c54;
            border: 3px solid #ffa502;
        }
        
        .status-idle {
            background: linear-gradient(135deg, #2c2c54 0%, #40407a 100%);
            color: white;
            border: 3px solid #706fd3;
        }
        
        .status-speaking {
            background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
            color: white;
            border: 3px solid #1e8449;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.9; transform: scale(1.02); }
        }
        
        /* Chat message styling */
        .chat-message {
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: 20%;
        }
        
        .assistant-message {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            margin-right: 20%;
        }
        
        /* Camera feed container */
        .camera-container {
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            background: #1e1e2e;
            padding: 0.5rem;
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.5rem;
            font-weight: 700;
            color: #2c2c54;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #667eea;
        }
        
        /* Command display */
        .command-display {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            font-size: 1.1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            min-height: 80px;
            display: flex;
            align-items: center;
        }
        
        .response-display {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            font-size: 1.1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            min-height: 80px;
            display: flex;
            align-items: center;
        }
        
        /* Icon styling */
        .status-icon {
            font-size: 2rem;
            margin-right: 0.5rem;
            vertical-align: middle;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0 2rem 0;'>
            <h1 style='color: #2c2c54; font-size: 2.5rem; margin-bottom: 0.5rem;'>
                🤖 Bumblebee Robot Dashboard
            </h1>
            <p style='color: #706fd3; font-size: 1.1rem;'>
                Voice-Activated Assistant • Fully Automatic • No Buttons Required
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Get the shared ROS data
    ros = get_ros_interface()

    # Main Layout: Status at top, then Camera and Chat side by side
    status_placeholder = st.empty()
    
    # Two column layout for camera and conversation
    col_camera, col_conversation = st.columns([2, 1])
    
    with col_camera:
        st.markdown('<div class="section-header">📹 Live Camera Feed</div>', unsafe_allow_html=True)
        camera_placeholder = st.empty()
    
    with col_conversation:
        st.markdown('<div class="section-header">💬 Conversation</div>', unsafe_allow_html=True)
        user_command_placeholder = st.empty()
        st.markdown('<div style="margin: 1rem 0;"></div>', unsafe_allow_html=True)
        llm_response_placeholder = st.empty()

    # --- 4. REAL-TIME UPDATE LOOP ---
    # We use a loop to constantly update the placeholders
    while True:
        # A. Update Status Indicator (Recording Status)
        s_text = ros.status.upper()
        if "RECORDING" in s_text or "LISTENING" in s_text:
            status_html = f"""
                <div class="status-container status-recording">
                    <span class="status-icon">🎙️</span>
                    <span>RECORDING AUDIO</span>
                    <div style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">
                        Listening for your command...
                    </div>
                </div>
            """
        elif "PROCESSING" in s_text:
            status_html = f"""
                <div class="status-container status-processing">
                    <span class="status-icon">⚙️</span>
                    <span>PROCESSING</span>
                    <div style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">
                        Analyzing your request...
                    </div>
                </div>
            """
        elif "SPEAKING" in s_text:
            status_html = f"""
                <div class="status-container status-speaking">
                    <span class="status-icon">🔊</span>
                    <span>SPEAKING</span>
                    <div style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">
                        Delivering response...
                    </div>
                </div>
            """
        else:
            status_html = f"""
                <div class="status-container status-idle">
                    <span class="status-icon">💤</span>
                    <span>IDLE</span>
                    <div style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">
                        Ready to listen...
                    </div>
                </div>
            """
        
        status_placeholder.markdown(status_html, unsafe_allow_html=True)

        # B. Update Camera Feed (Full width in its column)
        if ros.latest_image is not None:
            camera_placeholder.image(
                ros.latest_image, 
                channels="RGB", 
                use_container_width=True,
                caption="Live Camera Feed"
            )
        else:
            camera_placeholder.markdown("""
                <div style="background: #1e1e2e; padding: 3rem; border-radius: 15px; text-align: center; color: #706fd3;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📷</div>
                    <div style="font-size: 1.2rem;">Waiting for camera stream...</div>
                </div>
            """, unsafe_allow_html=True)

        # C. Update User Command Display
        if ros.user_text and ros.user_text != "...":
            user_html = f"""
                <div class="command-display">
                    <div>
                        <div style="font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem; opacity: 0.9;">
                            👤 USER COMMAND:
                        </div>
                        <div style="font-size: 1.2rem;">
                            {ros.user_text}
                        </div>
                    </div>
                </div>
            """
        else:
            user_html = """
                <div class="command-display" style="opacity: 0.6;">
                    <div style="text-align: center; width: 100%;">
                        👤 Waiting for user command...
                    </div>
                </div>
            """
        user_command_placeholder.markdown(user_html, unsafe_allow_html=True)

        # D. Update LLM Response Display
        if ros.bot_text and ros.bot_text != "...":
            bot_html = f"""
                <div class="response-display">
                    <div>
                        <div style="font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem; opacity: 0.9;">
                            🤖 LLM RESPONSE:
                        </div>
                        <div style="font-size: 1.2rem;">
                            {ros.bot_text}
                        </div>
                    </div>
                </div>
            """
        else:
            bot_html = """
                <div class="response-display" style="opacity: 0.6;">
                    <div style="text-align: center; width: 100%;">
                        🤖 Waiting for LLM response...
                    </div>
                </div>
            """
        llm_response_placeholder.markdown(bot_html, unsafe_allow_html=True)

        # Limit refresh rate to save CPU (approx 10 FPS is enough for UI)
        time.sleep(0.1) 

if __name__ == "__main__":
    main()
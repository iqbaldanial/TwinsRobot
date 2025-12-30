#!/usr/bin/env python3

import rospy
import sys
import os
from airina_fyp.srv import Talk, TalkResponse, ListenToQuestion, StartTask
from std_msgs.msg import String

try:
    # Try to import Deepgram utilities
    from deepgram_utils import get_deepgram_utils
    DEEPGRAM_AVAILABLE = True
    rospy.loginfo("Using Deepgram for speech services")
except ImportError:
    rospy.logwarn("Deepgram not available, falling back to gTTS/Vosk")
    DEEPGRAM_AVAILABLE = False
    from gtts import gTTS
    import json
    import vosk
    import pyaudio

try:
    from airina_fyp.src.script.general_llm import GeneralLLM
    LLM_AVAILABLE = True
except ImportError:
    rospy.logerr("general_llm module not found. Using dummy LLM for testing.")
    LLM_AVAILABLE = False
    class GeneralLLM:
        def __init__(self, use_tavily=False):
            self.use_tavily = use_tavily
        
        def answer_question(self, question):
            return f"This is a test response to: {question}"

class GeneralSpeechServiceServer:
    '''
        rosnode: general_speech_server
        service:
            talk - initiates a conversation by delivering a specific message to the person before they ask any questions.
        
        Now supports both Deepgram and legacy gTTS/Vosk backends
    '''
    def __init__(self):
        rospy.init_node('general_speech_server', anonymous=True)
        
        # Service to handle general speech
        self.talk_service = rospy.Service('/talk', Talk, self.handle_talk)
        self.listen_to_question_service = rospy.Service('/listen_to_question', ListenToQuestion, self.handle_listen_to_question)
        
        self.initiate_conversation = "Hi, can you please start an interesting topic with youngster like me? Greet the person and keep it short and simple with one or two sentences."
        
        # Initialize speech backend
        if DEEPGRAM_AVAILABLE:
            try:
                self.deepgram_utils = get_deepgram_utils()
                if self.deepgram_utils:
                    rospy.loginfo("General Speech Server initialized with Deepgram backend")
                else:
                    raise Exception("Failed to get Deepgram utilities")
            except Exception as e:
                rospy.logerr(f"Failed to initialize Deepgram backend: {e}")
                rospy.logwarn("Falling back to gTTS/Vosk backend")
                self._init_legacy_backend()
        else:
            self._init_legacy_backend()
        
        # Initialize LLM
        self.llm = GeneralLLM()

        print("General Speech Server is Ready!")
    
    def _init_legacy_backend(self):
        """Initialize legacy gTTS/Vosk backend"""
        global DEEPGRAM_AVAILABLE
        DEEPGRAM_AVAILABLE = False
        self.deepgram_utils = None
        
        # Initialize Vosk
        self.model_path = "/home/mustar/vosk-model-small-en-us-0.15"
        if not os.path.exists(self.model_path):
            # Try alternative paths
            alternative_paths = [
                "/home/mustar/juno2_materials/robotedge_speech/vosk-model-small-en-us-0.15",
                "./vosk-model-small-en-us-0.15",
                "~/vosk-model-small-en-us-0.15"
            ]
            for alt_path in alternative_paths:
                if os.path.exists(os.path.expanduser(alt_path)):
                    self.model_path = os.path.expanduser(alt_path)
                    break
        
        try:
            self.model = vosk.Model(self.model_path)
            self.rec = vosk.KaldiRecognizer(self.model, 16000)
            self.p = pyaudio.PyAudio()
            rospy.loginfo("Vosk initialized successfully for general speech server")
        except Exception as e:
            rospy.logerr(f"Failed to initialize Vosk: {e}")
            self.model = None

    def handle_listen_to_question(self, req):
        # Announce readiness to listen
        # self.text2audio("I am ready to listen to your question.")
        
        # Listen to question with reduced timeout for faster response
        question = self.audio2text(6)  # Reduced from 10 to 6 seconds
        while not question:
            self.text2audio('Sorry I cannot detect your question, can you please repeat?')
            rospy.sleep(1)
            question = self.audio2text(6)  # Reduced timeout here too
        
        # Generate answer using LLM
        answer = self.llm.answer_question(question)
        
        # Speak the answer
        self.text2audio(answer)
        
        return TalkResponse(response=answer)
    
    def handle_talk(self, req):
        # based on the message received
        if req.message:
            response_message = req.message
        else:
            # response_message = self.llm.answer_question(self.initiate_conversation)
            response_message = self.llm.answer_question(self.initiate_conversation)
        
        # Speak the message
        self.text2audio(response_message)
        
        rospy.loginfo(f"Talk: {response_message}")
        return TalkResponse(response=response_message)
    
    def text2audio(self, text):
        """Convert text to speech and play it"""
        if DEEPGRAM_AVAILABLE and self.deepgram_utils:
            try:
                self.deepgram_utils.text2audio(text)
                return
            except Exception as e:
                rospy.logerr(f"Deepgram TTS error, falling back to gTTS: {e}")
        
        # Fallback to gTTS
        try:
            tts = gTTS(text, lang="en")
            tts.save("general_speech_output.mp3")
            os.system("mpg321 general_speech_output.mp3")
            os.remove("general_speech_output.mp3")
        except Exception as e:
            rospy.logerr(f"Text-to-speech error: {e}")
    
    def audio2text(self, timeout=10):
        """Convert audio to text"""
        if DEEPGRAM_AVAILABLE and self.deepgram_utils:
            try:
                return self.deepgram_utils.audio2text(timeout)
            except Exception as e:
                rospy.logerr(f"Deepgram STT error, falling back to Vosk: {e}")

        rospy.logerr(f"Deepgram not available")


if __name__ == "__main__":
    try:
        GeneralSpeechServiceServer = GeneralSpeechServiceServer()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Main Error")
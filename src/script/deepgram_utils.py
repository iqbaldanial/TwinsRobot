#!/usr/bin/env python3

"""
Deepgram Utilities for ROS General Robot (REST API Only)
========================================================

This module provides utility functions for Deepgram speech recognition and 
text-to-speech using direct HTTP API calls for maximum compatibility.
No Deepgram SDK required - works with any Python version.
"""

import os
import rospy
import json
import threading
import time
import tempfile
import pyaudio
import requests
import wave
import websocket
import ssl
import asyncio
import aiohttp
from threading import Event

class DeepgramUtils:
    """
    Utility class providing Deepgram speech recognition and TTS functionality
    using direct HTTP API calls
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DeepgramUtils, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize Deepgram configuration"""
        if self._initialized:
            return
        
        # Get API key from environment
        self.api_key = os.getenv("DEEPGRAM_API_KEYSA")
        if not self.api_key:
            rospy.logerr("DEEPGRAM_API_KEYSA environment variable not set!")
            rospy.logerr("Please set it with: export DEEPGRAM_API_KEYSA=your_api_key_here")
            raise ValueError("Deepgram API key not found")
        
        # Configuration parameters
        self.tts_model = rospy.get_param('~deepgram_tts_model', 'aura-2-orion-en')
        self.stt_model = rospy.get_param('~deepgram_stt_model', 'nova-3')
        self.language = rospy.get_param('~deepgram_language', 'en-US')
        self.enable_silence_detection = rospy.get_param('~enable_silence_detection', False)  # Disable by default
        
        # API endpoints
        self.tts_url = "https://api.deepgram.com/v1/speak"
        self.stt_url = "https://api.deepgram.com/v1/listen"
        
        # Audio configuration
        self.sample_rate = 16000
        self.channels = 1
        self.audio_format = pyaudio.paInt16
        self.chunk_size = 8192
        
        # Initialize PyAudio for microphone input
        self.p = pyaudio.PyAudio()
        
        rospy.loginfo("Deepgram REST API utilities initialized successfully")
        self._initialized = True
    
    def text2audio(self, text, lang="en", use_streaming=False):
        """
        Convert text to speech using Deepgram TTS (REST API only for SDK 3.x)
        
        Args:
            text (str): Text to convert to speech
            lang (str): Language code (maintained for compatibility)
            use_streaming (bool): Ignored for SDK 3.x, always uses REST
        """
        if not text or not text.strip():
            rospy.logwarn("Empty text provided to text2audio")
            return
        
        try:
            self._text2audio_rest(text)
        except Exception as e:
            rospy.logerr(f"Deepgram TTS error: {e}")
    
    def _text2audio_rest(self, text):
        """Convert text to speech using Deepgram REST API"""
        try:
            # Prepare request
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {"text": text}
            
            # Make request with model parameter
            params = {"model": self.tts_model}
            
            response = requests.post(
                self.tts_url,
                headers=headers,
                json=payload,
                params=params,
                stream=True,
                timeout=30
            )
            
            response.raise_for_status()
            
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_filename = temp_file.name
                
                # Write audio data to file
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        temp_file.write(chunk)
            
            # Play the audio file
            self._play_audio_file(temp_filename)
            
            # Clean up
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
        except requests.exceptions.RequestException as e:
            rospy.logerr(f"Deepgram TTS API error: {e}")
            raise
        except Exception as e:
            rospy.logerr(f"Deepgram TTS general error: {e}")
            raise
    
    def _play_audio_file(self, filename):
        """Play audio file using available system players"""
        try:
            # Try different audio players in order of preference
            players = [
                f"mpg321 '{filename}' > /dev/null 2>&1",
                f"mplayer '{filename}' > /dev/null 2>&1", 
                f"ffplay -nodisp -autoexit '{filename}' > /dev/null 2>&1",
                f"cvlc --play-and-exit '{filename}' > /dev/null 2>&1"
            ]
            
            for player_cmd in players:
                exit_code = os.system(player_cmd)
                if exit_code == 0:
                    return  # Successfully played
            
            rospy.logwarn("No audio player found. Please install mpg321, mplayer, ffmpeg, or vlc")
            
        except Exception as e:
            rospy.logerr(f"Audio playback error: {e}")
    
    def audio2text(self, timeout=10, listen_phrase="", use_punctuation_end=False):
        try:
            
            rospy.loginfo("Using direct REST API for command transcription")
            
            # Calls the microphone recording and transcription logic
            return self._audio2text_microphone(timeout)
                
        except Exception as e:
            rospy.logerr(f"Audio2text error: {e}")
            self._text2audio_rest("I'm experiencing technical difficulties. Please try speaking again.")
            return ""
        
    #     """
    #     Convert audio to text using streaming WebSocket API with automatic speech detection
        
    #     Args:
    #         timeout (int): Maximum time to listen in seconds
    #         listen_phrase (str): Phrase to announce before listening
    #         use_punctuation_end (bool): If True, immediately use interim results ending with punctuation.
    #                                   If False, wait for final results from Deepgram.
            
    #     Returns:
    #         str: Transcribed text or None if no speech detected
    #     """
    #     try:
    #         # Try WebSocket with reasonable timeout
    #         rospy.logdebug("Attempting WebSocket connection for speech recognition")
    #         try:
    #             # Run the streaming transcription
    #             loop = asyncio.new_event_loop()
    #             asyncio.set_event_loop(loop)
    #             result = loop.run_until_complete(self._stream_audio_to_websocket(timeout, use_punctuation_end))
    #             loop.close()
                
    #             if result is not None and result.strip():
    #                 rospy.loginfo(f"WebSocket transcription successful: {result}")
    #                 return result
    #             else:
    #                 rospy.loginfo("WebSocket returned empty result, falling back to REST API")
                    
    #         except Exception as e:
    #             rospy.loginfo(f"WebSocket attempt failed: {e}, falling back to REST API")
            
    #         # If WebSocket fails, fall back to REST API
    #         rospy.loginfo("Using REST API for transcription")
    #         return self._audio2text_microphone(timeout)
                
    #     except Exception as e:
    #         rospy.logerr(f"Audio2text error: {e}")
    #         self._text2audio_rest("I'm experiencing technical difficulties. Please try speaking again.")
    #         return ""
    
    # async def _stream_audio_to_websocket(self, timeout=10, use_punctuation_end=False):
    #     """
    #     Stream audio to Deepgram WebSocket for real-time transcription
        
    #     Args:
    #         timeout (int): Maximum time to listen in seconds
    #         use_punctuation_end (bool): If True, immediately use interim results ending with punctuation
            
    #     Returns:
    #         str: Final transcribed text
    #     """
    #     # WebSocket URL with parameters for streaming
    #     params = {
    #         'model': self.stt_model,
    #         'language': self.language,
    #         'encoding': 'linear16',
    #         'sample_rate': '16000',
    #         'channels': '1',
    #         'endpointing': '500',  # Increased back to 500ms for better stability
    #         'vad_events': 'true',  # Voice activity detection
    #         'interim_results': 'true',  # Enable interim results
    #         'smart_format': 'true',
    #         'utterance_end_ms': '1500'  # End utterance after 1.5s of silence
    #     }
        
    #     # Build WebSocket URL
    #     ws_url = f"wss://api.deepgram.com/v1/listen?" + "&".join([f"{k}={v}" for k, v in params.items()])
        
    #     headers = {
    #         "Authorization": f"Token {self.api_key}",
    #     }
        
    #     final_transcript = ""
    #     last_interim = ""
    #     self._stop_recording = False
        
    #     try:
    #         # Initialize audio recording
    #         import pyaudio
    #         audio = pyaudio.PyAudio()
            
    #         # Audio configuration
    #         CHUNK = 1024
    #         FORMAT = pyaudio.paInt16
    #         CHANNELS = 1
    #         RATE = 16000
            
    #         stream = audio.open(
    #             format=FORMAT,
    #             channels=CHANNELS,
    #             rate=RATE,
    #             input=True,
    #             frames_per_buffer=CHUNK
    #         )
            
    #         async with aiohttp.ClientSession() as session:
    #             # Add connection timeout to prevent hanging
    #             connector_timeout = aiohttp.ClientTimeout(connect=3.0, total=timeout + 5)
    #             try:
    #                 # Try to establish WebSocket connection with timeout
    #                 async with session.ws_connect(
    #                     ws_url, 
    #                     headers=headers, 
    #                     timeout=connector_timeout
    #                 ) as ws:
    #                     rospy.loginfo("WebSocket connection established for streaming STT")
                        
    #                     # Start recording and transcription tasks
    #                     recording_task = asyncio.create_task(
    #                         self._send_audio_stream(stream, ws, timeout)
    #                     )
    #                     transcription_task = asyncio.create_task(
    #                         self._receive_transcripts(ws, use_punctuation_end)
    #                     )
                        
    #                     # Wait for either task to complete
    #                     done, pending = await asyncio.wait(
    #                         [recording_task, transcription_task],
    #                         return_when=asyncio.FIRST_COMPLETED,
    #                         timeout=timeout + 2  # Give extra time for cleanup
    #                     )
                        
    #                     # Cancel remaining tasks
    #                     for task in pending:
    #                         task.cancel()
    #                         try:
    #                             await task
    #                         except asyncio.CancelledError:
    #                             pass
                        
    #                     # Get results from completed tasks
    #                     final_transcript = ""
    #                     last_interim = ""
    #                     for task in done:
    #                         try:
    #                             if task == transcription_task:
    #                                 result = await task
    #                                 if isinstance(result, tuple) and len(result) == 2:
    #                                     final_transcript, last_interim = result
    #                                 else:
    #                                     final_transcript = result or ""
    #                             elif task == recording_task:
    #                                 speech_detected = await task
    #                         except Exception as e:
    #                             rospy.logerr(f"Error getting task result: {e}")
                        
    #                     # Close the stream properly
    #                     await self._close_websocket(ws)
                
    #             except asyncio.TimeoutError:
    #                 rospy.logerr(f"WebSocket connection timeout after 3 seconds")
    #                 return None
    #             except aiohttp.ClientError as e:
    #                 rospy.logerr(f"WebSocket connection error: {e}")
    #                 return None
                    
    #     except Exception as e:
    #         rospy.logerr(f"Error in streaming WebSocket: {e}")
    #         return None
    #     finally:
    #         # Cleanup audio resources
    #         try:
    #             if 'stream' in locals():
    #                 stream.stop_stream()
    #                 stream.close()
    #             if 'audio' in locals():
    #                 audio.terminate()
    #         except Exception:
    #             pass
        
    #     if final_transcript and final_transcript.strip():
    #         rospy.loginfo(f"Final transcript: {final_transcript}")
    #         return final_transcript.strip()
    #     elif last_interim and len(last_interim.strip()) > 0 and len(last_interim.strip().split()) >= 2:
    #         # If we have meaningful interim text (at least 2 words) but no final, use the interim
    #         rospy.loginfo(f"Using interim result as final: {last_interim}")
    #         return last_interim.strip()
    #     else:
    #         rospy.logwarn("No speech detected or transcription failed")
    #         return None
    
    # async def _send_audio_stream(self, stream, ws, timeout):
    #     """
    #     Send audio stream to WebSocket
        
    #     Args:
    #         stream: PyAudio stream object
    #         ws: WebSocket connection
    #         timeout (int): Maximum recording time
            
    #     Returns:
    #         bool: True if speech was detected
    #     """
    #     start_time = time.time()
    #     speech_detected = False
        
    #     try:
    #         while time.time() - start_time < timeout:
    #             # Read audio chunk
    #             data = stream.read(1024, exception_on_overflow=False)
                
    #             # Send audio data to WebSocket
    #             await ws.send_bytes(data)
                
    #             # Brief pause to prevent overwhelming the API
    #             await asyncio.sleep(0.01)
                
    #             # Check if we should stop (this would be set by transcript receiver)
    #             if hasattr(self, '_stop_recording') and self._stop_recording:
    #                 break
                    
    #             speech_detected = True
                
    #     except Exception as e:
    #         rospy.logerr(f"Error sending audio stream: {e}")
        
    #     return speech_detected
    
    # async def _receive_transcripts(self, ws, use_punctuation_end=False):
    #     """
    #     Receive and process transcripts from WebSocket
        
    #     Args:
    #         ws: WebSocket connection
    #         use_punctuation_end (bool): If True, immediately use interim results ending with punctuation
            
    #     Returns:
    #         tuple: (final_transcript, last_interim) - Returns both final and last interim results
    #     """
    #     final_transcript = ""
    #     last_interim = ""
    #     last_update_time = time.time()
    #     silence_threshold = 2.0  # Increased to 2.0 seconds for more natural pauses
    #     no_speech_timeout = 5.0  # Timeout if no speech detected at all
        
    #     try:
    #         async for message in ws:
    #             current_time = time.time()
                
    #             # Check for timeout with no new transcripts (silence detection) at every iteration
    #             # Only trigger silence detection if enabled and we have substantial content (more than 2 words)
    #             # and there's been actual silence (no new updates)
    #             if (self.enable_silence_detection and 
    #                 last_interim and 
    #                 len(last_interim.strip().split()) >= 2 and 
    #                 (current_time - last_update_time > silence_threshold)):
    #                 rospy.loginfo(f"Silence detected ({silence_threshold}s), using last interim result: '{last_interim}'")
    #                 if not final_transcript:
    #                     final_transcript = last_interim
    #                 self._stop_recording = True
    #                 break
                
    #             # Check for overall timeout (no speech at all)
    #             if current_time - last_update_time > no_speech_timeout and not last_interim:
    #                 rospy.logwarn(f"No speech detected for {no_speech_timeout}s, ending transcription")
    #                 break
                
    #             if message.type == aiohttp.WSMsgType.TEXT:
    #                 try:
    #                     response = json.loads(message.data)
                        
    #                     if response.get("type") == "Results":
    #                         channel = response.get("channel", {})
    #                         alternatives = channel.get("alternatives", [])
                            
    #                         if alternatives:
    #                             transcript = alternatives[0].get("transcript", "")
    #                             is_final = channel.get("is_final", False)
    #                             confidence = alternatives[0].get("confidence", 0.0)
                                
    #                             if transcript.strip():
    #                                 # Update last_update_time only for substantial changes
    #                                 if not last_interim or len(transcript) > len(last_interim):
    #                                     last_update_time = time.time()
                                    
    #                                 if is_final:
    #                                     rospy.loginfo(f"Final transcript: {transcript}")
    #                                     final_transcript = transcript
    #                                     # Stop recording after getting final result
    #                                     self._stop_recording = True
    #                                     break
    #                                 else:
    #                                     # Interim result - use if confidence is high enough
    #                                     last_interim = transcript
    #                                     rospy.loginfo(f"Interim transcript (conf: {confidence:.2f}): {transcript}")
                                        
    #                                     # If interim result ends with sentence-ending punctuation and feature is enabled, use it immediately
    #                                     if use_punctuation_end and transcript.strip().endswith(('.', '?', '!')):
    #                                         rospy.loginfo(f"Interim result ends with punctuation - using immediately: {transcript}")
    #                                         final_transcript = transcript
    #                                         self._stop_recording = True
    #                                         break
                                        
    #                                     # If we have high confidence interim with substantial content, 
    #                                     # use it after a short delay to allow for final result
    #                                     if (confidence >= 0.95 and 
    #                                         len(transcript.strip().split()) >= 4 and
    #                                         current_time - last_update_time > 1.0):
    #                                         rospy.loginfo(f"High confidence interim result with delay - using as final: {transcript}")
    #                                         final_transcript = transcript
    #                                         self._stop_recording = True
    #                                         break
                                        
    #                     elif response.get("type") == "SpeechStarted":
    #                         rospy.loginfo("Speech detected, listening...")
    #                         last_update_time = time.time()
                            
    #                     elif response.get("type") == "UtteranceEnd":
    #                         rospy.loginfo("End of utterance detected")
    #                         # If we have interim text but no final, use the last interim
    #                         if not final_transcript and last_interim and len(last_interim.strip().split()) >= 2:
    #                             final_transcript = last_interim
    #                             rospy.loginfo(f"Using last interim as final after utterance end: {final_transcript}")
    #                         self._stop_recording = True
    #                         break
                            
    #                 except json.JSONDecodeError as e:
    #                     rospy.logerr(f"Error decoding JSON message: {e}")
    #                 except KeyError as e:
    #                     rospy.logerr(f"Key error in response: {e}")
                        
    #             elif message.type == aiohttp.WSMsgType.ERROR:
    #                 rospy.logerr(f"WebSocket error: {message.data}")
    #                 break
                    
    #     except Exception as e:
    #         rospy.logerr(f"Error receiving transcripts: {e}")
        
    #     # Return both final transcript and last interim for fallback
    #     return (final_transcript, last_interim)
    
    # async def _close_websocket(self, ws):
    #     """
    #     Properly close the WebSocket connection
        
    #     Args:
    #         ws: WebSocket connection
    #     """
    #     try:
    #         close_msg = '{"type": "CloseStream"}'
    #         await ws.send_str(close_msg)
    #         await ws.close()
    #     except Exception as e:
    #         rospy.logerr(f"Error closing WebSocket: {e}")

    def _audio2text_microphone(self, timeout=10):
        """Capture audio from microphone and transcribe using Deepgram REST API"""
        recognized_text = ""
        
        try:
            # Record audio from microphone
            audio_data = self._record_audio(timeout)
            
            if not audio_data:
                rospy.logwarn("No audio recorded")
                # self._text2audio_rest("I couldn't hear you clearly. Please try speaking again.")
                return ""
            
            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name
                self._save_audio_to_wav(audio_data, temp_filename)
            
            # Transcribe using Deepgram API
            recognized_text = self._transcribe_audio_file(temp_filename)
            
            # Clean up
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            
        except Exception as e:
            rospy.logerr(f"Microphone transcription error: {e}")
            # self._text2audio_rest("I'm having trouble with the microphone. Please check your audio and try again.")
        
        return recognized_text
    
    def _record_audio(self, duration):
        """Record audio from microphone"""
        try:
            # Open microphone stream
            stream = self.p.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
                
            )
            
            rospy.loginfo("Recording audio...")
            
            frames = []
            for _ in range(0, int(self.sample_rate / self.chunk_size * duration)):
                if rospy.is_shutdown():
                    break
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)
            
            # Clean up
            stream.stop_stream()
            stream.close()
            
            return b''.join(frames)
            
        except Exception as e:
            rospy.logerr(f"Audio recording error: {e}")
            return None
    
    def _save_audio_to_wav(self, audio_data, filename):
        """Save raw audio data to WAV file"""
        try:
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.p.get_sample_size(self.audio_format))
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data)
        except Exception as e:
            rospy.logerr(f"Error saving audio to WAV: {e}")
            raise
    
    def _transcribe_audio_file(self, filename):
        """Transcribe audio file using Deepgram REST API"""
        try:
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "audio/wav"
            }
            
            params = {
                "model": self.stt_model,
                "language": self.language,
                "punctuate": "true",
                "smart_format": "true"
            }
            
            with open(filename, 'rb') as audio_file:
                response = requests.post(
                    self.stt_url,
                    headers=headers,
                    params=params,
                    data=audio_file,
                    timeout=30
                )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract transcript from response
            if 'results' in result and 'channels' in result['results']:
                channels = result['results']['channels']
                if channels and 'alternatives' in channels[0]:
                    alternatives = channels[0]['alternatives']
                    if alternatives and 'transcript' in alternatives[0]:
                        transcript = alternatives[0]['transcript'].strip()
                        confidence = alternatives[0].get('confidence', 0.0)
                        
                        if transcript and confidence > 0.7:
                            rospy.loginfo(f"Recognized (confidence: {confidence:.2f}): {transcript}")
                            return transcript
                        # else:
                        #     rospy.loginfo(f"Low confidence ({confidence:.2f}) or empty transcript")
                        #     if confidence > 0.0:  # Low confidence but some speech detected
                        #         self._text2audio_rest("I'm not sure I understood you correctly. Could you please repeat that?")
                        #     else:  # No speech detected
                        #         self._text2audio_rest("I didn't hear anything. Please speak a bit louder.")
            
            rospy.logwarn("No valid transcript found in response")
            # self._text2audio_rest("I'm having trouble understanding you. Please try speaking again.")
            return ""
            
        except requests.exceptions.RequestException as e:
            rospy.logerr(f"Deepgram STT API error: {e}")
            self._text2audio_rest("I'm having connection issues. Please try again in a moment.")
            return ""
        except Exception as e:
            rospy.logerr(f"Transcription error: {e}")
            self._text2audio_rest("Something went wrong with speech recognition. Please try again.")
            return ""
    
    def audio2text_vosk(self, timeout=10):
        """
        Compatibility method for Vosk-style audio recognition
        Redirects to Deepgram audio2text method
        """
        return self.audio2text(timeout)
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'p') and self.p:
                self.p.terminate()
            rospy.logdebug("Deepgram utilities cleaned up")
        except Exception as e:
            rospy.logerr(f"Cleanup error: {e}")


# Global instance for easy access
_deepgram_utils = None
_init_lock = threading.Lock()

def get_deepgram_utils():
    """
    Get the global DeepgramUtils instance (singleton)
    
    Returns:
        DeepgramUtils: The global Deepgram utilities instance
    """
    global _deepgram_utils
    
    with _init_lock:
        if _deepgram_utils is None:
            try:
                _deepgram_utils = DeepgramUtils()
            except Exception as e:
                rospy.logerr(f"Failed to initialize Deepgram utilities: {e}")
                return None
    
    return _deepgram_utils


# Convenience functions for backward compatibility
def text2audio(text, lang="en"):
    """Convenience function for text-to-speech"""
    utils = get_deepgram_utils()
    if utils:
        utils.text2audio(text, lang)
    else:
        rospy.logerr("Deepgram utilities not available")


def audio2text(timeout=10, listen_phrase="Please speak now", use_punctuation_end=False):
    """Convenience function for speech recognition with optimized timeout"""
    utils = get_deepgram_utils()
    if utils:
        return utils.audio2text(timeout, listen_phrase, use_punctuation_end)
    else:
        rospy.logerr("Deepgram utilities not available")
        return ""


def audio2text_vosk(timeout=10):
    """Convenience function for Vosk compatibility with optimized timeout"""
    return audio2text(timeout)


# Test function
def test_deepgram_utils():
    """Test function for Deepgram utilities"""
    rospy.init_node('deepgram_utils_test', anonymous=True)
    
    utils = get_deepgram_utils()
    if not utils:
        rospy.logerr("Failed to initialize Deepgram utilities")
        return
    
    # Test TTS
    rospy.loginfo("Testing text-to-speech...")
    utils.text2audio("Hello, this is a test of Deepgram text to speech functionality.")
    
    # Test STT
    rospy.loginfo("Testing speech recognition...")
    result = utils.audio2text(10, "Please say something for the speech recognition test.", False)
    rospy.loginfo(f"Recognition result: {result}")
    
    utils.cleanup()


if __name__ == "__main__":
    test_deepgram_utils()
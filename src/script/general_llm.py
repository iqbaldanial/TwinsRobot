import os
import rospy
import openai

class GeneralLLM:
   
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        # self.system_prompt = "You are a helpful robot assistant named Bumblebee. Keep your answers concise and conversational."
        self.pickable_items = ["crumpled paper", "plastic bottle", "drink carton", "dustbin", "context_item"]

        # 1. System Prompt for Conversational Tone and Tool Use
        self.system_prompt = (
            "You are a helpful and concise robot assistant named Bumblebee. Respond conversationally to all requests. "
            "If the user asks for a physical action, you MUST use the 'initiate_arm_task' tool to extract the action, object, and target."
            "If the object is ambiguous (e.g., 'it', 'that'), always use 'context_item' for the object field. "
        )

        # 2. Tool Definition (JSON Schema for structured output)
        self.tools = [{
            "type": "function",
            "function": {
                "name": "initiate_arm_task",
                "description": "Initiates a physical pick, move, or place action by the robot arm.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "The specific movement action, e.g., 'pick', 'place', 'hold', or 'grab'."
                        },
                        "object": {
                            "type": "string",
                            "enum": self.pickable_items, 
                            "description": "The primary object being manipulated (e.g., 'plastic bottle', 'drink carton', 'crumpled paper')."
                        },
                        "target": {
                            "type": "string",
                            "enum": ["dustbin", "ground", "rest", "default"], 
                            "description": "The destination of the object, if applicable (e.g., 'dustbin')."
                        }
                    },
                    "required": ["action", "object"]
                }
            }
        }]

        # initialize OpenAI Client
        openai_api_key = os.getenv("OPENAI_API_KEYSA")
        if not openai_api_key:
            rospy.logerr("OPENAI_API_KEYSA environment variable not set.")
            raise ValueError("OpenAI API key not found")
        
        try:
            # The openai client automatically handles the environment variable
            self.client = openai.OpenAI(api_key=openai_api_key)
            rospy.loginfo(f"OpenAI LLM initialized with model: {self.model}")
        except Exception as e:
            rospy.logerr(f"Failed to initialize OpenAI client: {e}")
            raise

    def answer_question(self, question):
        # """Answers a question using the configured OpenAI model."""
        # messages = [
        #     {"role": "system", "content": self.system_prompt},
        #     {"role": "user", "content": question}
        # ]
        
        # try:
        #     response = self.client.chat.completions.create(
        #         model=self.model,
        #         messages=messages,
        #         temperature=0.7,
        #         max_tokens=50
        #     )
        #     return response.choices[0].message.content.strip()
        # except Exception as e:
        #     rospy.logerr(f"OpenAI question answering error: {e}")
        #     return "I'm having trouble connecting to my brain. Please try again."

        # This function is retained for generality but is not strictly used by main.py
        # which uses process_command directly.
        
        # Simple path for direct conversational answering
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question}]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=50
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            rospy.logerr(f"OpenAI question answering error: {e}")
            return "I'm having trouble connecting to my brain."
    
    def process_command(self, command):
        # """Processes a user command, leveraging answer_question for conversational response."""
        # # rospy.loginfo(f"Processing command: {command}")
        # return self.answer_question(command)

        """Answers a question, potentially with a tool call."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": command}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=50
            )
            
            # Returns the message object which may contain 'content' (text) or 'tool_calls' (JSON structure)
            return response.choices[0].message
            
        except Exception as e:
            rospy.logerr(f"OpenAI question answering error: {e}")
            # Returns a dummy object on failure to prevent crashes
            return type('Obj', (object,), {'content': "I'm having trouble connecting to my brain. Please try speaking again."})()
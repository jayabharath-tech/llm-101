"""
Module to generate stories based on user input.
"""
import os

from groq import Groq
import streamlit as st
from dotenv import load_dotenv, find_dotenv

MODEL = "llama-3.3-70b-versatile"

_ = load_dotenv(find_dotenv(), override=True)

# Initialize Groq client with API key from environment
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

st.title("Story-Teller chatbot")


def check_moderation(text: str) -> bool:
    """
    Check if the input text violates content policy using the model.
    """

    try:
        _message = [
            {"role": "system",
             "content": "You are a content moderation system. Respond ONLY with 'SAFE' if the content is appropriate, or 'UNSAFE' if it violates policies (contains hate speech, violence, explicit content, etc.)."
             },
            {
                "role": "user",
                "content": f"Is this text safe? Text: {text}"
            }
        ]
        response = client.chat.completions.create(
            model=MODEL,
            messages=_message,
            temperature=0.0,
            max_tokens=10
        )

        # Clean and check the response
        result = response.choices[0].message.content.strip().upper()
        # print(f"Moderation result for '{text}': {result}")  # Debug logging
        
        # Return True if safe, False if unsafe
        return "SAFE" in result

    except Exception as e:
        print(f"Moderation check failed: {e}")
        return True  # Default to allowing if moderation check fails (fail open)


def get_chat_completion(messages, model=MODEL, temperature=0):
    """
    Get a chat completion from the Groq model with streaming.
    """
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True
        )
        # Yield chunks for Streamlit streaming
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        print(f"Chat completion failed: {e}")
        yield "Error: Failed to generate response."


class StoryTeller:
    # Maximum number of conversation turns to keep in context (excluding system message and few-shot examples)
    MAX_CONTEXT_MESSAGES = 10  # 5 user-assistant pairs

    def __init__(self, st):
        self.st = st
        self.initialise_session()

    def initialise_session(self):
        system_message = (
            """
        You are a creative storyteller who generates stories based on user requests. \
        
        Guidelines:
        1. Accept story requests (any genre: horror, romance, sci-fi, fantasy, etc.)
        2. Accept follow-up questions about stories you've told (e.g., "make it longer", "change the character", "what happens next", "add more details")
        3. Reject only non-story requests (e.g., jokes, general knowledge questions, math problems)
        4. Keep stories under 200 words
        5. Output only the story content or your response
        """
        )

        # Set a default model
        if "groq_model" not in self.st.session_state:
            self.st.session_state["groq_model"] = MODEL

        # Initialize chat history for model context
        if "messages" not in self.st.session_state:
            # few shots - demonstrate both story generation and follow-up handling
            self.st.session_state.messages = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": "tell me a horror story"
                },
                {
                    "role": "assistant",
                    "content": "The darkness in the attic had been quiet for years, but on my 18th birthday, it began to whisper my name. When I finally worked up the courage to investigate, I found a room filled with old photographs of myself as a child."
                },
                {
                    "role": "user",
                    "content": "make it longer and add a character name"
                },
                {
                    "role": "assistant",
                    "content": "The darkness in Emily's attic had been quiet for years, but on her 18th birthday, it began to whisper her name. When she finally worked up the courage to investigate, she found a room filled with old photographs of herself as a child. In each photo, a shadowy figure stood behind her, getting closer with every picture. The final photograph was dated tomorrow."
                },
                {
                    "role": "user",
                    "content": "what happens next?"
                },
                {
                    "role": "assistant",
                    "content": "Emily's hands trembled as she heard footsteps on the attic stairs. The shadow from the photographs materialized behind her, its cold breath on her neck. 'I've been waiting,' it whispered, 'for you to finally see me.' As Emily turned, she realized the shadow wore her face."
                },
                {
                    "role": "user",
                    "content": "tell me a joke"
                },
                {
                    "role": "assistant",
                    "content": "I am limited to story telling only."
                },
                {
                    "role": "user",
                    "content": "what is python?"
                },
                {
                    "role": "assistant",
                    "content": "I am limited to story telling only."
                }

            ]

        # Initialize UI display history (includes flagged interactions)
        if "ui_messages" not in self.st.session_state:
            self.st.session_state.ui_messages = []

        # Display chat messages from history on app rerun
        for message in self.st.session_state.ui_messages:
            with self.st.chat_message(message["role"]):
                self.st.markdown(message["content"])

    def get_limited_context(self):
        """
        Get a limited context window for the LLM.
        Keeps: system message + few-shot examples + recent conversation
        """
        messages = self.st.session_state.messages
        
        # Find where few-shot examples end (after the last "what is python?" example)
        # Few-shot examples are the first 11 messages (1 system + 10 examples)
        few_shot_end_index = 11
        
        # Get system message and few-shot examples
        base_messages = messages[:few_shot_end_index]
        
        # Get recent conversation (everything after few-shot examples)
        recent_conversation = messages[few_shot_end_index:]
        
        # Limit recent conversation to MAX_CONTEXT_MESSAGES
        if len(recent_conversation) > self.MAX_CONTEXT_MESSAGES:
            recent_conversation = recent_conversation[-self.MAX_CONTEXT_MESSAGES:]
        
        # Combine base messages with limited recent conversation
        return base_messages + recent_conversation

    def start(self):
        # Accept user input
        if prompt := self.st.chat_input("Add text here"):

            # Display user message in chat message container
            with self.st.chat_message("user"):
                self.st.markdown(prompt)

            # Add user message to UI history (always shown)
            self.st.session_state.ui_messages.append({"role": "user", "content": prompt})

            # Moderation check
            if not check_moderation(prompt):
                moderation_flagged_response = "⚠ I'm afraid I can't assist with that. Input flagged by moderation. Please rephrase and try again."
                with self.st.chat_message("assistant"):
                    self.st.markdown(moderation_flagged_response)

                # Add moderation response to UI history (shown but not sent to model)
                self.st.session_state.ui_messages.append({"role": "assistant", "content": moderation_flagged_response})
                return

            # Add user message to model context (only safe messages)
            self.st.session_state.messages.append({"role": "user", "content": prompt})

            # Display assistant response in chat message container
            with self.st.chat_message("assistant"):
                # Use limited context for the LLM
                limited_context = self.get_limited_context()
                response = self.st.write_stream(
                    get_chat_completion(
                        limited_context,
                        model=self.st.session_state["groq_model"]
                    )
                )

            # Add response to both UI and model context
            self.st.session_state.messages.append({"role": "assistant", "content": response})
            self.st.session_state.ui_messages.append({"role": "assistant", "content": response})


story_teller = StoryTeller(st)
story_teller.start()

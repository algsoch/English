"""
AI Engine Module - Handles interaction with Google Gemini API
"""
import logging
import json
import aiohttp
import asyncio
import os
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import time
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants - Load from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash") 
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))

# Interview topics and skill levels
INTERVIEW_TOPICS = {
    "machine_learning": "Machine Learning",
    "data_science": "Data Science",
    "deep_learning": "Deep Learning",
    "python": "Python Programming",
    "github": "GitHub & Version Control",
    "data_engineering": "Data Engineering",
    "cloud_computing": "Cloud Computing",
    "devops": "DevOps",
    "software_engineering": "Software Engineering",
    "algorithms": "Algorithms & Data Structures",
    "statistics": "Statistics & Probability",
    "sql": "SQL & Databases",
    "big_data": "Big Data Technologies"
}

SKILL_LEVELS = {
    "beginner": "Beginner (0-1 years experience)",
    "intermediate": "Intermediate (1-3 years experience)",
    "advanced": "Advanced (3-5 years experience)",
    "expert": "Expert (5+ years experience)",
    "learning": "Learning Mode (simplified explanations)",
    "child": "Child-Friendly (5-year-old explanations)"
}

# Interview modes
INTERVIEW_MODES = {
    "interview": "Professional Interview",
    "learning": "Learning & Explanations",
    "child": "Child-Friendly Explanations"
}

# Cache for conversation history - using UUIDs as keys
conversation_cache = {}

# Cache for user information
user_info_cache = {}

class GeminiClient:
    """
    Client for interacting with Google Gemini API for model inference
    """
    def __init__(self, model: str = GEMINI_MODEL, api_key: str = GEMINI_API_KEY):
        self.model = model
        self.api_key = api_key
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.session = None

    async def ensure_session(self):
        """Ensure the aiohttp session is created"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def generate_response(
        self, 
        prompt: str, 
        conversation_id: Optional[str] = None,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate response using the Gemini API
        
        Args:
            prompt: User input text
            conversation_id: Optional ID to maintain conversation context
            temperature: Controls creativity (0.0-1.0)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt to set context
            
        Returns:
            Tuple containing response text and metadata
        """
        await self.ensure_session()
        
        # Generate a conversation ID if none is provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            
        # Get conversation history if conversation_id is provided
        history = []
        if conversation_id in conversation_cache:
            history = conversation_cache[conversation_id]
        
        logger.info(f"Using conversation ID: {conversation_id} with {len(history)} messages in history")
        
        # Prepare request payload
        payload = {
            "contents": []
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["contents"].append({
                "role": "user",
                "parts": [{"text": f"System instructions: {system_prompt}\n\nUser: {prompt}"}]
            })
        else:
            # Add conversation history if available
            if history:
                payload["contents"].extend(history)
            
            # Add current user prompt
            payload["contents"].append({
                "role": "user",
                "parts": [{"text": prompt}]
            })
        
        # Add generation config
        payload["generationConfig"] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "topP": 0.95,
            "topK": 40
        }
        
        # Build URL with API key
        url = f"{self.api_url}?key={self.api_key}"
        
        try:
            logger.info(f"Sending request to Gemini API: {prompt[:50]}...")
            async with self.session.post(url, json=payload) as response:
                response_text = await response.text()
                
                if response.status != 200:
                    logger.error(f"Gemini API error: {response.status} - {response_text}")
                    
                    # Log the full payload for debugging (without sensitive info)
                    debug_payload = payload.copy()
                    logger.debug(f"Request payload: {json.dumps(debug_payload)}")
                    
                    raise Exception(f"Gemini API error: {response.status} - {response_text}")
                
                response_data = json.loads(response_text)
                
                # Extract the response text
                result_text = ""
                if "candidates" in response_data and len(response_data["candidates"]) > 0:
                    candidate = response_data["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        for part in candidate["content"]["parts"]:
                            if "text" in part:
                                result_text += part["text"]
                
                # Update conversation history
                if not system_prompt:  # Only update history if not using system prompt approach
                    if conversation_id not in conversation_cache:
                        conversation_cache[conversation_id] = []
                    
                    # Add user prompt to history
                    conversation_cache[conversation_id].append({
                        "role": "user",
                        "parts": [{"text": prompt}]
                    })
                    
                    # Add assistant response to history
                    conversation_cache[conversation_id].append({
                        "role": "model",
                        "parts": [{"text": result_text}]
                    })
                    
                    # Limit history size to avoid token limits
                    if len(conversation_cache[conversation_id]) > 10:
                        # Keep only the last 10 messages
                        conversation_cache[conversation_id] = conversation_cache[conversation_id][-10:]
                
                return result_text, response_data
                
        except Exception as e:
            logger.error(f"Error generating response from Gemini: {e}")
            return "I'm sorry, I'm having trouble generating a response right now. Please try again later.", {"error": str(e)}

# Initialize the Gemini client
gemini_client = GeminiClient()

# Create interview system prompts based on topic, skill level, and mode
def get_interview_system_prompt(topic: str, skill_level: str, mode: str = "interview", user_info: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate an appropriate system prompt for the selected interview topic and skill level
    
    Args:
        topic: The interview topic (e.g., "machine_learning", "python")
        skill_level: The skill level (e.g., "beginner", "expert")
        mode: The interview mode (e.g., "interview", "learning", "child")
        user_info: Optional user information including name and email
        
    Returns:
        Customized system prompt for the interview
    """
    topic_display = INTERVIEW_TOPICS.get(topic, topic.replace("_", " ").title())
    level_display = SKILL_LEVELS.get(skill_level, skill_level)
    
    # Get user name if available
    user_name = "candidate"
    if user_info and "name" in user_info:
        user_name = user_info["name"]
    
    # Base prompt varies by mode
    if mode == "learning":
        base_prompt = f"""
You are TechTeacher AI, an educational assistant specializing in {topic_display}.
You are teaching {user_name} about {topic_display} at a {level_display} level.

Your task is to:
1. Explain technical concepts related to {topic_display} in a clear, accessible way
2. Break down complex ideas into simpler components
3. Provide helpful analogies and examples
4. Answer questions thoroughly but simply
5. Suggest follow-up topics to explore
6. Relate new concepts to ones previously discussed
7. Be encouraging and supportive

Avoid unnecessary jargon. When you must use technical terms, explain them.
Use practical examples that relate to real-world applications.
"""
    elif mode == "child":
        base_prompt = f"""
You are KidTech AI, an educational assistant explaining technology to young children.
You are teaching {user_name} about {topic_display} as if they were a 5-year-old child.

Your task is to:
1. Use extremely simple language a 5-year-old would understand
2. Create fun, colorful analogies related to everyday things children know
3. Break complex ideas into tiny, simple pieces
4. Keep explanations very short - just 2-3 sentences at a time
5. Use lots of examples with characters and stories
6. Be very encouraging and exciting
7. Avoid ALL technical jargon and complex vocabulary

Make learning fun, focus on building curiosity rather than technical accuracy.
"""
    else:  # Default interview mode
        base_prompt = f"""
You are TechInterviewer AI, an advanced technical interviewer specializing in {topic_display}.
You are conducting a {level_display} level interview with {user_name}.

Your task is to:
1. Ask relevant technical questions about {topic_display} suitable for a {level_display} candidate
2. Evaluate the candidate's responses and provide constructive feedback
3. Probe deeper with follow-up questions when appropriate
4. Challenge their understanding with increasingly complex scenarios
5. Provide code examples or ask for code solutions when relevant
6. Be professional, respectful and encouraging
7. At appropriate intervals, provide a brief assessment of the candidate's strengths and areas for improvement

Start with an introduction and a relatively straightforward question to establish baseline knowledge.
Progress to more complex topics based on the candidate's responses.
"""

    # Add topic-specific instructions
    if topic == "machine_learning":
        base_prompt += """
Focus on: algorithms (regression, classification, clustering), model evaluation metrics, feature engineering, 
overfitting/underfitting, regularization techniques, and practical implementation considerations.
"""
    elif topic == "data_science":
        base_prompt += """
Focus on: data cleaning, exploratory data analysis, statistical testing, visualization techniques,
experimental design, A/B testing, and communicating insights from data.
"""
    elif topic == "deep_learning":
        base_prompt += """
Focus on: neural network architectures, activation functions, backpropagation, optimization algorithms,
transfer learning, CNNs, RNNs, transformers, and frameworks (TensorFlow/PyTorch).
"""
    elif topic == "python":
        base_prompt += """
Focus on: core language features, data structures, OOP concepts, functional programming,
performance optimization, libraries/frameworks, testing, and best practices.
"""
    elif topic == "github":
        base_prompt += """
Focus on: git workflows, branching strategies, CI/CD integration, code reviews, issue tracking,
collaboration features, resolving merge conflicts, and GitHub Actions.
"""
    elif topic == "data_engineering":
        base_prompt += """
Focus on: data pipelines, ETL processes, data warehousing, distributed computing frameworks,
data modeling, database optimization, and data quality assurance.
"""
    
    return base_prompt

async def generate_ai_response(
    prompt: str, 
    conversation_id: Optional[str] = None,
    interview_topic: str = "machine_learning",
    skill_level: str = "intermediate",
    interview_mode: str = "interview",
    user_info: Optional[Dict[str, Any]] = None,
    end_interview: bool = False
) -> str:
    """
    Generate an AI response for the interview application
    
    Args:
        prompt: User input text
        conversation_id: Optional conversation ID for context
        interview_topic: The technical subject for the interview
        skill_level: The candidate's skill level
        interview_mode: The mode of interaction (interview, learning, child)
        user_info: Optional user information including name and email
        end_interview: Whether to end the interview and provide a final assessment
        
    Returns:
        AI-generated response string
    """
    # Store user info in cache if provided
    if user_info and conversation_id:
        user_info_cache[conversation_id] = user_info
    
    # Get cached user info if available
    cached_user_info = user_info_cache.get(conversation_id, {})
    if not user_info and cached_user_info:
        user_info = cached_user_info
    
    # Get the appropriate system prompt based on topic, skill level, and mode
    system_prompt = get_interview_system_prompt(
        interview_topic, 
        skill_level, 
        interview_mode,
        user_info
    )
    
    try:
        # Generate conversation ID if none provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            
        # Check if this is the first message in the conversation
        is_first_message = conversation_id not in conversation_cache or len(conversation_cache.get(conversation_id, [])) == 0
        
        # For the first message, add an introduction and first question
        if is_first_message:
            # Customize welcome prompt based on mode
            if interview_mode == "learning":
                welcome_prompt = f"This is a learning session about {INTERVIEW_TOPICS.get(interview_topic, interview_topic)} at a {SKILL_LEVELS.get(skill_level, skill_level)} level. Please introduce yourself as an educational assistant and start explaining the fundamental concepts of this topic."
            elif interview_mode == "child":
                welcome_prompt = f"This is a child-friendly explanation of {INTERVIEW_TOPICS.get(interview_topic, interview_topic)} for a 5-year-old. Please introduce yourself and start explaining the very basics of this topic in an extremely simple way with fun examples."
            else:
                welcome_prompt = f"This is a technical interview for {INTERVIEW_TOPICS.get(interview_topic, interview_topic)} at a {SKILL_LEVELS.get(skill_level, skill_level)} level. Please introduce yourself and ask the first interview question."
            
            intro_response, _ = await gemini_client.generate_response(
                prompt=welcome_prompt,
                conversation_id=conversation_id,
                system_prompt=system_prompt
            )
            
            # Store this introduction in the conversation history
            if conversation_id not in conversation_cache:
                conversation_cache[conversation_id] = []
                
            conversation_cache[conversation_id].append({
                "role": "user",
                "parts": [{"text": welcome_prompt}]
            })
            
            conversation_cache[conversation_id].append({
                "role": "model",
                "parts": [{"text": intro_response}]
            })
            
            # If the user hasn't sent any content yet, return the introduction
            if not prompt.strip():
                return intro_response
        
        # Handle end of interview request with final assessment
        if end_interview:
            end_prompt = f"The interview/session is now ending. Please provide a comprehensive final assessment of the conversation, including: 1) A summary of topics covered, 2) Strengths demonstrated, 3) Areas for improvement, 4) Overall evaluation, and 5) Suggested resources for further learning."
            
            final_assessment, _ = await gemini_client.generate_response(
                prompt=end_prompt,
                conversation_id=conversation_id,
                system_prompt=system_prompt
            )
            
            # Add the final assessment to conversation history
            if conversation_id not in conversation_cache:
                conversation_cache[conversation_id] = []
                
            conversation_cache[conversation_id].append({
                "role": "user",
                "parts": [{"text": end_prompt}]
            })
            
            conversation_cache[conversation_id].append({
                "role": "model",
                "parts": [{"text": final_assessment}]
            })
            
            # Return the final assessment
            return final_assessment
        
        # Regular response for ongoing conversation
        response_text, metadata = await gemini_client.generate_response(
            prompt=prompt,
            conversation_id=conversation_id,
            system_prompt=system_prompt
        )
        
        # Log metadata for analysis (excluding large parts)
        if "candidates" in metadata:
            candidate_count = len(metadata["candidates"])
            logger.info(f"Response received with {candidate_count} candidates")
        
        # Return empty response if we got nothing back
        if not response_text or response_text.strip() == "":
            logger.warning("Empty response received from Gemini API")
            return "I apologize for the technical issue. Let's continue with the interview. Could you please elaborate on your previous answer?"
        
        return response_text
    except Exception as e:
        logger.error(f"Error in generate_ai_response: {e}")
        
        # Fallback response
        return "I apologize for the technical difficulty. Let's continue with the interview. Could you tell me more about your experience with this topic?"

def get_available_topics() -> Dict[str, str]:
    """Return the available interview topics"""
    return INTERVIEW_TOPICS

def get_available_skill_levels() -> Dict[str, str]:
    """Return the available skill levels"""
    return SKILL_LEVELS

def get_available_modes() -> Dict[str, str]:
    """Return the available interview modes"""
    return INTERVIEW_MODES

def get_interview_modes() -> Dict[str, str]:
    """Alias for get_available_modes for backward compatibility"""
    return get_available_modes()

def set_user_info(conversation_id: str, name: str, email: str) -> None:
    """Store user information for a conversation"""
    if not conversation_id:
        return
    
    user_info_cache[conversation_id] = {
        "name": name,
        "email": email
    }
    logger.info(f"User info saved for conversation {conversation_id}: {name}, {email}")

def get_user_info(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Get stored user information for a conversation"""
    return user_info_cache.get(conversation_id)

# Example usage
if __name__ == "__main__":
    async def test():
        response = await generate_ai_response("", interview_topic="python", skill_level="intermediate")
        print(response)
    
    asyncio.run(test())
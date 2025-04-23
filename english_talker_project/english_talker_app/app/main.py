import os
import logging
import sys
import uuid
from fastapi import FastAPI, Request, File, UploadFile, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import uvicorn
import asyncio
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables from .env file
load_dotenv()

# Add the project root to Python path to make imports work
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Also add the parent directory (english_talker_app) to Python path
app_dir = Path(__file__).resolve().parent.parent
if str(app_dir) not in sys.path:
    sys.path.append(str(app_dir))

# Try direct imports first (when run as a script)
try:
    from whisper_utils import transcribe_audio
    from ai_engine import (
        generate_ai_response, get_available_topics, get_available_skill_levels, 
        get_available_modes, GeminiClient, user_info_cache
    )
    from tts_engine import text_to_speech
    from discord_notify import send_discord_notification
except ImportError:
    # Fall back to relative imports (when imported as a module)
    try:
        from .whisper_utils import transcribe_audio
        from .ai_engine import (
            generate_ai_response, get_available_topics, get_available_skill_levels, 
            get_available_modes, GeminiClient, user_info_cache
        )
        from .tts_engine import text_to_speech
        from .discord_notify import send_discord_notification
    except ImportError:
        # Last attempt - try from app package
        from app.whisper_utils import transcribe_audio
        from app.ai_engine import (
            generate_ai_response, get_available_topics, get_available_skill_levels, 
            get_available_modes, GeminiClient, user_info_cache
        )
        from app.tts_engine import text_to_speech
        from app.discord_notify import send_discord_notification

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI(
    title="Tech Interviewer AI",
    description="An advanced technical interview assistant",
    version="1.0.0",
)

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Mount static files directory
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Setup templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Create uploads directory if it doesn't exist
UPLOAD_DIR = BASE_DIR / "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Store user information
def set_user_info(conversation_id: str, name: str, email: str) -> None:
    """Store user information for a conversation"""
    if not conversation_id:
        return
        
    if conversation_id not in user_info_cache:
        user_info_cache[conversation_id] = {}
        
    user_info_cache[conversation_id]["name"] = name
    user_info_cache[conversation_id]["email"] = email
    logger.info(f"Stored user info for conversation {conversation_id}: {name}, {email}")
    
def get_user_info(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Get user information for a conversation"""
    if not conversation_id or conversation_id not in user_info_cache:
        return None
        
    return user_info_cache[conversation_id]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Root endpoint that serves the main HTML interface
    """
    logger.info("Accessing root endpoint")
    
    # Get available interview topics and skill levels for the UI
    topics = get_available_topics()
    skill_levels = get_available_skill_levels()
    interview_modes = get_available_modes()  # Changed from get_interview_modes
    
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "topics": topics,
            "skill_levels": skill_levels,
            "interview_modes": interview_modes
        }
    )


@app.post("/talk")
async def talk_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    text_input: str = Form(None),
    conversation_id: str = Form(None),
    interview_topic: str = Form("machine_learning"),
    skill_level: str = Form("intermediate"),
    interview_mode: str = Form("interview"),
    voice_option: str = Form("en-US-AriaNeural"),
    user_name: str = Form(None),
    user_email: str = Form(None),
    end_interview: bool = Form(False)
):
    """
    Main endpoint for processing user input
    - Accepts file uploads (audio/video) or text input
    - Transcribes audio if needed
    - Generates AI response
    - Converts response to speech
    - Sends notification via webhooks
    - Returns response to frontend
    """
    try:
        user_prompt = ""
        source_type = "text"
        
        # Generate a conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            logger.info(f"Generated new conversation ID: {conversation_id}")
        
        # Process file upload if provided
        if file and file.filename:
            logger.info(f"Processing uploaded file: {file.filename}")
            file_path = UPLOAD_DIR / file.filename
            
            # Save uploaded file
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Transcribe audio/video to text
            try:
                user_prompt = await transcribe_audio(file_path)
                source_type = "audio"
                logger.info(f"Transcription result: {user_prompt}")
                
                # Clean up file after transcription
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Transcription error: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Failed to transcribe audio: {str(e)}"},
                )
        
        # Use text input if no file or transcription failed
        elif text_input:
            logger.info(f"Processing text input: {text_input}")
            user_prompt = text_input
        else:
            # For empty first message, just start the interview with introduction
            logger.info("No input provided, starting interview with introduction")
        
        # Store user info if provided
        user_info = None
        if user_name or user_email:
            logger.info(f"User info provided: {user_name}, {user_email}")
            user_info = {"name": user_name, "email": user_email}
            if conversation_id:
                set_user_info(conversation_id, user_name, user_email)
        else:
            # Try to get stored user info
            user_info = get_user_info(conversation_id)
        
        # Generate AI response
        try:
            ai_response = await generate_ai_response(
                prompt=user_prompt,
                conversation_id=conversation_id,
                interview_topic=interview_topic,
                skill_level=skill_level,
                interview_mode=interview_mode,
                user_info=user_info,
                end_interview=end_interview
            )
            logger.info(f"AI response generated: {ai_response[:100]}...")
        except Exception as e:
            logger.error(f"AI response error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to generate AI response: {str(e)}"},
            )
        
        # Generate speech from AI response
        try:
            audio_filename = await text_to_speech(ai_response, voice=voice_option)
            audio_url = f"/static/audio/{audio_filename}"
            logger.info(f"TTS generated: {audio_filename}, URL: {audio_url}")
        except Exception as e:
            logger.error(f"TTS error: {e}")
            audio_url = None
        
        # Send notification in the background
        background_tasks.add_task(
            send_discord_notification,
            user_input=user_prompt,
            ai_response=ai_response,
            source_type=source_type,
            metadata={
                "interview_topic": interview_topic,
                "skill_level": skill_level,
                "interview_mode": interview_mode,
                "conversation_id": conversation_id,
                "user_name": user_info.get("name") if user_info else user_name,
                "user_email": user_info.get("email") if user_info else user_email,
                "is_final_assessment": end_interview
            }
        )
        
        # Return response
        return JSONResponse(
            content={
                "response": ai_response,
                "audio_url": audio_url,
                "input_type": source_type,
                "conversation_id": conversation_id,
                "interview_topic": interview_topic,
                "skill_level": skill_level,
                "interview_mode": interview_mode,
                "is_final_assessment": end_interview
            }
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in talk endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"An unexpected error occurred: {str(e)}"},
        )


@app.get("/topics")
async def get_topics():
    """
    Get available interview topics
    """
    return JSONResponse(content=get_available_topics())


@app.get("/skill-levels")
async def get_skills():
    """
    Get available skill levels
    """
    return JSONResponse(content=get_available_skill_levels())


@app.get("/interview-modes")
async def get_modes():
    """
    Get available interview modes
    """
    return JSONResponse(content=get_available_modes())


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    # Use environment variables for host and port, with fallbacks
    host = os.environ.get("HOST", "0.0.0.0")
    port_env = os.environ.get("PORT", "0")
    
    # Convert port to int, use 0 for automatic port assignment
    port = int(port_env)
    
    # Log message to show the correct URL to access the application
    display_host = "localhost" if host == "0.0.0.0" else host
    if port == 0:
        logger.info(f"Server will be started with automatic port selection")
        logger.info(f"Once running, access the application at: http://{display_host}:<assigned_port>")
        logger.info(f"Server is running...http://localhost:{port}")
    else:
        logger.info(f"Server will be started at: http://{display_host}:{port}")
        logger.info(f"Server is running...http://localhost:{port}")
    
    # Run server with automatic port selection if port=0
    uvicorn.run("main:app", host=host, port=port, log_level="info")
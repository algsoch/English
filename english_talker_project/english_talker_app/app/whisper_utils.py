"""
Whisper Utils Module - For audio/video transcription using OpenAI's Whisper
"""
import logging
import asyncio
import os
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Union, Any, List
import aiofiles
import aiohttp
import ffmpeg

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
WHISPER_ENDPOINT = os.environ.get("WHISPER_API_ENDPOINT", None)
WHISPER_API_KEY = os.environ.get("WHISPER_API_KEY", None)
AUDIO_SAMPLE_RATE = 16000  # Hz (required for Whisper)
MAX_DURATION = 600  # seconds (10 minutes max)

class WhisperTranscriber:
    """Handles transcription using either local Whisper or OpenAI's API"""
    
    def __init__(
        self, 
        model: str = WHISPER_MODEL,
        api_key: Optional[str] = WHISPER_API_KEY,
        api_endpoint: Optional[str] = WHISPER_ENDPOINT
    ):
        self.model = model
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.session = None
        
    async def ensure_session(self):
        """Ensure aiohttp session is created"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
            
    async def convert_to_audio(self, file_path: Union[str, Path]) -> Path:
        """
        Convert video or non-WAV audio to WAV format with required sample rate
        
        Args:
            file_path: Path to the source file
            
        Returns:
            Path to the converted WAV file
        """
        file_path = Path(file_path)
        output_path = file_path.parent / f"{file_path.stem}_converted.wav"
        
        try:
            # Run ffmpeg to convert the file to WAV with proper settings
            process = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-i", str(file_path),
                "-ar", str(AUDIO_SAMPLE_RATE),
                "-ac", "1",  # mono
                "-c:a", "pcm_s16le",  # 16-bit PCM
                "-y",  # overwrite existing file
                str(output_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"FFmpeg error: {stderr.decode()}")
                raise Exception(f"FFmpeg conversion failed: {stderr.decode()}")
                
            logger.info(f"Successfully converted {file_path} to WAV format at {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting file to WAV: {e}")
            raise
    
    async def transcribe_api(self, audio_path: Path) -> str:
        """
        Transcribe audio using OpenAI's Whisper API
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        if not self.api_key:
            raise ValueError("API key is required for Whisper API transcription")
            
        await self.ensure_session()
        
        try:
            # Endpoint to use (official OpenAI API or custom endpoint)
            endpoint = self.api_endpoint or "https://api.openai.com/v1/audio/transcriptions"
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            # Prepare multipart form data
            form_data = aiohttp.FormData()
            form_data.add_field(
                name="file",
                value=open(audio_path, "rb"),
                content_type="audio/wav",
                filename=audio_path.name
            )
            form_data.add_field("model", "whisper-1")
            form_data.add_field("response_format", "json")
            
            # Make the API request
            async with self.session.post(endpoint, headers=headers, data=form_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Whisper API error: {response.status} - {error_text}")
                    raise Exception(f"Whisper API error: {response.status} - {error_text}")
                
                result = await response.json()
                return result.get("text", "")
                
        except Exception as e:
            logger.error(f"Error transcribing with Whisper API: {e}")
            raise
    
    async def transcribe_local(self, audio_path: Path) -> str:
        """
        Transcribe audio using local Whisper installation
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Create temp file for output
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
                output_path = tmp.name
            
            # Run whisper CLI command
            cmd = [
                "whisper",
                str(audio_path),
                "--model", self.model,
                "--output_format", "json",
                "--output_dir", str(Path(output_path).parent),
                "--language", "en"  # Force English language
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Whisper CLI error: {stderr.decode()}")
                raise Exception(f"Whisper CLI error: {stderr.decode()}")
            
            # Load the JSON output
            json_path = Path(output_path).parent / f"{audio_path.stem}.json"
            async with aiofiles.open(json_path, mode='r') as f:
                result = json.loads(await f.read())
            
            # Clean up the temp file
            try:
                os.remove(json_path)
            except:
                pass
                
            return result.get("text", "")
            
        except Exception as e:
            logger.error(f"Error transcribing with local Whisper: {e}")
            
            # Try simpler subprocess call as fallback
            try:
                logger.info("Attempting simpler fallback transcription...")
                result = subprocess.run(
                    ["whisper", str(audio_path), "--model", self.model, "--language", "en"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Extract transcription from stdout
                output_lines = result.stdout.splitlines()
                for line in output_lines:
                    if line.strip() and not line.startswith("["):
                        return line.strip()
                        
                return output_lines[-1] if output_lines else ""
                
            except Exception as fallback_error:
                logger.error(f"Fallback transcription also failed: {fallback_error}")
                raise Exception("All transcription attempts failed") from e

# Initialize transcriber with default settings
transcriber = WhisperTranscriber()

async def transcribe_audio(file_path: Union[str, Path]) -> str:
    """
    Main function to transcribe audio or video file
    
    Args:
        file_path: Path to the audio or video file
        
    Returns:
        Transcribed text
    """
    file_path = Path(file_path)
    
    try:
        # Convert to proper audio format if needed
        if file_path.suffix.lower() not in ['.wav']:
            logger.info(f"Converting {file_path} to WAV format...")
            audio_path = await transcriber.convert_to_audio(file_path)
        else:
            audio_path = file_path
        
        # Try API transcription first if available
        if transcriber.api_key:
            try:
                logger.info("Attempting transcription with Whisper API...")
                return await transcriber.transcribe_api(audio_path)
            except Exception as e:
                logger.warning(f"API transcription failed, falling back to local: {e}")
        
        # Fall back to local transcription
        logger.info("Attempting transcription with local Whisper...")
        return await transcriber.transcribe_local(audio_path)
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise
    finally:
        # Clean up temporary files
        if 'audio_path' in locals() and audio_path != file_path:
            try:
                os.remove(audio_path)
            except:
                pass

# Example usage
if __name__ == "__main__":
    async def test():
        result = await transcribe_audio("test_audio.mp3")
        print(f"Transcription: {result}")
    
    asyncio.run(test())
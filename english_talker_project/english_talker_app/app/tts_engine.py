"""
Text-to-Speech Engine Module - Converts AI responses to spoken audio
"""
import logging
import asyncio
import os
import subprocess
import tempfile
import uuid
import platform
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
# Ensure audio files are saved in the correct path that matches the static mount
AUDIO_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "static" / "audio"
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

# TTS Engine types
class TTSEngine(str, Enum):
    PYTTSX3 = "pyttsx3"  # Offline TTS
    EDGE_TTS = "edge_tts"  # Microsoft Edge TTS (requires internet)
    GTTS = "gtts"  # Google Text-to-Speech (requires internet)
    SYSTEM = "system"  # System's native TTS capabilities

# Default configuration
DEFAULT_ENGINE = os.environ.get("TTS_ENGINE", TTSEngine.EDGE_TTS)
DEFAULT_VOICE = os.environ.get("TTS_VOICE", "en-US-AriaNeural")
DEFAULT_RATE = int(os.environ.get("TTS_RATE", "0"))  # 0 = normal speed
DEFAULT_VOLUME = float(os.environ.get("TTS_VOLUME", "1.0"))  # 0.0 to 1.0

class TTSManager:
    """Manager class for Text-to-Speech conversion using various engines"""
    
    def __init__(
        self,
        engine: Union[TTSEngine, str] = DEFAULT_ENGINE,
        voice: str = DEFAULT_VOICE,
        rate: int = DEFAULT_RATE,
        volume: float = DEFAULT_VOLUME
    ):
        self.engine = engine if isinstance(engine, TTSEngine) else TTSEngine(engine)
        self.voice = voice
        self.rate = rate
        self.volume = volume
        
        # Create output directory if it doesn't exist
        os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
        
    async def text_to_speech(self, text: str) -> Path:
        """
        Convert text to speech using the configured engine
        
        Args:
            text: The text to convert to speech
            
        Returns:
            Path to the generated audio file
        """
        logger.info(f"Converting text to speech using {self.engine} engine")
        
        # Choose the appropriate TTS method based on engine
        if self.engine == TTSEngine.PYTTSX3:
            return await self._tts_pyttsx3(text)
        elif self.engine == TTSEngine.EDGE_TTS:
            return await self._tts_edge(text)
        elif self.engine == TTSEngine.GTTS:
            return await self._tts_google(text)
        elif self.engine == TTSEngine.SYSTEM:
            return await self._tts_system(text)
        else:
            logger.warning(f"Unknown TTS engine {self.engine}, falling back to pyttsx3")
            return await self._tts_pyttsx3(text)
            
    async def _tts_pyttsx3(self, text: str) -> Path:
        """
        Convert text to speech using pyttsx3 (offline TTS engine)
        
        Args:
            text: The text to convert to speech
            
        Returns:
            Path to the generated audio file
        """
        try:
            import pyttsx3
            
            # Generate a unique filename
            output_file = AUDIO_OUTPUT_DIR / f"tts_{uuid.uuid4()}.mp3"
            
            # This needs to run in a separate thread as pyttsx3 is blocking
            def run_tts():
                engine = pyttsx3.init()
                
                # Configure the engine
                engine.setProperty('rate', 150 + (self.rate * 10))  # Adjust rate
                engine.setProperty('volume', self.volume)  # Volume 0.0 to 1.0
                
                # Set voice if specified
                if self.voice:
                    voices = engine.getProperty('voices')
                    for voice in voices:
                        if self.voice.lower() in voice.id.lower():
                            engine.setProperty('voice', voice.id)
                            break
                
                # Save to file
                engine.save_to_file(text, str(output_file))
                engine.runAndWait()
            
            # Run in an executor to avoid blocking
            await asyncio.get_event_loop().run_in_executor(None, run_tts)
            
            logger.info(f"Audio saved to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error using pyttsx3: {e}")
            # Fall back to system TTS if pyttsx3 fails
            return await self._tts_system(text)
    
    async def _tts_edge(self, text: str) -> Path:
        """
        Convert text to speech using Microsoft Edge TTS
        
        Args:
            text: The text to convert to speech
            
        Returns:
            Path to the generated audio file
        """
        try:
            # Generate a unique filename
            output_file = AUDIO_OUTPUT_DIR / f"tts_{uuid.uuid4()}.mp3"
            
            # Adjust rate parameter for edge-tts
            rate_param = f"+{self.rate}%" if self.rate > 0 else f"{self.rate}%"
            
            # Build edge-tts command
            cmd = [
                "edge-tts",
                "--text", text,
                "--voice", self.voice,
                "--rate", rate_param,
                "--output", str(output_file)
            ]
            
            # Run edge-tts
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Edge TTS error: {stderr.decode()}")
                raise Exception(f"Edge TTS failed: {stderr.decode()}")
                
            logger.info(f"Audio saved to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error using Edge TTS: {e}")
            # Fall back to pyttsx3 if Edge TTS fails
            return await self._tts_pyttsx3(text)
    
    async def _tts_google(self, text: str) -> Path:
        """
        Convert text to speech using Google Text-to-Speech
        
        Args:
            text: The text to convert to speech
            
        Returns:
            Path to the generated audio file
        """
        try:
            from gtts import gTTS
            
            # Generate a unique filename
            output_file = AUDIO_OUTPUT_DIR / f"tts_{uuid.uuid4()}.mp3"
            
            # Set language code (extract from voice setting)
            lang = "en"
            if "-" in self.voice:
                lang = self.voice.split("-")[0]
            
            # gTTS is blocking, so run in a separate thread
            def run_gtts():
                tts = gTTS(text=text, lang=lang, slow=self.rate < 0)
                tts.save(str(output_file))
            
            # Run in an executor to avoid blocking
            await asyncio.get_event_loop().run_in_executor(None, run_gtts)
            
            logger.info(f"Audio saved to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error using Google TTS: {e}")
            # Fall back to pyttsx3 if Google TTS fails
            return await self._tts_pyttsx3(text)
    
    async def _tts_system(self, text: str) -> Path:
        """
        Convert text to speech using system's native TTS capabilities
        
        Args:
            text: The text to convert to speech
            
        Returns:
            Path to the generated audio file
        """
        try:
            # Generate a unique filename
            output_file = AUDIO_OUTPUT_DIR / f"tts_{uuid.uuid4()}.mp3"
            
            # Command depends on operating system
            system = platform.system()
            
            if system == "Windows":
                # Create a temporary VBS script
                vbs_file = tempfile.NamedTemporaryFile(suffix='.vbs', delete=False)
                vbs_path = vbs_file.name
                vbs_file.close()
                
                # Write the VBS script to use Windows Speech API
                with open(vbs_path, 'w') as f:
                    f.write(f'Dim speech\n')
                    f.write(f'Set speech = CreateObject("SAPI.SpVoice")\n')
                    
                    # Set voice if specified
                    if self.voice:
                        f.write(f'Dim voices\n')
                        f.write(f'Set voices = speech.GetVoices\n')
                        f.write(f'For Each v In voices\n')
                        f.write(f'    If InStr(v.GetDescription(), "{self.voice}") > 0 Then\n')
                        f.write(f'        Set speech.Voice = v\n')
                        f.write(f'        Exit For\n')
                        f.write(f'    End If\n')
                        f.write(f'Next\n')
                    
                    # Set rate (-10 to 10, default is 0)
                    f.write(f'speech.Rate = {self.rate}\n')
                    
                    # Set volume (0 to 100)
                    volume = int(self.volume * 100)
                    f.write(f'speech.Volume = {volume}\n')
                    
                    # Create file stream for saving
                    f.write(f'Dim filestream\n')
                    f.write(f'Set filestream = CreateObject("SAPI.SpFileStream")\n')
                    f.write(f'filestream.Open "{str(output_file)}", 3, False\n')
                    f.write(f'Set speech.AudioOutputStream = filestream\n')
                    f.write(f'speech.Speak "{text.replace(chr(34), chr(34) + " + chr(34) + " + chr(34))}", 3\n')
                    f.write(f'filestream.Close\n')
                
                # Run the VBS script
                process = await asyncio.create_subprocess_exec(
                    'cscript.exe', vbs_path, '/nologo',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                # Clean up
                try:
                    os.remove(vbs_path)
                except:
                    pass
                
                if process.returncode != 0:
                    logger.error(f"Windows TTS error: {stderr.decode()}")
                    raise Exception(f"Windows TTS failed: {stderr.decode()}")
                    
            elif system == "Darwin":  # macOS
                # Create a temporary AIFF file (macOS say command)
                aiff_file = AUDIO_OUTPUT_DIR / f"tts_temp_{uuid.uuid4()}.aiff"
                
                # Build say command
                cmd = [
                    "say",
                    "-o", str(aiff_file),
                    text
                ]
                
                # Add voice if specified
                if self.voice:
                    cmd.extend(["-v", self.voice])
                
                # Add rate if specified (-10 to 10, default is 0)
                if self.rate != 0:
                    cmd.extend(["-r", str(100 + (self.rate * 10))])
                
                # Run say command
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    logger.error(f"macOS TTS error: {stderr.decode()}")
                    raise Exception(f"macOS TTS failed: {stderr.decode()}")
                
                # Convert AIFF to MP3
                cmd = [
                    "ffmpeg",
                    "-i", str(aiff_file),
                    "-y",
                    str(output_file)
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                # Clean up
                try:
                    os.remove(aiff_file)
                except:
                    pass
                
                if process.returncode != 0:
                    logger.error(f"FFmpeg error: {stderr.decode()}")
                    raise Exception(f"FFmpeg conversion failed: {stderr.decode()}")
                    
            elif system == "Linux":
                # Use espeak on Linux
                # First, create a wav file
                wav_file = AUDIO_OUTPUT_DIR / f"tts_temp_{uuid.uuid4()}.wav"
                
                # Build espeak command
                cmd = [
                    "espeak",
                    "-w", str(wav_file),
                    text
                ]
                
                # Add voice if specified
                if self.voice:
                    cmd.extend(["-v", self.voice])
                
                # Add rate if specified (words per minute, default is 175)
                if self.rate != 0:
                    cmd.extend(["-s", str(175 + (self.rate * 20))])
                
                # Run espeak command
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    logger.error(f"Linux TTS error: {stderr.decode()}")
                    raise Exception(f"Linux TTS failed: {stderr.decode()}")
                
                # Convert WAV to MP3
                cmd = [
                    "ffmpeg",
                    "-i", str(wav_file),
                    "-y",
                    str(output_file)
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                # Clean up
                try:
                    os.remove(wav_file)
                except:
                    pass
                
                if process.returncode != 0:
                    logger.error(f"FFmpeg error: {stderr.decode()}")
                    raise Exception(f"FFmpeg conversion failed: {stderr.decode()}")
            
            logger.info(f"Audio saved to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error using system TTS: {e}")
            # Fall back to pyttsx3 if system TTS fails
            return await self._tts_pyttsx3(text)

# Create default TTS manager instance
tts_manager = TTSManager()

async def text_to_speech(
    text: str,
    engine: Optional[Union[TTSEngine, str]] = None,
    voice: Optional[str] = None,
    rate: Optional[int] = None,
    volume: Optional[float] = None
) -> str:
    """
    Main function to convert text to speech
    
    Args:
        text: The text to convert to speech
        engine: Optional override for TTS engine
        voice: Optional override for voice
        rate: Optional override for speech rate
        volume: Optional override for volume
        
    Returns:
        Path to the generated audio file (relative to static directory)
    """
    # Create a new manager with overrides or use the default
    manager = tts_manager
    if engine or voice or rate is not None or volume is not None:
        manager = TTSManager(
            engine=engine or tts_manager.engine,
            voice=voice or tts_manager.voice,
            rate=rate if rate is not None else tts_manager.rate,
            volume=volume if volume is not None else tts_manager.volume
        )
    
    try:
        # Convert text to speech
        audio_path = await manager.text_to_speech(text)
        
        # Extract just the filename for the URL
        filename = audio_path.name
        logger.info(f"Audio filename for URL: {filename}")
        
        # Return just the filename - the audio_url will be constructed in main.py
        return filename
        
    except Exception as e:
        logger.error(f"Text-to-speech failed: {e}")
        raise

# Example usage
if __name__ == "__main__":
    async def test():
        audio_path = await text_to_speech(
            "Hello, I am the English Talker AI. How can I help you practice your English today?"
        )
        print(f"Audio saved to: {audio_path}")
    
    asyncio.run(test())
"""
Notification Module - Sends conversation logs to Discord and Slack via webhooks
"""
import logging
import os
import json
import asyncio
import aiohttp
from typing import Optional, Dict, Any, Union, List
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
DEFAULT_USERNAME = "Tech Interviewer AI"
DEFAULT_AVATAR_URL = "https://i.imgur.com/XxxxXxx.png"  # Replace with your bot avatar URL
ENABLE_DISCORD = DISCORD_WEBHOOK_URL != ""
ENABLE_SLACK = SLACK_WEBHOOK_URL != ""

class NotificationService:
    """Base class for notification services"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
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
    
    async def send_notification(
        self,
        user_input: str,
        ai_response: str,
        source_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send a notification (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement this method")

class DiscordNotifier(NotificationService):
    """Handles sending notifications to Discord via webhooks"""
    
    def __init__(
        self, 
        webhook_url: str = DISCORD_WEBHOOK_URL,
        username: str = DEFAULT_USERNAME,
        avatar_url: str = DEFAULT_AVATAR_URL
    ):
        super().__init__(webhook_url)
        self.username = username
        self.avatar_url = avatar_url
    
    async def send_webhook(
        self,
        content: str = "",
        embeds: Optional[List[Dict[str, Any]]] = None,
        tts: bool = False
    ) -> bool:
        """
        Send a message to Discord via webhook
        
        Args:
            content: Text content of the message
            embeds: List of Discord embeds
            tts: Whether to use text-to-speech for the message
            
        Returns:
            True if successful, False otherwise
        """
        if not self.webhook_url:
            logger.warning("Discord webhook URL not set, skipping notification")
            return False
            
        try:
            await self.ensure_session()
            
            # Prepare payload
            payload = {
                "username": self.username,
                "avatar_url": self.avatar_url,
                "tts": tts
            }
            
            if content:
                payload["content"] = content
                
            if embeds:
                payload["embeds"] = embeds
            
            # Send webhook
            headers = {"Content-Type": "application/json"}
            async with self.session.post(self.webhook_url, json=payload, headers=headers) as response:
                if response.status not in [200, 204]:
                    logger.error(f"Discord webhook error: {response.status}")
                    return False
                return True
                    
        except Exception as e:
            logger.error(f"Error sending Discord webhook: {e}")
            return False
            
    async def send_notification(
        self,
        user_input: str,
        ai_response: str,
        source_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send an interview conversation log to Discord
        
        Args:
            user_input: User's answer
            ai_response: Interviewer's response
            source_type: Source of user input (text, audio, video)
            metadata: Additional information (e.g., interview topic, skill level)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract metadata
            interview_topic = "Unknown Topic"
            skill_level = "Unknown Level"
            interview_mode = "interview"
            conversation_id = None
            user_name = "Anonymous"
            user_email = "Not provided"
            is_final_assessment = False
            
            if metadata:
                interview_topic = metadata.get("interview_topic", interview_topic)
                skill_level = metadata.get("skill_level", skill_level)
                interview_mode = metadata.get("interview_mode", interview_mode)
                conversation_id = metadata.get("conversation_id")
                user_name = metadata.get("user_name", user_name)
                user_email = metadata.get("user_email", user_email)
                is_final_assessment = metadata.get("is_final_assessment", False)
            
            # Format topic and skill level for display
            if "_" in interview_topic:
                interview_topic = interview_topic.replace("_", " ").title()
                
            if "_" in skill_level:
                skill_level = skill_level.replace("_", " ").title()
                
            if "_" in interview_mode:
                interview_mode = interview_mode.replace("_", " ").title()
            
            # Create timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create embed
            embed = {
                "title": f"Technical {'Assessment' if is_final_assessment else 'Interview'}: {interview_topic}",
                "description": f"Mode: {interview_mode} | Skill Level: {skill_level}",
                "color": 3447003 if not is_final_assessment else 16776960,  # Blue for regular, Yellow for assessment
                "timestamp": datetime.now().isoformat(),
                "footer": {
                    "text": f"Conversation ID: {conversation_id or 'N/A'}"
                },
                "fields": [
                    {
                        "name": "👤 User Information",
                        "value": f"Name: {user_name}\nEmail: {user_email}",
                        "inline": True
                    }
                ]
            }
            
            # Add exchange fields
            if not is_final_assessment:
                embed["fields"].extend([
                    {
                        "name": "🙋 Candidate Response",
                        "value": user_input[:1024] if user_input else "(No response provided)",
                        "inline": False
                    },
                    {
                        "name": "🤖 Interviewer Response",
                        "value": ai_response[:1024] if ai_response else "(No response generated)",
                        "inline": False
                    }
                ])
            else:
                # For final assessment, just show the AI's evaluation
                embed["fields"].extend([
                    {
                        "name": "📊 Final Assessment",
                        "value": ai_response[:1024] if ai_response else "(No assessment generated)",
                        "inline": False
                    }
                ])
            
            # Add source type info
            if source_type != "text":
                embed["fields"].insert(1, {
                    "name": "Response Format",
                    "value": f"{source_type.capitalize()} Recording",
                    "inline": True
                })
            
            # Send the webhook
            content_msg = f"💬 **New interview exchange at {timestamp}**"
            if is_final_assessment:
                content_msg = f"📝 **Final assessment for {user_name} at {timestamp}**"
                
            return await self.send_webhook(
                content=content_msg,
                embeds=[embed]
            )
            
        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")
            return False

class SlackNotifier(NotificationService):
    """Handles sending notifications to Slack via webhooks"""
    
    def __init__(self, webhook_url: str = SLACK_WEBHOOK_URL):
        super().__init__(webhook_url)
    
    async def send_notification(
        self,
        user_input: str,
        ai_response: str,
        source_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send an interview conversation log to Slack
        
        Args:
            user_input: User's answer
            ai_response: Interviewer's response
            source_type: Source of user input (text, audio, video)
            metadata: Additional information (e.g., interview topic, skill level)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.webhook_url:
            logger.warning("Slack webhook URL not set, skipping notification")
            return False
            
        try:
            await self.ensure_session()
            
            # Extract metadata
            interview_topic = "Unknown Topic"
            skill_level = "Unknown Level"
            interview_mode = "interview"
            conversation_id = None
            user_name = "Anonymous"
            user_email = "Not provided"
            is_final_assessment = False
            
            if metadata:
                interview_topic = metadata.get("interview_topic", interview_topic)
                skill_level = metadata.get("skill_level", skill_level)
                interview_mode = metadata.get("interview_mode", interview_mode)
                conversation_id = metadata.get("conversation_id")
                user_name = metadata.get("user_name", user_name)
                user_email = metadata.get("user_email", user_email)
                is_final_assessment = metadata.get("is_final_assessment", False)
            
            # Format topic and skill level for display
            if "_" in interview_topic:
                interview_topic = interview_topic.replace("_", " ").title()
                
            if "_" in skill_level:
                skill_level = skill_level.replace("_", " ").title()
                
            if "_" in interview_mode:
                interview_mode = interview_mode.replace("_", " ").title()
            
            # Create timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Base blocks for both message types
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{'Final Assessment' if is_final_assessment else 'Technical Interview'}: {interview_topic}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Mode:*\n{interview_mode}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Skill Level:*\n{skill_level}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*User Name:*\n{user_name}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*User Email:*\n{user_email}"
                        }
                    ]
                },
                {
                    "type": "divider"
                }
            ]
            
            # Add different content based on whether this is a final assessment
            if is_final_assessment:
                blocks.extend([
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*📊 Final Assessment:*\n{ai_response[:1500] if ai_response else '(No assessment generated)'}"
                        }
                    }
                ])
            else:
                blocks.extend([
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*🙋 Candidate Response:*\n{user_input[:1500] if user_input else '(No response provided)'}"
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*🤖 Interviewer Response:*\n{ai_response[:1500] if ai_response else '(No response generated)'}"
                        }
                    }
                ])
            
            # Add context at the end
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Conversation ID: {conversation_id or 'N/A'} | Recorded at {timestamp}"
                    }
                ]
            })
            
            # Create payload
            payload = {
                "blocks": blocks,
                "text": f"{'Final Assessment' if is_final_assessment else 'New Interview Exchange'}: {interview_topic} ({skill_level})"  # Fallback text
            }
            
            # Send to Slack
            headers = {"Content-Type": "application/json"}
            async with self.session.post(self.webhook_url, json=payload, headers=headers) as response:
                if response.status not in [200, 204]:
                    response_text = await response.text()
                    logger.error(f"Slack webhook error {response.status}: {response_text}")
                    return False
                return True
                
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False

# Create notifier instances
discord_notifier = None
slack_notifier = None

if ENABLE_DISCORD:
    discord_notifier = DiscordNotifier()
    
if ENABLE_SLACK:
    slack_notifier = SlackNotifier()

async def send_discord_notification(
    user_input: str,
    ai_response: str,
    source_type: str = "text",
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Main function to send notifications to Discord and/or Slack
    
    Args:
        user_input: User's answer
        ai_response: Interviewer's response
        source_type: Source of user input (text, audio, video)
        metadata: Additional metadata
        
    Returns:
        True if any notification was sent successfully, False otherwise
    """
    results = []
    
    # Send to Discord if enabled
    if ENABLE_DISCORD and discord_notifier:
        try:
            discord_result = await discord_notifier.send_notification(
                user_input=user_input,
                ai_response=ai_response,
                source_type=source_type,
                metadata=metadata
            )
            results.append(discord_result)
            logger.info(f"Discord notification sent: {discord_result}")
        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")
    
    # Send to Slack if enabled
    if ENABLE_SLACK and slack_notifier:
        try:
            slack_result = await slack_notifier.send_notification(
                user_input=user_input,
                ai_response=ai_response,
                source_type=source_type,
                metadata=metadata
            )
            results.append(slack_result)
            logger.info(f"Slack notification sent: {slack_result}")
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
    
    # Return True if any notification was sent successfully
    return any(results) if results else False

# Example usage
if __name__ == "__main__":
    async def test():
        result = await send_discord_notification(
            user_input="In machine learning, I would approach the overfitting problem by using regularization techniques like L1 or L2 regularization, and cross-validation to tune hyperparameters.",
            ai_response="That's a good answer! Could you elaborate on how you would choose between L1 and L2 regularization for a specific problem?",
            source_type="text",
            metadata={
                "interview_topic": "machine_learning",
                "skill_level": "intermediate",
                "interview_mode": "interview",
                "user_name": "John Doe",
                "user_email": "john.doe@example.com",
                "conversation_id": "abc-123-def-456"
            }
        )
        print(f"Notifications sent: {result}")
    
    asyncio.run(test())
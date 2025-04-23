# English Talker Project

A containerized application for English language practice with AI-powered conversation and speech capabilities.

## Features

- Speech-to-text transcription
- AI-powered conversation
- Text-to-speech for AI responses
- Multiple conversation topics and skill levels
- Discord notifications

## Running with Docker

### Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Setup

1. Clone this repository
2. Configure your environment variables (optional)
   - Create a `.env` file in the project root with your API keys and settings

### Build and Run

```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

The application will be available at http://localhost:3500

## Publishing to Docker Hub

To publish this image to Docker Hub under the username `algsoch`:

```bash
# Login to Docker Hub
docker login --username algsoch

# Build the image
docker-compose build

# Push the image to Docker Hub
docker push algsoch/english-talker:latest
```

Once published, others can use your image with:

```bash
docker pull algsoch/english-talker:latest
docker run -p 3500:3500 algsoch/english-talker:latest
```

## Development

### Environment Variables

The following environment variables can be set in your `.env` file:

- `API_KEY`: Your AI service API key
- `DISCORD_WEBHOOK_URL`: Webhook URL for Discord notifications
- Additional environment variables as needed by your application

## Volumes

The Docker configuration mounts the following directories for persistent data:

- `./english_talker_app/uploads`: For user uploaded audio files
- `./english_talker_app/static/audio`: For generated TTS audio files

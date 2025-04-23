import os

# File: /e:/English/File_strucure.py


def create_file_structure(base_dir):
    """
    Creates the file structure for the English Talker AI project.
    """
    structure = {
        "english_talker_app": {
            "app": [
                "main.py",  # FastAPI app logic
                "whisper_utils.py",  # Audio/video transcription
                "ai_engine.py",  # Mistral prompt + response
                "tts_engine.py",  # Text-to-speech handler
                "discord_notify.py",  # Discord integration
            ],
            "static": [
                "style.css",  # Tailwind-enhanced styles
            ],
            "templates": [
                "index.html",  # Frontend layout
            ],
        },
        "Dockerfile": None,  # Azure deployment config
        "requirements.txt": None,
        "README.md": None,
    }

    def create_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def create_file(path):
        if not os.path.exists(path):
            with open(path, "w") as f:
                pass

    def build_structure(base, struct):
        for key, value in struct.items():
            if isinstance(value, dict):
                dir_path = os.path.join(base, key)
                create_dir(dir_path)
                build_structure(dir_path, value)
            elif isinstance(value, list):
                dir_path = os.path.join(base, key)
                create_dir(dir_path)
                for file_name in value:
                    create_file(os.path.join(dir_path, file_name))
            elif value is None:
                create_file(os.path.join(base, key))

    build_structure(base_dir, structure)
    print(f"File structure created at: {base_dir}")


if __name__ == "__main__":
    base_directory = os.path.abspath("english_talker_project")
    create_file_structure(base_directory)
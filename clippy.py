import os
import sys
import json
import shutil
import subprocess
import tempfile
from urllib.parse import urlparse

import requests
import yt_dlp

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

PLACEHOLDER_CONFIG = {
    "speech_endpoint": "<your Azure Speech endpoint, e.g. https://your-resource.cognitiveservices.azure.com/>",
    "speech_api_key": "<your Azure Speech API key>",
    "foundry_endpoint": "<your Azure AI Foundry endpoint, e.g. https://your-resource.services.ai.azure.com>",
    "foundry_api_key": "<your Azure AI Foundry API key>",
    "foundry_model": "<your model deployment name, e.g. gpt-4o>",
}


def load_config():
    if not os.path.isfile(CONFIG_FILE):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(PLACEHOLDER_CONFIG, f, indent=4)
        print(f"Config file created at: {CONFIG_FILE}")
        print()
        print("Please populate it with the following values:")
        print()
        print("FOR TRANSCRIPTION (SPEECH-TO-TEXT)")
        print("These contain values for a connection to a Foundry resource that can perform transcription using MAI-Transcribe-1.")
        print("  speech_endpoint  - Your Azure Speech endpoint URL")
        print("  speech_api_key   - Your Azure Speech API key")
        print()
        print("FOR CLIP SELECTION")
        print("This contains values for a connection to a particular language model in Foundry that will review the transcript and select what to clip out.")
        print("  foundry_endpoint - Your Azure AI Foundry endpoint URL")
        print("  foundry_api_key  - Your Azure AI Foundry API key")
        print("  foundry_model    - The model deployment name to use")
        sys.exit(1)

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Validate no placeholders remain
    for key, value in config.items():
        if isinstance(value, str) and value.startswith("<") and value.endswith(">"):
            print(f"Config value for '{key}' is still a placeholder.")
            print(f"Please update it in: {CONFIG_FILE}")
            sys.exit(1)

    return config


def get_content_type(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    mapping = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".mp4": "audio/mp4",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }
    return mapping.get(ext, "application/octet-stream")


def is_youtube_url(input_str):
    try:
        parsed = urlparse(input_str)
        if parsed.scheme not in ("http", "https"):
            return False
        host = parsed.hostname or ""
        return "youtube.com" in host.lower() or "youtu.be" in host.lower()
    except Exception:
        return False


def download_youtube_video(url):
    output_directory = os.path.join(tempfile.gettempdir(), "clippy-downloads")
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)
    output_template = os.path.join(output_directory, "%(title)s [%(id)s].%(ext)s")

    downloaded_file = None

    def progress_hook(d):
        nonlocal downloaded_file
        if d["status"] == "finished":
            downloaded_file = d.get("filename")

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "progress_hooks": [progress_hook],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if downloaded_file:
        mp4_path = os.path.splitext(downloaded_file)[0] + ".mp4"
        if os.path.isfile(mp4_path):
            return mp4_path
        if os.path.isfile(downloaded_file):
            return downloaded_file

    # Fallback: find the most recently created mp4 in the output directory
    mp4_files = [
        os.path.join(output_directory, f)
        for f in os.listdir(output_directory)
        if f.endswith(".mp4")
    ]
    if mp4_files:
        return max(mp4_files, key=os.path.getmtime)

    raise RuntimeError("yt-dlp completed, but no MP4 file was found.")


def convert_to_mp3(video_path):
    mp3_path = os.path.splitext(video_path)[0] + ".mp3"
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "libmp3lame",
            "-q:a", "0",
            mp3_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("ffmpeg failed to convert video to MP3:\n" + result.stderr)
    return mp3_path


def transcribe_audio(audio_file_path, config):
    endpoint = config["speech_endpoint"].rstrip("/")
    request_url = f"{endpoint}/speechtotext/transcriptions:transcribe?api-version=2025-10-15"

    definition = {
        "locales": ["en"],
        "enhancedMode": {
            "enabled": True,
            "model": "mai-transcribe-1",
        },
    }

    content_type = get_content_type(audio_file_path)

    with open(audio_file_path, "rb") as audio_file:
        files = {
            "audio": (os.path.basename(audio_file_path), audio_file, content_type),
        }
        data = {
            "definition": json.dumps(definition),
        }
        headers = {
            "Ocp-Apim-Subscription-Key": config["speech_api_key"],
        }

        response = requests.post(request_url, headers=headers, files=files, data=data)

    response.raise_for_status()
    body = response.json()

    # Extract combined text
    text = None
    combined_phrases = body.get("combinedPhrases", [])
    if combined_phrases:
        text = combined_phrases[0].get("text")
    if not text:
        text = body.get("text")

    # Extract phrases with timestamps
    phrases = []
    for phrase in body.get("phrases", []):
        offset_ms = phrase.get("offsetMilliseconds", 0)
        duration_ms = phrase.get("durationMilliseconds", 0)
        phrase_text = phrase.get("text", "")
        phrases.append({
            "start_ms": offset_ms,
            "end_ms": offset_ms + duration_ms,
            "text": phrase_text,
        })

    return text, phrases


def format_timestamp_ms(ms):
    total_seconds = ms // 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def build_timestamped_transcript(phrases):
    lines = []
    for phrase in phrases:
        start = format_timestamp_ms(phrase["start_ms"])
        end = format_timestamp_ms(phrase["end_ms"])
        text = phrase["text"].strip()
        lines.append(f"[{start} - {end}] {text}")
    return "\n".join(lines)


def build_clip_selection_prompt(timestamped_transcript, preferred_clip_count=5, min_clip_seconds=300, max_clip_seconds=600, guidance=""):
    guidance_section = ""
    if guidance:
        guidance_section = f"""
Additional guidance from the user:
{guidance}

Prioritize clips that match this guidance. If the guidance specifies a theme or topic, focus on segments that are most relevant to it.

"""

    return f"""Analyze the timestamped transcript below and choose the most interesting clips for reuse.
{guidance_section}Your goal is to identify the strongest segments that could stand alone as compelling clips.
Look for moments with any of the following:
- emotionally intense exchanges
- surprising revelations
- memorable storytelling
- controversy or conflict
- humor
- strong quotable moments
- clear narrative arcs

Requirements:
- Return ONLY JSON.
- Return a JSON object with a single property named "clips".
- "clips" must be an array.
- Each item in "clips" must be an object with exactly these properties:
  - "title": string
  - "reason": string
  - "start_seconds": integer
  - "end_seconds": integer
- Choose up to {preferred_clip_count} clips.
- Each clip should ideally be between {min_clip_seconds} and {max_clip_seconds} seconds long.
- If the transcript is too short or does not support that many clips, return fewer clips.
- Each clip should get into its core idea immediately — start the clip right when the compelling subject matter begins, not during unrelated preamble or small talk leading up to it.
- End each clip before the conversation trails off into a different topic. Cut it tight — the clip should feel focused and self-contained around a single idea or moment.
- Prefer clips with clean starts and clean endings.
- Avoid overlapping clips unless overlap is truly necessary.
- Base your timing on the transcript timestamps.
- "start_seconds" and "end_seconds" must be measured from the beginning of the media.
- "end_seconds" must be greater than "start_seconds".
- Do not include any prose outside the JSON object.

Timestamped transcript:
{timestamped_transcript}"""


def select_interesting_clips(phrases, config, guidance=""):
    timestamped_transcript = build_timestamped_transcript(phrases)

    endpoint = config["foundry_endpoint"].rstrip("/")
    request_url = f"{endpoint}/openai/responses?api-version=2025-04-01-preview"

    prompt = build_clip_selection_prompt(timestamped_transcript, guidance=guidance)

    payload = {
        "model": config["foundry_model"],
        "input": [
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are an expert clip producer for long-form audio and video content. "
                            "You identify the most compelling standalone segments and return only valid JSON."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    }
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_object",
            }
        },
        "background": False,
    }

    headers = {
        "Content-Type": "application/json",
        "api-key": config["foundry_api_key"],
    }

    response = requests.post(request_url, headers=headers, json=payload, timeout=86400)
    response.raise_for_status()

    body = response.json()

    # Extract text from the first output message's content
    response_text = ""
    for output in body.get("output", []):
        if output.get("type") == "message":
            content = output.get("content", [])
            if content:
                response_text = content[0].get("text", "")
                break

    if not response_text:
        raise RuntimeError("The model did not return any message text.")

    parsed = json.loads(response_text)
    return parsed.get("clips", [])


def format_duration(total_seconds):
    if total_seconds < 0:
        total_seconds = 0
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours >= 1:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def validate_dependencies():
    print("Checking dependencies...")
    print()
    all_ok = True

    # Python libraries
    for module_name, pip_name in [("requests", "requests"), ("yt_dlp", "yt-dlp")]:
        try:
            __import__(module_name)
            print(f"  [OK] Python library '{pip_name}' is installed.")
        except ImportError:
            print(f"  [FAIL] Python library '{pip_name}' is not installed. Run: pip install {pip_name}")
            all_ok = False

    # ffmpeg
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            version_line = result.stdout.splitlines()[0] if result.stdout else "unknown version"
            print(f"  [OK] ffmpeg is installed. ({version_line})")
        else:
            print("  [FAIL] ffmpeg returned an error.")
            all_ok = False
    except FileNotFoundError:
        print("  [FAIL] ffmpeg is not installed or not on PATH.")
        all_ok = False

    # config.json
    if not os.path.isfile(CONFIG_FILE):
        print(f"  [FAIL] config.json not found. It will be created on first run at: {CONFIG_FILE}")
        all_ok = False
    else:
        print(f"  [OK] config.json found.")
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)

            required_keys = ["speech_endpoint", "speech_api_key", "foundry_endpoint", "foundry_api_key", "foundry_model"]
            for key in required_keys:
                value = config.get(key, "")
                if not value or (isinstance(value, str) and value.startswith("<") and value.endswith(">")):
                    print(f"  [FAIL] config.json: '{key}' is missing or still a placeholder.")
                    all_ok = False
                else:
                    print(f"  [OK] config.json: '{key}' is set.")
        except (json.JSONDecodeError, Exception) as e:
            print(f"  [FAIL] config.json could not be parsed: {e}")
            all_ok = False

    print()
    if all_ok:
        print("All dependencies OK.")
    else:
        print("One or more dependencies are missing. Please fix the issues above.")
    print()

    return all_ok


def generate_tailored_content(clip_transcript, title, purpose, config, source_url=None):
    endpoint = config["foundry_endpoint"].rstrip("/")
    request_url = f"{endpoint}/openai/responses?api-version=2025-04-01-preview"

    source_instruction = ""
    if source_url:
        source_instruction = f'\n\nAt the end of the content, include a line like "Clipped from {source_url}" as a natural sign-off or attribution.'

    prompt = f"""I have a video clip titled "{title}" with the following transcript:

{clip_transcript}

The user's goal: {purpose}

Based on the transcript and the user's goal, generate tailored content that helps them accomplish what they described. For example, if they want to post on LinkedIn, write a compelling LinkedIn post. If they want to create a tweet thread, write that. If they want a blog excerpt, write that. Match the format to their stated goal.{source_instruction}

Return ONLY the tailored content, ready to use. No preamble or explanation."""

    payload = {
        "model": config["foundry_model"],
        "input": [
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are a skilled content writer who adapts tone and format to match "
                            "the user's intended platform and audience. Be concise, engaging, and authentic."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    }
                ],
            },
        ],
        "background": False,
    }

    headers = {
        "Content-Type": "application/json",
        "api-key": config["foundry_api_key"],
    }

    response = requests.post(request_url, headers=headers, json=payload, timeout=86400)
    response.raise_for_status()

    body = response.json()

    response_text = ""
    for output in body.get("output", []):
        if output.get("type") == "message":
            content = output.get("content", [])
            if content:
                response_text = content[0].get("text", "")
                break

    return response_text.strip()


def extract_clip(clip_index, clip, source_file_path, phrases, user_input, purpose, config):
    start_seconds = clip.get("start_seconds", 0)
    end_seconds = clip.get("end_seconds", 0)
    title = clip.get("title", "clip")

    # Create output folder named after the selected clip number
    clips_dir = os.path.join(os.getcwd(), "CLIPS")
    os.makedirs(clips_dir, exist_ok=True)
    folder_name = f"CLIP{clip_index + 1}"
    folder_path = os.path.join(clips_dir, folder_name)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

    source_ext = os.path.splitext(source_file_path)[1] or ".mp4"
    output_clip_path = os.path.join(folder_path, f"clip{source_ext}")

    print()
    print(f"Extracting clip {clip_index + 1}: {title}")
    print(f"  From {format_duration(start_seconds)} to {format_duration(end_seconds)}")
    print(f"  Saving to: CLIPS/{folder_name}/")

    duration_secs = end_seconds - start_seconds
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start_seconds),
            "-i", source_file_path,
            "-t", str(duration_secs),
            "-c:v", "libx264",
            "-c:a", "aac",
            output_clip_path,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("ffmpeg failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        print()
        return

    # Write transcript for this clip's time range
    clip_transcript_lines = []
    start_ms = start_seconds * 1000
    end_ms = end_seconds * 1000
    for phrase in phrases:
        if phrase["end_ms"] > start_ms and phrase["start_ms"] < end_ms:
            clip_transcript_lines.append(phrase["text"].strip())

    clip_transcript_text = "\n".join(clip_transcript_lines)

    transcript_path = os.path.join(folder_path, "transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(clip_transcript_text)

    # Write info file with title and reason
    info_content = f"# {title}\n\n{clip.get('reason', '')}\n\nClipped from {user_input}\n"

    # Generate tailored content if the user specified a purpose
    if purpose:
        print(f"  Generating tailored content...")
        source_url = user_input if is_youtube_url(user_input) else None
        try:
            tailored = generate_tailored_content(clip_transcript_text, title, purpose, config, source_url)
            if tailored:
                info_content += f"\n## Tailored Content\n\n{tailored}\n"
        except Exception as e:
            print(f"  Warning: Could not generate tailored content: {e}")

    info_path = os.path.join(folder_path, "info.md")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(info_content)

    print(f"Done! Saved to: {folder_path}")
    print()


def main():
    print("=" * 50)
    print("  CLIPPY")
    print("  https://github.com/TimHanewich/clippy")
    print("  AI-powered video clipping from long-form content.")
    print("=" * 50)
    print()

    if not validate_dependencies():
        return

    config = load_config()

    # Clear previous clips
    clips_dir = os.path.join(os.getcwd(), "CLIPS")
    if os.path.exists(clips_dir):
        shutil.rmtree(clips_dir)

    print("Welcome to Clippy!")
    print("Enter the path to a video/audio file or a YouTube URL:")
    try:
        user_input = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if not user_input:
        print("No file path or URL was provided.", file=sys.stderr)
        return

    # Ask for optional clipping guidance
    print()
    print("Any guidance for the clipper? (e.g. 'focus on AI topics', 'find the funniest moments')")
    print("Or press Enter to skip.")
    try:
        clipping_guidance = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    try:
        source_file_path = user_input

        if is_youtube_url(user_input):
            print()
            print("Detected a YouTube URL. Downloading video as MP4...")
            source_file_path = download_youtube_video(user_input)
            print(f"Video downloaded to: {source_file_path}")

        if not os.path.isfile(source_file_path):
            print(f"File not found: {source_file_path}", file=sys.stderr)
            return

        # If source is a video, extract audio as MP3 for transcription
        ext_lower = os.path.splitext(source_file_path)[1].lower()
        if ext_lower in (".mp4", ".mkv", ".webm", ".avi", ".mov"):
            print()
            print("Converting video to MP3 for transcription...")
            audio_file_path = convert_to_mp3(source_file_path)
            print(f"Audio extracted to: {audio_file_path}")
        else:
            audio_file_path = source_file_path

        print()
        print("Step 1 of 2: Transcribing audio...")

        text, phrases = transcribe_audio(audio_file_path, config)

        print("Step 1 of 2 complete.")
        print()
        print("Transcript:")
        print(text if text else "<no plain text extracted>")
        print()

        print("Step 2 of 2: Selecting interesting clips with Foundry...")

        clips = select_interesting_clips(phrases, config, clipping_guidance)

        print("Step 2 of 2 complete.")
        print()
        print("Selected clips:")

        if not clips:
            print("<no clips returned>")
        else:
            for i, clip in enumerate(clips, start=1):
                print(f"Clip {i}:")
                print(f"Title: {clip.get('title', '')}")
                print(f"Reason: {clip.get('reason', '')}")
                print(f"Start: {clip.get('start_seconds', 0)} seconds")
                print(f"End: {clip.get('end_seconds', 0)} seconds")
                duration = clip.get("end_seconds", 0) - clip.get("start_seconds", 0)
                print(f"Duration: {format_duration(duration)}")
                print()

            while True:
                print(f"Which clip would you like to extract? (1-{len(clips)}, 'all', or 'q' to quit)")
                try:
                    choice = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                if choice.lower() == "q":
                    break

                # Determine which clips to extract
                if choice.lower() == "all":
                    indices_to_extract = list(range(len(clips)))
                else:
                    try:
                        clip_index = int(choice) - 1
                    except ValueError:
                        print("Invalid selection.")
                        print()
                        continue

                    if clip_index < 0 or clip_index >= len(clips):
                        print(f"Please choose a number between 1 and {len(clips)}.")
                        print()
                        continue

                    indices_to_extract = [clip_index]

                # Ask about purpose/tailored content
                print()
                print("What are you clipping this for? (e.g. 'LinkedIn post', 'tweet thread', 'blog excerpt')")
                print("Or press Enter to skip tailored content generation.")
                try:
                    purpose = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                for idx in indices_to_extract:
                    extract_clip(idx, clips[idx], source_file_path, phrases, user_input, purpose, config)

                if len(indices_to_extract) > 1:
                    print("All clips extracted.")
                print()

    except Exception as ex:
        print(file=sys.stderr)
        print("Something went sideways:", file=sys.stderr)
        print(str(ex), file=sys.stderr)


if __name__ == "__main__":
    main()

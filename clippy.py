import os
import sys
import json
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
        print("  speech_endpoint  - Your Azure Speech endpoint URL")
        print("  speech_api_key   - Your Azure Speech API key")
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


def download_youtube_audio_as_mp3(url):
    output_directory = os.path.join(tempfile.gettempdir(), "clippy-downloads")
    os.makedirs(output_directory, exist_ok=True)
    output_template = os.path.join(output_directory, "%(title)s [%(id)s].%(ext)s")

    downloaded_file = None

    def progress_hook(d):
        nonlocal downloaded_file
        if d["status"] == "finished":
            downloaded_file = d.get("filename")

    ydl_opts = {
        "format": "bestaudio/best",
        "extract_audio": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "0",
        }],
        "outtmpl": output_template,
        "progress_hooks": [progress_hook],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # The post-processor changes the extension to .mp3
    if downloaded_file:
        mp3_path = os.path.splitext(downloaded_file)[0] + ".mp3"
        if os.path.isfile(mp3_path):
            return mp3_path

    # Fallback: find the most recently created mp3 in the output directory
    mp3_files = [
        os.path.join(output_directory, f)
        for f in os.listdir(output_directory)
        if f.endswith(".mp3")
    ]
    if mp3_files:
        return max(mp3_files, key=os.path.getmtime)

    raise RuntimeError("yt-dlp completed, but no MP3 file was found.")


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


def build_clip_selection_prompt(timestamped_transcript, preferred_clip_count=5, min_clip_seconds=300, max_clip_seconds=600):
    return f"""Analyze the timestamped transcript below and choose the most interesting clips for reuse.

Your goal is to identify the strongest segments that could stand alone as compelling clips.
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
- Prefer clips with clean starts and clean endings.
- Avoid overlapping clips unless overlap is truly necessary.
- Base your timing on the transcript timestamps.
- "start_seconds" and "end_seconds" must be measured from the beginning of the media.
- "end_seconds" must be greater than "start_seconds".
- Do not include any prose outside the JSON object.

Timestamped transcript:
{timestamped_transcript}"""


def select_interesting_clips(phrases, config):
    timestamped_transcript = build_timestamped_transcript(phrases)

    endpoint = config["foundry_endpoint"].rstrip("/")
    request_url = f"{endpoint}/openai/responses?api-version=2025-04-01-preview"

    prompt = build_clip_selection_prompt(timestamped_transcript)

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


def main():
    config = load_config()

    print("Welcome to Clippy Transcriber.")
    print("Enter the path to an MP3 file or a YouTube URL:")
    try:
        user_input = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if not user_input:
        print("No file path or URL was provided.", file=sys.stderr)
        return

    try:
        audio_file_path = user_input

        if is_youtube_url(user_input):
            print()
            print("Detected a YouTube URL. Downloading audio as MP3...")
            audio_file_path = download_youtube_audio_as_mp3(user_input)
            print(f"Audio downloaded to: {audio_file_path}")

        if not os.path.isfile(audio_file_path):
            print(f"Audio file not found: {audio_file_path}", file=sys.stderr)
            return

        print()
        print("Step 1 of 2: Transcribing audio...")

        text, phrases = transcribe_audio(audio_file_path, config)

        print("Step 1 of 2 complete.")
        print()
        print("Transcript:")
        print(text if text else "<no plain text extracted>")
        print()

        print("Step 2 of 2: Selecting interesting clips with Foundry...")

        clips = select_interesting_clips(phrases, config)

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

            # Prompt user to pick a clip
            print(f"Which clip would you like to extract? (1-{len(clips)}, or 'q' to quit)")
            try:
                choice = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return

            if choice.lower() == "q":
                return

            try:
                clip_index = int(choice) - 1
            except ValueError:
                print("Invalid selection.", file=sys.stderr)
                return

            if clip_index < 0 or clip_index >= len(clips):
                print(f"Please choose a number between 1 and {len(clips)}.", file=sys.stderr)
                return

            selected_clip = clips[clip_index]
            start_seconds = selected_clip.get("start_seconds", 0)
            end_seconds = selected_clip.get("end_seconds", 0)
            title = selected_clip.get("title", "clip")

            # Sanitize title for filename
            safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title).strip()
            if not safe_title:
                safe_title = f"clip_{clip_index + 1}"

            ext = os.path.splitext(audio_file_path)[1] or ".mp3"
            output_filename = f"{safe_title}{ext}"
            output_path = os.path.join(os.getcwd(), output_filename)

            print()
            print(f"Extracting clip: {title}")
            print(f"  From {format_duration(start_seconds)} to {format_duration(end_seconds)}")
            print(f"  Saving to: {output_filename}")

            duration_secs = end_seconds - start_seconds
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", audio_file_path,
                    "-ss", str(start_seconds),
                    "-t", str(duration_secs),
                    "-c", "copy",
                    output_path,
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print("ffmpeg failed:", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
                return

            print()
            print(f"Done! Clip saved to: {output_path}")

    except Exception as ex:
        print(file=sys.stderr)
        print("Something went sideways:", file=sys.stderr)
        print(str(ex), file=sys.stderr)


if __name__ == "__main__":
    main()

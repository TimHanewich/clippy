# Clippy

AI-powered video clipping from long-form content. Give it a YouTube URL or local video file, and Clippy will transcribe it, identify the most compelling segments, and extract them as standalone clips, ready to post.

## What It Does

1. **Downloads** the video (YouTube URLs are downloaded as MP4 automatically)
2. **Transcribes** the audio using Azure AI Speech
3. **Identifies clips** by sending the timestamped transcript to an AI model that selects the most interesting, self-contained segments
4. **Extracts** your chosen clips with ffmpeg into individual folders
5. **Generates tailored content** (optional), tell it you're posting to LinkedIn, X, a blog, etc. and it drafts a post for you based on the clip's transcript

## Output

Each extracted clip is saved to a `CLIPS/` folder:

```
CLIPS/
├── CLIP1/
│   ├── clip.mp4          # The extracted video segment
│   ├── transcript.txt    # Transcript for this clip's time range
│   └── info.md           # Title, reason, source, and tailored content
├── CLIP2/
│   ├── clip.mp4
│   ├── transcript.txt
│   └── info.md
└── ...
```

## Requirements

- **Python 3.10+**
- **ffmpeg** installed and on PATH
- **Python packages**: `requests`, `yt-dlp`
- **Azure AI Speech** endpoint and API key
- **Azure AI Foundry** endpoint, API key, and model deployment

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/TimHanewich/clippy.git
   cd clippy
   ```

2. Install dependencies:
   ```bash
   pip install requests yt-dlp
   ```

3. Run it once to generate `config.json`:
   ```bash
   python clippy.py
   ```

4. Populate `config.json` with your Azure credentials:
   ```json
   {
       "speech_endpoint": "https://your-resource.cognitiveservices.azure.com/",
       "speech_api_key": "your-speech-api-key",
       "foundry_endpoint": "https://your-resource.services.ai.azure.com",
       "foundry_api_key": "your-foundry-api-key",
       "foundry_model": "gpt-4o"
   }
   ```

5. Run again:
   ```bash
   python clippy.py
   ```

## Usage

```
python clippy.py
```

The interactive prompts will guide you through:

1. **Source**, Paste a YouTube URL or a local file path
2. **Clipping guidance** (optional), Steer the AI toward a theme (e.g. "focus on the AI discussion", "find the funniest moments")
3. **Clip selection**, Pick individual clips by number, type `all` to extract everything, or `q` to quit
4. **Tailored content** (optional), Describe what you're creating (e.g. "LinkedIn post") and the AI drafts it for you

## How It Works

- **Transcription**: Audio is sent to the [Azure AI Speech](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/) batch transcription API, which returns phrase-level timestamps.
- **Clip Selection**: The timestamped transcript is sent to an AI model via the [Azure AI Foundry](https://azure.microsoft.com/en-us/products/ai-studio) Responses API. The model identifies compelling standalone segments based on narrative arcs, emotional intensity, humor, controversy, and other signals.
- **Extraction**: ffmpeg re-encodes the selected time range for frame-accurate cuts (no black frames at the start).
- **Tailored Content**: A second AI call takes the clip transcript and your stated goal to generate platform-ready content.
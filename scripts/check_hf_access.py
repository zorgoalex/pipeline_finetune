#!/usr/bin/env python3
"""Check Hugging Face token and model access for pyannote diarization."""
import os
import sys

def main():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: No HF_TOKEN or HUGGING_FACE_HUB_TOKEN environment variable set.")
        print("  1. Create a token at https://huggingface.co/settings/tokens")
        print("  2. Accept model conditions at https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("  3. Export: export HF_TOKEN=hf_your_token_here")
        sys.exit(1)
    print(f"HF token found: {token[:8]}...{token[-4:]}")
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        print("pyannote diarization pipeline loaded successfully!")
    except ImportError:
        print("WARNING: pyannote.audio not installed yet. Will be available after Phase 3 setup.")
    except Exception as e:
        print(f"ERROR loading pyannote pipeline: {e}")
        print("  Make sure you have accepted the model conditions on Hugging Face.")
        sys.exit(1)

if __name__ == "__main__":
    main()

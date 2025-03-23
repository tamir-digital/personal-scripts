import tensorflow_hub as hub
import tensorflow as tf  # Add to top with other imports
import csv
import urllib
import subprocess
import re
import json
import os
from pathlib import Path
from lightning_whisper_mlx import LightningWhisperMLX
import numpy as np
try:
    import webrtcvad
    import soundfile as sf
except ImportError:
    print("Missing dependencies: pip install webrtcvad soundfile")
    exit(1)


# Add YAMNet initialization (cache the model)
def load_yamnet():
    # Download YAMNet class map
    class_map_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
    class_map_path = tf.keras.utils.get_file(
        'yamnet_class_map.csv', class_map_url, cache_subdir='yamnet'
    )    
    # Load class names
    class_names = {}
    with open(class_map_path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            class_names[int(row[0])] = row[2]
    
    # Load model from TF Hub
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    return model, class_names

# Load YAMNet once at startup
YAMNET_MODEL, YAMNET_CLASS_NAMES = load_yamnet()

def is_speech_yamnet(audio_path, speech_threshold=0.5):
    """Classify audio using YAMNet and check for speech dominance"""
    try:
        # Load audio (already in 16kHz mono from extract_audio_segment)
        audio, sample_rate = sf.read(audio_path)
        
        # Check if audio is too quiet or empty
        rms = np.sqrt(np.mean(np.square(audio)))
        if rms < 0.001:  # Very low audio level
            print(f"Audio in {audio_path} is too quiet (RMS: {rms:.6f}). Amplifying for YAMNet.")
            # Normalize audio to reasonable level if it's too quiet
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.7
        
        # Ensure audio is in float32 format in [-1, 1) range
        audio = audio.astype(np.float32)
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / max(abs(audio.max()), abs(audio.min()))
        
        # Print audio stats for debugging
        print(f"Audio stats: min={audio.min():.4f}, max={audio.max():.4f}, rms={rms:.4f}")
        
        # Run inference
        scores, embeddings, spectrogram = YAMNET_MODEL(audio)
        mean_scores = np.mean(scores, axis=0)

        # Get top 5 classes for debugging
        top_indices = np.argsort(mean_scores)[-5:][::-1]
        print(f"YAMNet top classes for {audio_path}:")
        for idx in top_indices:
            print(f"  {YAMNET_CLASS_NAMES[idx]}: {mean_scores[idx]:.4f}")

        # Find speech class score
        speech_idx = [i for i, name in YAMNET_CLASS_NAMES.items() if name == 'Speech'][0]
        speech_score = mean_scores[speech_idx]
        print(f"  Speech score: {speech_score:.4f} (threshold: {speech_threshold})")

        # Get the highest scoring class
        top_class_idx = np.argmax(mean_scores)
        is_top_class = (top_class_idx == speech_idx)
        if is_top_class:
            print(f"  Speech is the most likely class")

        # Return True if speech score is above threshold OR speech is the most likely class
        return speech_score >= speech_threshold or is_top_class

    except Exception as e:
        print(f"YAMNet failed: {str(e)}")
        return False

def is_speech(audio_path, aggressiveness=3):
    """Robust VAD check with proper audio validation"""
    vad = webrtcvad.Vad(aggressiveness)
    
    try:
        # Read audio with soundfile (handles more formats)
        audio, sample_rate = sf.read(audio_path)
        
        # Validate audio format
        if sample_rate not in [8000, 16000, 32000, 48000]:
            print(f"Unsupported sample rate: {sample_rate}")
            return False
            
        # Convert to 16-bit PCM if needed
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)
            
        # Ensure mono audio
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        frame_duration = 10  # ms
        frame_size = int(sample_rate * frame_duration / 1000)
        frames = np.split(audio, range(frame_size, len(audio), frame_size))
        
        # Check valid frames
        speech_frames = 0
        for frame in frames:
            if len(frame) < frame_size:
                continue
            frame_bytes = frame.tobytes()
            if vad.is_speech(frame_bytes, sample_rate):
                speech_frames += 1
                
        # Require at least 25% speech frames to consider valid
        return (speech_frames / len(frames)) > 0.25 if frames else False
        
    except Exception as e:
        print(f"VAD failed for {audio_path}: {str(e)}")
        return False

def detect_non_silence(input_file, silence_thresh=-30, silence_duration=0.5):
    """Detect non-silent intervals using ffmpeg's silencedetect filter"""
    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-af", f"silencedetect=n={silence_thresh}dB:d={silence_duration}",
        "-f", "null",
        "-",
    ]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    return parse_silence_output(result.stderr)

def parse_silence_output(output):
    """Parse ffmpeg output to get non-silent segments"""
    silences = []
    pattern = r"silence_start: (\d+\.\d+).*silence_end: (\d+\.\d+)"
    matches = re.findall(pattern, output, re.DOTALL)
    
    # Convert matches to float tuples
    silences = [(float(start), float(end)) for start, end in matches]
    
    # Calculate non-silent intervals
    segments = []
    prev_end = 0.0
    
    # Handle before first silence
    if silences and silences[0][0] > 0:
        segments.append((0.0, silences[0][0]))
    
    # Handle between silences
    for i in range(len(silences)-1):
        segments.append((silences[i][1], silences[i+1][0]))
    
    # Handle after last silence
    if silences:
        last_silence_end = silences[-1][1]
        segments.append((last_silence_end, float('inf')))
    
    # If no silences found, entire audio is non-silent
    if not silences:
        segments.append((0.0, float('inf')))

    # In parse_silence_output, after creating segments:
    min_duration = 0.5  # Minimum segment length in seconds
    segments = [(s, e) for s, e in segments if (e - s) >= min_duration]

    return segments

def extract_audio_segment(input_file, start, end, output_dir):
    """Extract 16kHz WAV segment using ffmpeg"""
    safe_start = max(0.0, start)
    safe_end = max(safe_start + 0.5, end)  # Ensure minimum 0.5s duration
    duration = end - safe_start
    
    output_path = os.path.join(output_dir, f"segment_{safe_start:.2f}-{end:.2f}.wav")
    
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(safe_start),
        "-t", str(safe_end),
        "-i", input_file,
        "-af", "highpass=f=300,lowpass=f=3400,dynaudnorm=f=150:g=15",  # Added dynaudnorm
        "-ar", "16000",        # Sample rate
        "-ac", "1",            # Mono channel
        "-acodec", "pcm_s16le",# 16-bit PCM
        "-fflags", "+genpts",  # Fix timestamp issues
        "-loglevel", "error",  # Reduce ffmpeg noise
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error extracting segment {safe_start}-{safe_end}: {e.stderr.decode()}")
        return None

def format_srt_timestamp(seconds):
    """Convert seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")

def save_srt(segments, output_path):
    """Save transcribed segments in SRT format"""
    with open(output_path, "w") as f:
        for i, segment in enumerate(segments, 1):
            start = format_srt_timestamp(segment["start"])
            end = format_srt_timestamp(segment["end"])
            f.write(f"{i}\n{start} --> {end}\n{segment['text'].strip()}\n\n")

def process_video(input_path, model_name="distil-medium.en"):
    """Main processing function"""
    # Set up paths
    input_dir = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_srt = os.path.join(input_dir, f"{base_name}.srt")
    
    # Initialize Whisper
    whisper = LightningWhisperMLX(
        model=model_name,
        batch_size=12,
        quant=None
    )
    
    """Main processing function"""
    # Get total duration using ffprobe
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
         "-of", "default=noprint_wrappers=1:nokey=1", input_path],
        stdout=subprocess.PIPE, text=True
    )
    total_duration = float(probe.stdout.strip())
    
    # Detect non-silent segments and clamp end times
    segments = detect_non_silence(input_path)
    segments = [(start, min(end, total_duration)) for (start, end) in segments]
    
    all_transcriptions = []
    
    # Process each segment
    for idx, (start, end) in enumerate(segments):
        if end == float('inf'):
            end = float(probe.stdout.strip())
        
        print(f"Processing segment {idx+1}: {start:.2f}-{end:.2f}")
        
        # Extract audio segment
        wav_path = extract_audio_segment(input_path, start, end, input_dir)
        if not wav_path or not os.path.exists(wav_path):
            continue

# Add audio validation
#       try:
#           # Normalize audio file headers
#           subprocess.run(["sox", wav_path, wav_path], check=True)
#       except:
#           pass

             # Add VAD check here
        if not is_speech(wav_path):
            print(f"Skipping non-speech segment: {wav_path}")
            try:
                os.remove(wav_path)
            except:
                pass
            continue   
                # Add YAMNet check after VAD
        if not is_speech_yamnet(wav_path):
            print(f"Skipping non-speech (YAMNet filtered): {wav_path}")
            try:
                os.remove(wav_path)
            except:
                pass
            continue

        try:
            result = whisper.transcribe(audio_path=wav_path)
            
            # Create single segment using our detected times
            adjusted_segment = {
                "start": start,
                "end": end,
                "text": result['text'].strip()
            }
            all_transcriptions.append(adjusted_segment)
            
        except Exception as e:
            print(f"Transcription failed for {wav_path}: {str(e)}")
        
        # Clean up temporary WAV file
        try:
            os.remove(wav_path)
        except:
            pass
    
    # Sort segments by start time and save SRT
    all_transcriptions.sort(key=lambda x: x["start"])
    print(f"Saving transcript to {output_srt}")
    save_srt(all_transcriptions, output_srt)
    return output_srt

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcribe non-silent parts of a video")
    parser.add_argument("input_path", help="Path to input video file")
    parser.add_argument("--model", default="distil-medium.en", 
                      help="Whisper model to use (default: distil-medium.en)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input file {args.input_path} not found")
        exit(1)
    
    result_path = process_video(args.input_path, args.model)
    print(f"Transcription complete: {result_path}")

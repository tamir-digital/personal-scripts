import tensorflow_hub as hub
import tensorflow as tf  # Add to top with other imports
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import csv
import subprocess
import re
import os
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

def clean_text(text):
    """Normalize transcription text"""
    # Remove leading/trailing whitespace
    text = text.strip()
    
    if not text:
        return ""
        
    # Remove repeated phrases (more than 2 repetitions)
    text = re.sub(r'\b(\w+(?:\s+\w+){0,3})(?:\s+\1){2,}', r'\1', text)
    
    # Fix stutters (single word repetition)
    text = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', text)
    
    # Remove extra commas and spaces
    text = re.sub(r',,+', ',', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common filler words at beginning
    text = re.sub(r'^(um|uh|so|like|well|you know)\s+', '', text, flags=re.IGNORECASE)
    
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    
    return text


def process_whisper_result(wav_path, pipe, start):
    """Process audio with Whisper and handle result properly"""
    try:
        # Transcribe with language and word timestamps
        result = pipe(
            wav_path, return_timestamps="word") 

        # Extract segments from result
        segments = []
        
               # Debug the structure
        print("Result structure:", result.keys())

        # Handle different result structures
        if 'chunks' in result:
            whisper_segments = result['chunks']
        elif 'output' in result and 'segments' in result['output']:
            whisper_segments = result['output']['segments']
        else:
            whisper_segments = [result]  # Fallback
            
        # Process each segment
        for seg in whisper_segments:
            print(seg)
            # Validate timestamp format
            seg_start = float(seg.get('start', 0))
            seg_end = float(seg.get('end', seg_start + 1))

            # Apply time offset from original segment
            adjusted_start = start + seg_start
            adjusted_end = start + seg_end
            
            # Clean text
            text = clean_text(seg.get('text', ''))
            
            print (f"{adjusted_start} - {adjusted_end}: {text}")
            if text:
                segments.append({
                    "start": adjusted_start,
                    "end": adjusted_end,
                    "text": text
                })
        return segments
                
    except Exception as e:
        print(f"Transcription failed: {str(e)}")
        if "word_timestamp" in str(e):
            print("Make sure you installed the PR that supports word timestamps")
        return []

def detect_non_silence(input_file, silence_thresh=-35, silence_duration=1.0):
    """Detect non-silent intervals using ffmpeg's silencedetect filter"""
    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-af", f"silencedetect=n={silence_thresh}dB:d={silence_duration}",
        "-f", "null",
        "-",
    ]
    
    try:
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, check=True)
        return parse_silence_output(result.stderr, input_file)
    except subprocess.CalledProcessError as e:
        print(f"Error detecting silence: {e}")
        # Return whole file as one segment
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
             "-of", "default=noprint_wrappers=1:nokey=1", input_file],
            stdout=subprocess.PIPE, text=True
        )
        duration = float(probe.stdout.strip())
        return [(0.0, duration)]

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

# Load YAMNet once at startup
YAMNET_MODEL, YAMNET_CLASS_NAMES = load_yamnet()

def is_speech_yamnet(audio_path, speech_threshold=0.4):
    """Classify audio using YAMNet and check for speech dominance"""
    try:
        # Load audio (already in 16kHz mono from extract_audio_segment)
        audio, sample_rate = sf.read(audio_path)
        
        # Ensure we have enough audio
        if len(audio) < sample_rate * 0.5:  # Less than 0.5 seconds
            print(f"Audio in {audio_path} is too short for YAMNet analysis")
            return False
            
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
        print(f"YAMNet analysis failed: {str(e)}")
        # Default to True in case of YAMNet failure to avoid skipping segments
        return True

def parse_silence_output(output, input_file):
    """Parse ffmpeg output to get non-silent segments"""
    silences = []
    pattern = r"silence_start: (\d+\.\d+).*?silence_end: (\d+\.\d+)"
    matches = re.findall(pattern, output, re.DOTALL)
    
    # Convert matches to float tuples
    silences = [(float(start), float(end)) for start, end in matches]
    
    # Get total duration for proper end handling
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
             "-of", "default=noprint_wrappers=1:nokey=1", input_file],
            stdout=subprocess.PIPE, text=True
        )
        total_duration = float(probe.stdout.strip())
    except:
        total_duration = 3600  # Default to 1 hour if duration check fails
    
    # Calculate non-silent intervals
    segments = []
    
    # If no silences found, return whole file
    if not silences:
        segments.append((0.0, total_duration))
        return segments
        
    # Handle first segment (from start to first silence)
    if silences[0][0] > 0:
        segments.append((0.0, silences[0][0]))
    
    # Handle segments between silences
    for i in range(len(silences)-1):
        segments.append((silences[i][1], silences[i+1][0]))
    
    # Handle last segment (from last silence to end)
    segments.append((silences[-1][1], total_duration))
    
    # Filter out very short segments
    min_duration = 0.5  # Minimum segment length in seconds
    segments = [(s, e) for s, e in segments if (e - s) >= min_duration]
    
    return segments

def extract_audio_segment(input_file, start, end, output_dir):
    """Extract 16kHz WAV segment using ffmpeg with improved handling"""
    safe_start = max(0.0, start)
    
    # Calculate actual duration
    duration = end - safe_start
    if duration <= 0:
        print(f"Invalid segment duration: {duration}s")
        return None
        
    # Ensure minimum duration
    safe_end = max(safe_start + 0.5, end)
    
    output_path = os.path.join(output_dir, f"segment_{safe_start:.2f}-{safe_end:.2f}.wav")
    
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(safe_start),
        "-t", str(duration),  # Use duration instead of end time
        "-i", input_file,
        "-af", "highpass=f=200,lowpass=f=3800,dynaudnorm=f=150:g=15",  # Improved audio filter
        "-ar", "16000",        # Sample rate
        "-ac", "1",            # Mono channel
        "-acodec", "pcm_s16le",# 16-bit PCM
        "-fflags", "+genpts",  # Fix timestamp issues
        "-loglevel", "error",  # Reduce ffmpeg noise
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        
        # Verify file exists and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return output_path
        else:
            print(f"Extracted file is too small or doesn't exist: {output_path}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error extracting segment {safe_start}-{safe_end}: {e.stderr.decode() if e.stderr else str(e)}")
        return None

def format_srt_timestamp(seconds):
    """Convert seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def save_srt(segments, output_path):
    """Save transcribed segments in SRT format with error handling"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, 1):
                start = format_srt_timestamp(segment["start"])
                end = format_srt_timestamp(segment["end"])
                
                # Ensure we don't have blank lines
                text = segment["text"].strip()
                if not text:
                    text = "[inaudible]"
                    
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
        
        print(f"Successfully saved SRT file with {len(segments)} segments")
        return True
    except Exception as e:
        print(f"Error saving SRT file: {str(e)}")
        return False

def clean_segment_timing(segments, min_duration=0.5, min_gap=0.1):
    """Ensure segments have proper timing and don't overlap"""
    if not segments:
        return []
        
    # Sort by start time
    sorted_segs = sorted(segments, key=lambda x: x["start"])
    
    # Ensure minimum duration and fix overlaps
    for i in range(len(sorted_segs)):
        # Ensure minimum duration
        if sorted_segs[i]["end"] - sorted_segs[i]["start"] < min_duration:
            sorted_segs[i]["end"] = sorted_segs[i]["start"] + min_duration
        
        # Fix overlaps with next segment
        if i < len(sorted_segs) - 1:
            if sorted_segs[i]["end"] + min_gap > sorted_segs[i+1]["start"]:
                # If segments are very close or overlapping
                if sorted_segs[i]["end"] > sorted_segs[i+1]["start"]:
                    # Actual overlap - adjust endpoint of current segment
                    sorted_segs[i]["end"] = sorted_segs[i+1]["start"] - min_gap
                    
                    # Ensure we didn't make segment too short
                    if sorted_segs[i]["end"] - sorted_segs[i]["start"] < min_duration:
                        # If fixing overlap would make segment too short, adjust start times
                        mid_point = (sorted_segs[i]["start"] + sorted_segs[i+1]["start"]) / 2
                        sorted_segs[i]["end"] = mid_point - min_gap/2
                        sorted_segs[i+1]["start"] = mid_point + min_gap/2
    
    # Final verification - remove any invalid segments
    return [s for s in sorted_segs if s["end"] > s["start"]]

def merge_overlapping_segments(segments, max_gap=0.1, activated=False):
    """Merge segments that are close together in time"""

    if not activated:
        return segments

    if not segments:
        return []
        
    sorted_segments = sorted(segments, key=lambda x: x["start"])
    result = [sorted_segments[0]]
    
    for current in sorted_segments[1:]:
        previous = result[-1]
        
        # If segments are close in time, merge them
        if current["start"] - previous["end"] <= max_gap:
            # Merge with previous segment
            result[-1] = {
                "start": previous["start"],
                "end": max(previous["end"], current["end"]),
                "text": f"{previous['text']} {current['text']}"
            }
        else:
            # Add as new segment
            result.append(current)
    
    return result

def optimize_segments(segments):
    """Merge short segments and split long ones"""
    optimized = []
    buffer = ""
    current_start = None
    current_end = None
    
    for seg in sorted(segments, key=lambda x: x["start"]):
        if not buffer:
            current_start = seg["start"]
            
        buffer += " " + seg["text"].strip()
        current_end = seg["end"]
        
        # Split conditions
        duration = current_end - current_start
        if duration > 5 or any(punc in buffer for punc in ['.', '?', '!']):
            # Find last natural break
            split_points = [buffer.rfind('.'), buffer.rfind('?'), buffer.rfind('!')]
            last_break = max(split_points)
            
            if last_break > 0 and duration > 3:
                end_time = current_start + (duration * (last_break/len(buffer)))
                optimized.append({
                    "start": current_start,
                    "end": end_time,
                    "text": buffer[:last_break+1].strip()
                })
                buffer = buffer[last_break+1:].strip()
                current_start = end_time
            elif duration > 10:  # Force split for very long segments
                mid_point = len(buffer) // 2
                # Find space near midpoint
                space_pos = buffer.find(' ', mid_point)
                if space_pos > 0:
                    mid_time = current_start + (duration * (space_pos/len(buffer)))
                    optimized.append({
                        "start": current_start,
                        "end": mid_time,
                        "text": buffer[:space_pos].strip()
                    })
                    buffer = buffer[space_pos:].strip()
                    current_start = mid_time
    
    # Add remaining buffer
    if buffer:
        optimized.append({
            "start": current_start,
            "end": current_end,
            "text": buffer.strip()
        })
    
    # Remove empty segments
    return [s for s in optimized if s["text"]]

def process_video(input_path, model_name="distil-large-v3"):
    """Main processing function with fixes for accurate transcription"""
    print(f"Starting processing of {input_path} with model {model_name}")
    
    # Set up paths
    input_dir = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_srt = os.path.join(input_dir, f"{base_name}.srt")
    temp_dir = os.path.join(input_dir, f"temp_{base_name}")
    
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize Whisper
    model = AutoModelForSpeechSeq2Seq.from_pretrained("ivrit-ai/whisper-large-v3-turbo")
    processor = AutoProcessor.from_pretrained("ivrit-ai/whisper-large-v3-turbo")
    # Create the ASR pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device="mps:0" if torch.mps.is_available() else "cpu",  # Use GPU if available
    )


    # Get total duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
         "-of", "default=noprint_wrappers=1:nokey=1", input_path],
        stdout=subprocess.PIPE, text=True
    )
    total_duration = float(probe.stdout.strip())
    print(f"Total duration of input file: {total_duration:.2f} seconds")
    
    # Detect non-silent segments
    print("Detecting non-silent segments...")
    segments = detect_non_silence(input_path, silence_thresh=-35, silence_duration=1.0)
    segments = [(start, min(end, total_duration)) for (start, end) in segments]
    print(f"Detected {len(segments)} non-silent segments")
    
    # Process each segment
    all_transcriptions = []
    
    for idx, (start, end) in enumerate(segments):
        if end <= start:
            continue
            
        if end == float('inf'):
            end = total_duration
        
        print(f"Processing segment {idx+1}/{len(segments)}: {start:.2f}-{end:.2f}")
        
        # Extract audio segment
        wav_path = extract_audio_segment(input_path, start, end, temp_dir)
        if not wav_path or not os.path.exists(wav_path):
            print(f"Failed to extract segment {idx+1}")
            continue

        # Speech validation
        if not is_speech(wav_path) or not is_speech_yamnet(wav_path):
            print(f"Skipping non-speech segment: {wav_path}")
            try:
                os.remove(wav_path)
            except:
                pass
            continue

        # Process with Whisper
        segment_transcriptions = process_whisper_result(wav_path, pipe, start)
        all_transcriptions.extend(segment_transcriptions)
        
        # Clean up temporary WAV file
        try:
            os.remove(wav_path)
        except:
            pass
    
    postprocess = False
    if postprocess:

        # Post-process segments
        print(f"Post-processing {len(all_transcriptions)} transcribed segments")
        merged_segments = merge_overlapping_segments(all_transcriptions)
        print(f"After merging: {len(merged_segments)} segments")
        
        optimized_segments = optimize_segments(merged_segments)
        print(f"After optimization: {len(optimized_segments)} segments")
        
        clean_segments = clean_segment_timing(optimized_segments)
        print(f"Final segment count: {len(clean_segments)}")
        
        # Sort and save
        clean_segments.sort(key=lambda x: x["start"])

        transcript = clean_segments
    else:
        transcript = all_transcriptions
    print(f"Saving transcript to {output_srt}")
    save_srt(transcript, output_srt)
    
    # Clean up temp directory if it's empty
    try:
        os.rmdir(temp_dir)
    except:
        pass
        
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

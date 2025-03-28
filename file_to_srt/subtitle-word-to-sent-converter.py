import re
import os
import argparse

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

def format_srt_timestamp(seconds):
    """Convert seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")

def parse_srt_timestamp(timestamp):
    """Convert SRT timestamp to seconds"""
    timestamp = timestamp.replace(',', '.')
    h, m, s = timestamp.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def read_srt_file(file_path):
    """Read SRT file and extract data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by subtitle entries
    entries = re.split(r'\n\s*\n', content.strip())
    
    words = []
    for entry in entries:
        lines = entry.split('\n')
        if len(lines) < 3:
            continue
        
        # Get index
        try:
            index = int(lines[0])
        except:
            continue
        
        # Get timestamps
        timestamp_line = lines[1]
        timestamps = timestamp_line.split(' --> ')
        if len(timestamps) != 2:
            continue
        
        start_time = parse_srt_timestamp(timestamps[0])
        end_time = parse_srt_timestamp(timestamps[1])
        
        # Get text (could be multiple lines)
        text = ' '.join(lines[2:])
        
        words.append({
            "index": index,
            "start": start_time,
            "end": end_time,
            "text": text
        })
    
    return words

def is_sentence_end(text):
    """Check if text ends with sentence-ending punctuation"""
    return bool(re.search(r'[.!?]$', text))

def is_long_pause(curr_end, next_start, threshold=1.0):
    """Check if there's a significant pause between words"""
    return next_start - curr_end > threshold

def count_lines(text, max_chars_per_line=40):
    """Estimate the number of lines based on text length"""
    return len(text) // max_chars_per_line + 1

def merge_words_into_subtitles(words, max_lines=2, max_chars_per_line=40):
    """Merge words into proper subtitles based on sentences and maximum lines"""
    if not words:
        return []
    
    subtitles = []
    current_subtitle = {
        "start": words[0]["start"],
        "end": words[0]["end"],
        "text": words[0]["text"]
    }
    
    for i in range(1, len(words)):
        current_word = words[i]
        
        # Conditions to start a new subtitle:
        # 1. Previous word ended a sentence
        # 2. There's a significant pause
        # 3. Adding this word would make the subtitle too long (more than max_lines)
        
        prev_text = current_subtitle["text"]
        combined_text = f"{prev_text} {current_word['text']}".strip()
        
        # Check if adding this word would exceed line limit
        estimated_lines = count_lines(combined_text, max_chars_per_line)
        
        if (is_sentence_end(prev_text) or 
            is_long_pause(current_subtitle["end"], current_word["start"]) or
            estimated_lines > max_lines):
            
            # Finalize current subtitle
            subtitles.append(current_subtitle)
            
            # Start a new one
            current_subtitle = {
                "start": current_word["start"],
                "end": current_word["end"],
                "text": current_word["text"]
            }
        else:
            # Extend current subtitle
            current_subtitle["text"] = combined_text
            current_subtitle["end"] = current_word["end"]
    
    # Add the last subtitle
    if current_subtitle:
        subtitles.append(current_subtitle)
    
    return subtitles

def save_srt(subtitles, output_path):
    """Save subtitles in SRT format"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, subtitle in enumerate(subtitles, 1):
            start = format_srt_timestamp(subtitle["start"])
            end = format_srt_timestamp(subtitle["end"])
            text = clean_text(subtitle["text"])
            
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    
    print(f"Successfully saved SRT file with {len(subtitles)} subtitles")

def process_srt_file(input_path, output_path=None, max_lines=2, max_chars_per_line=40):
    """Process SRT file to merge word-level subtitles into sentence-level subtitles"""
    # Default output path if not specified
    if not output_path:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_processed.srt"
    
    print(f"Reading subtitles from {input_path}")
    words = read_srt_file(input_path)
    print(f"Found {len(words)} word-level subtitles")
    
    print("Merging into sentence-level subtitles...")
    subtitles = merge_words_into_subtitles(words, max_lines, max_chars_per_line)
    print(f"Created {len(subtitles)} sentence-level subtitles")
    
    print(f"Saving to {output_path}")
    save_srt(subtitles, output_path)
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert word-level SRT subtitles to sentence-level")
    parser.add_argument("input_path", help="Path to input SRT file with word-level subtitles")
    parser.add_argument("-o", "--output", help="Path to save the processed SRT file")
    parser.add_argument("-l", "--max-lines", type=int, default=2, 
                       help="Maximum number of lines per subtitle (default: 2)")
    parser.add_argument("-c", "--max-chars", type=int, default=40,
                       help="Maximum characters per line (default: 40)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input file {args.input_path} not found")
        exit(1)
    
    result_path = process_srt_file(
        args.input_path, 
        args.output, 
        args.max_lines, 
        args.max_chars
    )
    print(f"Conversion complete: {result_path}")

#!/bin/bash

# Default values
DEBUG_MODE=0
output_dir="non_silent_segments"
debug_dir="debug_silence_segments"

# Usage help function
print_usage() {
    echo "Usage: $0 [-h] [--debug]"
    echo "Extract non-silent segments from WAV files in the current directory"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message and exit"
    echo "  --debug        Create debug directory with silent segments for verification"
    echo ""
    echo "Silence detection parameters (edit script to modify):"
    echo "  - Noise threshold: -50dB (lower absolute value = more sensitive)"
    echo "  - Minimum silence duration: 0.5 seconds"
    exit 0
}

# Check for flags
for arg in "$@"; do
    case "$arg" in
        -h|--help)
            print_usage
            ;;
        --debug)
            DEBUG_MODE=1
            ;;
    esac
done

# Check for WAV files
shopt -s nullglob
wav_files=( *.wav )
shopt -u nullglob
if [ ${#wav_files[@]} -eq 0 ]; then
    echo "Error: No WAV files found in the current directory."
    exit 1
fi

# Initialize tracking variables
segments_created=0
silent_segments_created=0

# Process each WAV file
for file in "${wav_files[@]}"; do
    base_name=$(basename "$file" .wav)
    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")
    
    # Detect silence periods
    ffmpeg -i "$file" -af "silencedetect=noise=-50dB:d=0.5" -f null - 2> tmp.txt
    
    # Parse silence timings
    silence_starts=($(grep 'silence_start' tmp.txt | awk '{print $5}'))
    silence_ends=($(grep 'silence_end' tmp.txt | awk '{print $5}' | cut -d '|' -f1))

    # Debug mode: extract silent segments
    if [ "$DEBUG_MODE" -eq 1 ]; then
        silence_segment=1
        for i in "${!silence_starts[@]}"; do
            start="${silence_starts[i]}"
            end="${silence_ends[i]}"
            seg_duration=$(echo "$end - $start" | bc -l)

            if (( $(echo "$seg_duration > 0" | bc -l) )); then
                # Create debug directory if needed
                if [ ! -d "$debug_dir" ]; then
                    mkdir -p "$debug_dir"
                    silent_segments_created=1
                fi
                debug_file="${debug_dir}/${base_name}_silence_${silence_segment}.wav"
                ffmpeg -i "$file" -ss "$start" -to "$end" -c copy "$debug_file" -y
                ((silence_segment++))
            fi
        done
    fi

    # Generate non-silent segments
    segments=()
    prev_end=0
    segment_index=1

    # Calculate non-silent intervals
    for i in "${!silence_starts[@]}"; do
        current_start=${silence_starts[i]}
        current_end=${silence_ends[i]}
        
        if (( $(echo "$prev_end < $current_start" | bc -l) )); then
            segments+=("$prev_end" "$current_start")
        fi
        
        prev_end=$current_end
    done

    # Add final segment
    if (( $(echo "$prev_end < $duration" | bc -l) )); then
        segments+=("$prev_end" "$duration")
    fi

    # Extract non-silent segments
    for ((i=0; i<${#segments[@]}; i+=2)); do
        start="${segments[i]}"
        end="${segments[i+1]}"
        seg_duration=$(echo "$end - $start" | bc -l)

        if (( $(echo "$seg_duration > 0" | bc -l) )); then
            # Create output directory if needed
            if [ ! -d "$output_dir" ]; then
                mkdir -p "$output_dir"
                segments_created=1
            fi
            output_file="${output_dir}/${base_name}_segment${segment_index}.wav"
            ((segment_index++))
            ffmpeg -i "$file" -ss "$start" -to "$end" -c copy "$output_file" -y
        fi
    done
done

# Cleanup and final output
rm -f tmp.txt

if [ -d "$output_dir" ]; then
    echo "Processing complete. Non-silent segments saved in: $output_dir"
else
    echo "Processing complete. No non-silent segments found."
fi

if [ "$DEBUG_MODE" -eq 1 ]; then
    if [ -d "$debug_dir" ]; then
        echo "Debug mode enabled. Silent segments saved in: $debug_dir"
    else
        echo "Debug mode enabled. No silent segments found."
    fi
fi

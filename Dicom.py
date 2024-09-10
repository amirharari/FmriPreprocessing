import os
from datetime import datetime
from enum import Enum
from pydub import AudioSegment  # Ensure you have pydub installed: pip install pydub
from mutagen.mp3 import MP3
from termcolor import colored
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class NarrationType(Enum):
    TRAUMATIC = 'Traumatic'
    SAD = 'Sad'
    NEUTRAL = 'Neutral'


# Function to get the length of an audio file in milliseconds
def get_audio_length(file_path):
    audio = MP3(file_path)
    duration_sec = audio.info.length
    return duration_sec * 1000  # Length in milliseconds


# Function to read text file with proper encoding
def read_text_file_correctly(file_path):
    try:
        with open(file_path, 'r', encoding='utf-16') as file:  # Try with utf-16 first
            content = file.read()
            return content
    except UnicodeError:
        # If utf-16 doesn't work, try utf-8-sig
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            content = file.read()
            return content


def get_narration_type_to_time_frames_mapping(audio_directory_path, txt_path):
    audio_lengths = {NarrationType.TRAUMATIC: None, NarrationType.SAD: None, NarrationType.NEUTRAL: None}

    # Get the audio lengths
    for root, dirs, files in os.walk(audio_directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            audio_length = get_audio_length(file_path)
            if "natural" in file_name.lower():
                audio_lengths[NarrationType.NEUTRAL] = audio_length
            elif "sad" in file_name.lower():
                audio_lengths[NarrationType.SAD] = audio_length
            elif "traumatic" in file_name.lower():
                audio_lengths[NarrationType.TRAUMATIC] = audio_length
    # Check if all necessary files were found
    missing_files = [narration_type for narration_type, length in audio_lengths.items() if length is None]
    if missing_files:
        raise FileNotFoundError(f"Missing audio files for the following narration types: {', '.join([nt.value for nt in missing_files])}")
    # Read and process the text file with special encoding
    file_content = read_text_file_correctly(txt_path)

    mapped_narration_type = []
    in_log_frame = False

    for line in file_content.splitlines():
        line = line.strip()
        if line.startswith("*** LogFrame Start ***"):
            in_log_frame = True
        elif line.startswith("*** LogFrame End ***"):
            in_log_frame = False
        elif in_log_frame and line.startswith("SoundFile:"):
            # Extract the SoundFile path
            sound_file = line.split(": ", 1)[1].strip()

            # Map the sound file path to the appropriate enum
            if 'traumatic' in sound_file.lower():
                mapped_narration_type.append(NarrationType.TRAUMATIC)
            elif 'sad' in sound_file.lower():
                mapped_narration_type.append(NarrationType.SAD)
            elif 'neutral' in sound_file.lower():
                mapped_narration_type.append(NarrationType.NEUTRAL)
            else:
                raise ValueError(f"Invalid SoundFile format: {sound_file}")
    if not mapped_narration_type:
        raise ValueError("No valid sound file mappings were found in the log file.")
    # Initialize variables for session info
    session_start_datetime = None
    subject = None

    for line in file_content.splitlines():
        if line.startswith("SessionStartDateTimeUtc"):
            datetime_str = line.split(": ", 1)[1].strip()
            session_start_datetime = datetime.strptime(datetime_str, '%d/%m/%Y %H:%M:%S')
        elif line.startswith("Subject"):
            subject = line.split(": ", 1)[1].strip()

    current_time_mili = 0
    start_time = 6000  # Start time in milliseconds
    before_break = 5000
    break_time = 30000  # Break time in milliseconds
    current_time_mili += start_time

    narration_type_to_time_frames_mapping = {
        NarrationType.TRAUMATIC: [],
        NarrationType.SAD: [],
        NarrationType.NEUTRAL: []
    }

    # Function to add time frame for each narration type
    def add_time_frame(narration_type, current_time_mili, before_break, audio_lengths, mapping):
        current_time_mili += before_break
        start_time = current_time_mili + 15000
        end_time = current_time_mili + audio_lengths[narration_type] - 15000
        mapping[narration_type].append((start_time, end_time))
        return current_time_mili + audio_lengths[narration_type] + break_time

    # Add time frames based on mapped narration types
    for narration_type in mapped_narration_type:
        current_time_mili = add_time_frame(
            narration_type, current_time_mili, before_break, audio_lengths, narration_type_to_time_frames_mapping
        )

    print("Audio Lengths (in milliseconds):", audio_lengths)
    return narration_type_to_time_frames_mapping


# Function to print time frames with appropriate color
def print_colored_time_frames(time_frames, duration_ms):
    # Define the 15-minute time frame (900,000 ms)
    fifteen_min_ms = 900000  # 15 minutes in milliseconds

    # Colors for each NarrationType
    colors = {
        NarrationType.TRAUMATIC: 'red',
        NarrationType.SAD: 'blue',
        NarrationType.NEUTRAL: 'green'
    }

    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each narration type's time frames
    for narration_type, frames in time_frames.items():
        color = colors[narration_type]
        for start, end in frames:
            if start < fifteen_min_ms:
                end = min(end, fifteen_min_ms)  # Truncate if it exceeds the 15-minute mark
                ax.plot([start, end], [narration_type.value] * 2, color=color, linewidth=5)

    # Set labels and title
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Narration Type')
    ax.set_title('Narration Time Frames within 15 Minutes')
    ax.set_xlim(0, fifteen_min_ms)

    # Create a legend
    legend_handles = [mpatches.Patch(color=colors[narration_type], label=narration_type.value) for narration_type in
                      NarrationType]
    ax.legend(handles=legend_handles, loc='upper right')

    # Display the plot
    plt.show()

# Example usage:
# audio_directory = "C:/your_audio_files"
# txt_path = "C:/your_text_file.txt"
# narration_mapping = get_narration_type_to_time_frames_mapping(audio_directory, txt_path)
# print_colored_time_frames(narration_mapping, 900000)  # 15-minute time frame

# video_surveillance_agent.py
# Video Surveillance Agent using LLaVA for person detection
#
# This agent:
# 1. Extracts frames from a video every N seconds
# 2. Uses LLaVA to detect if a person is present in each frame
# 3. Tracks and reports when people enter and exit the scene
#
# Usage:
#   python video_surveillance_agent.py <video_path> [--interval=2] [--output-dir=frames]

import os
import sys
import cv2
import ollama
from datetime import timedelta
from typing import Optional
from dataclasses import dataclass
from PIL import Image

# =============================================================================
# CONFIGURATION
# =============================================================================

VISION_MODEL = "llava"
DEFAULT_INTERVAL = 2  # seconds between frames
DEFAULT_OUTPUT_DIR = "surveillance_frames"

# Prompt for person detection
DETECTION_PROMPT = """Look at this image carefully. Is there a person (or people) visible in this scene?

Answer with ONLY one of these two words:
- "YES" if you can see any person or people in the image
- "NO" if there are no people visible in the image

Your answer (YES or NO):"""


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FrameAnalysis:
    """Result of analyzing a single frame."""
    frame_number: int
    timestamp: float
    timestamp_str: str
    person_detected: bool
    raw_response: str
    image_path: str


@dataclass
class Event:
    """An entry or exit event."""
    event_type: str  # "ENTER" or "EXIT"
    timestamp: float
    timestamp_str: str
    frame_number: int


# =============================================================================
# VIDEO PROCESSING
# =============================================================================

def extract_frames(video_path: str, interval_seconds: float = 2.0, 
                   output_dir: str = DEFAULT_OUTPUT_DIR, 
                   max_dimension: int = 512) -> list[tuple[int, float, str]]:
    """
    Extract frames from video at specified intervals.
    
    Args:
        video_path: Path to the video file
        interval_seconds: Seconds between frame captures
        output_dir: Directory to save extracted frames
        max_dimension: Max width/height for resized frames
    
    Returns:
        List of (frame_number, timestamp_seconds, image_path) tuples
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📹 Video: {video_path}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {timedelta(seconds=int(duration))}")
    print(f"   Extracting frames every {interval_seconds} seconds...")
    
    # Calculate frame interval
    frame_interval = int(fps * interval_seconds)
    if frame_interval < 1:
        frame_interval = 1
    
    extracted = []
    frame_num = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num % frame_interval == 0:
            # Calculate timestamp
            timestamp = frame_num / fps
            
            # Resize frame for faster processing
            height, width = frame.shape[:2]
            if max(width, height) > max_dimension:
                scale = max_dimension / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height), 
                                   interpolation=cv2.INTER_AREA)
            
            # Save frame
            image_path = os.path.join(output_dir, f"frame_{frame_num:06d}.jpg")
            cv2.imwrite(image_path, frame)
            
            extracted.append((frame_num, timestamp, image_path))
        
        frame_num += 1
    
    cap.release()
    
    print(f"   Extracted {len(extracted)} frames")
    return extracted


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes:02d}:{secs:02d}"


# =============================================================================
# PERSON DETECTION
# =============================================================================

def detect_person(image_path: str, verbose: bool = False) -> tuple[bool, str]:
    """
    Use LLaVA to detect if a person is in the image.
    
    Args:
        image_path: Path to the image file
        verbose: Print debug info
    
    Returns:
        Tuple of (person_detected: bool, raw_response: str)
    """
    try:
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                'role': 'user',
                'content': DETECTION_PROMPT,
                'images': [image_path]
            }]
        )
        
        raw_response = response['message']['content'].strip()
        
        # Parse response - look for YES or NO
        response_upper = raw_response.upper()
        
        if "YES" in response_upper:
            person_detected = True
        elif "NO" in response_upper:
            person_detected = False
        else:
            # Ambiguous response - assume no person if we can't tell
            if verbose:
                print(f"    ⚠️  Ambiguous response: {raw_response[:50]}")
            person_detected = False
        
        return person_detected, raw_response
        
    except Exception as e:
        print(f"    ❌ Error analyzing frame: {e}")
        return False, f"Error: {e}"


def analyze_frames(frames: list[tuple[int, float, str]], 
                   verbose: bool = True) -> list[FrameAnalysis]:
    """
    Analyze all frames for person detection.
    
    Args:
        frames: List of (frame_number, timestamp, image_path) tuples
        verbose: Show progress
    
    Returns:
        List of FrameAnalysis results
    """
    results = []
    total = len(frames)
    
    print(f"\n🔍 Analyzing {total} frames for person detection...")
    print(f"   Using model: {VISION_MODEL}")
    print()
    
    for i, (frame_num, timestamp, image_path) in enumerate(frames):
        timestamp_str = format_timestamp(timestamp)
        
        if verbose:
            print(f"   [{i+1}/{total}] Frame {frame_num} at {timestamp_str}...", end=" ", flush=True)
        
        person_detected, raw_response = detect_person(image_path, verbose=False)
        
        if verbose:
            status = "👤 PERSON" if person_detected else "✓ Empty"
            print(status)
        
        results.append(FrameAnalysis(
            frame_number=frame_num,
            timestamp=timestamp,
            timestamp_str=timestamp_str,
            person_detected=person_detected,
            raw_response=raw_response,
            image_path=image_path
        ))
    
    return results


# =============================================================================
# EVENT DETECTION
# =============================================================================

def detect_events(analyses: list[FrameAnalysis]) -> list[Event]:
    """
    Detect entry and exit events from frame analyses.
    
    An ENTER event occurs when person_detected changes from False to True.
    An EXIT event occurs when person_detected changes from True to False.
    
    Args:
        analyses: List of FrameAnalysis results in chronological order
    
    Returns:
        List of Event objects
    """
    if not analyses:
        return []
    
    events = []
    previous_state = False  # Assume no person at start
    
    for analysis in analyses:
        current_state = analysis.person_detected
        
        if current_state and not previous_state:
            # Person entered
            events.append(Event(
                event_type="ENTER",
                timestamp=analysis.timestamp,
                timestamp_str=analysis.timestamp_str,
                frame_number=analysis.frame_number
            ))
        elif not current_state and previous_state:
            # Person exited
            events.append(Event(
                event_type="EXIT",
                timestamp=analysis.timestamp,
                timestamp_str=analysis.timestamp_str,
                frame_number=analysis.frame_number
            ))
        
        previous_state = current_state
    
    return events


def generate_report(analyses: list[FrameAnalysis], events: list[Event], 
                    video_path: str, interval: float) -> str:
    """Generate a summary report of the surveillance analysis."""
    
    report_lines = [
        "=" * 60,
        "📊 VIDEO SURVEILLANCE REPORT",
        "=" * 60,
        "",
        f"Video: {video_path}",
        f"Frame interval: {interval} seconds",
        f"Total frames analyzed: {len(analyses)}",
        "",
        "-" * 60,
        "PERSON DETECTION EVENTS",
        "-" * 60,
    ]
    
    if not events:
        report_lines.append("No entry/exit events detected.")
    else:
        for event in events:
            icon = "🚶➡️ " if event.event_type == "ENTER" else "🚶⬅️ "
            report_lines.append(
                f"{icon}{event.event_type} at {event.timestamp_str} (frame {event.frame_number})"
            )
    
    report_lines.extend([
        "",
        "-" * 60,
        "FRAME-BY-FRAME SUMMARY",
        "-" * 60,
    ])
    
    for analysis in analyses:
        status = "👤 Person" if analysis.person_detected else "   Empty"
        report_lines.append(f"[{analysis.timestamp_str}] {status}")
    
    # Statistics
    frames_with_person = sum(1 for a in analyses if a.person_detected)
    frames_without_person = len(analyses) - frames_with_person
    
    report_lines.extend([
        "",
        "-" * 60,
        "STATISTICS",
        "-" * 60,
        f"Frames with person: {frames_with_person} ({100*frames_with_person/len(analyses):.1f}%)",
        f"Frames without person: {frames_without_person} ({100*frames_without_person/len(analyses):.1f}%)",
        f"Total events: {len(events)}",
        "=" * 60,
    ])
    
    return "\n".join(report_lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function for the video surveillance agent."""
    
    print("=" * 60)
    print("🎥 Video Surveillance Agent")
    print("=" * 60)
    print()
    
    # Parse arguments
    video_path = None
    interval = DEFAULT_INTERVAL
    output_dir = DEFAULT_OUTPUT_DIR
    
    for arg in sys.argv[1:]:
        if arg.startswith("--interval="):
            interval = float(arg.split("=")[1])
        elif arg.startswith("--output-dir="):
            output_dir = arg.split("=")[1]
        elif not arg.startswith("--") and not video_path:
            video_path = arg
    
    if not video_path:
        print("❌ No video specified!")
        print()
        print("Usage: python video_surveillance_agent.py <video_path> [options]")
        print()
        print("Options:")
        print("  --interval=N     Seconds between frames (default: 2)")
        print("  --output-dir=DIR Directory for extracted frames")
        print()
        print("Example:")
        print("  python video_surveillance_agent.py surveillance.mp4 --interval=2")
        sys.exit(1)
    
    # Check Ollama
    print(f"🔌 Checking Ollama ({VISION_MODEL})...")
    try:
        ollama.list()
        print("✅ Ollama is running")
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("   Please start Ollama: ollama serve")
        print(f"   And pull the model: ollama pull {VISION_MODEL}")
        sys.exit(1)
    
    print()
    
    # Extract frames
    try:
        frames = extract_frames(video_path, interval, output_dir)
    except Exception as e:
        print(f"❌ Error extracting frames: {e}")
        sys.exit(1)
    
    if not frames:
        print("❌ No frames extracted!")
        sys.exit(1)
    
    # Analyze frames
    analyses = analyze_frames(frames, verbose=True)
    
    # Detect events
    events = detect_events(analyses)
    
    # Generate report
    report = generate_report(analyses, events, video_path, interval)
    print()
    print(report)
    
    # Save report
    report_path = os.path.join(output_dir, "surveillance_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n📄 Report saved to: {report_path}")
    
    # Print event summary
    if events:
        print("\n" + "=" * 60)
        print("🎯 DETECTED EVENTS SUMMARY")
        print("=" * 60)
        for event in events:
            if event.event_type == "ENTER":
                print(f"  🚶➡️  Person ENTERED at {event.timestamp_str}")
            else:
                print(f"  🚶⬅️  Person EXITED at {event.timestamp_str}")
        print("=" * 60)
    else:
        print("\n✓ No person detected throughout the video.")


if __name__ == "__main__":
    main()

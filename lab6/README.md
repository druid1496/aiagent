# Lab 6: Vision-Language Agents

Vision-language agents using Ollama's LLaVA model for image understanding and video surveillance.

## Exercises

1. **Exercise 1**: Vision-Language Chat Agent (`vision_chat_agent.py`)
2. **Exercise 2**: Video Surveillance Agent (`video_surveillance_agent.py`)

## Prerequisites

1. **Ollama** must be installed and running:
   ```bash
   # Install Ollama (macOS)
   brew install ollama
   
   # Start Ollama server
   ollama serve
   
   # Pull the LLaVA vision model
   ollama pull llava
   ```

2. **Python dependencies**:
   ```bash
   cd lab6
   source myenvlab6/bin/activate
   pip install -r requirements.txt
   ```

---

# Exercise 1: Vision-Language Chat Agent

A LangGraph-based chat agent that allows multi-turn conversations about uploaded images.

## Usage

```bash
source myenvlab6/bin/activate
python vision_chat_agent.py <image_path>

# Examples
python vision_chat_agent.py photo.jpg --new
python vision_chat_agent.py photo.jpg --resolution=512
```

## Commands
| Command | Description |
|---------|-------------|
| `history` | Show conversation history |
| `quit` | Exit (conversation saved) |
| `Ctrl+C` | Exit (conversation saved) |

## Graph Structure

```
START → get_user_input → [conditional] → call_vision_llm → print_response → get_user_input
              ↓
             END
```

## Sample Output (photo.jpg - Tree vs Graph Diagram)

### intro.py Result:
```
The image is a simple, hand-drawn whiteboard illustration with a few pink polka 
dots scattered across it. On the left side, there's a tree with branches and a 
leaf labeled as "tree." In the center of the board, the words "graph" are written 
in black text, and beneath them is a small circle with several red lines pointing 
to it from different directions, suggesting a connection or relationship between 
these points. To the right of the central circle, there's a sequence of three pink 
circles connected by black lines, indicating a series or sequence. The overall 
theme suggests a concept related to trees, structures, or connections, possibly 
in the context of technology, chemistry, or similar fields where graphs and 
networks are common.
```

---

# Exercise 2: Video Surveillance Agent

A video surveillance agent that extracts frames from a video and uses LLaVA to detect when people enter and exit the scene.

## How It Works

1. **Extract Frames**: Uses OpenCV to extract frames every N seconds
2. **Person Detection**: Sends each frame to LLaVA asking if a person is visible
3. **Event Detection**: Tracks state changes to detect ENTER and EXIT events
4. **Generate Report**: Creates a summary report with timestamps

## Usage

```bash
source myenvlab6/bin/activate
python video_surveillance_agent.py <video_path> [--interval=2]

# Examples
python video_surveillance_agent.py vedio.mp4
python video_surveillance_agent.py vedio.mp4 --interval=1
```

## Sample Output (vedio.mp4)

### Console Output:
```
============================================================
🎥 Video Surveillance Agent
============================================================

🔌 Checking Ollama (llava)...
✅ Ollama is running

📹 Video: vedio.mp4
   FPS: 29.98
   Total frames: 3613
   Duration: 0:02:00
   Extracting frames every 2.0 seconds...
   Extracted 62 frames

🔍 Analyzing 62 frames for person detection...
   Using model: llava

   [1/62] Frame 0 at 00:00... ✓ Empty
   [2/62] Frame 59 at 00:01... ✓ Empty
   [3/62] Frame 118 at 00:03... ✓ Empty
   ...
   [14/62] Frame 767 at 00:25... 👤 PERSON
   [15/62] Frame 826 at 00:27... 👤 PERSON
   [16/62] Frame 885 at 00:29... ✓ Empty
   [17/62] Frame 944 at 00:31... 👤 PERSON
   [18/62] Frame 1003 at 00:33... ✓ Empty
   ...
   [32/62] Frame 1829 at 01:01... 👤 PERSON
   [33/62] Frame 1888 at 01:02... ✓ Empty
   ...
   [62/62] Frame 3599 at 02:00... ✓ Empty
```

### Surveillance Report:
```
============================================================
📊 VIDEO SURVEILLANCE REPORT
============================================================

Video: vedio.mp4
Frame interval: 2.0 seconds
Total frames analyzed: 62

------------------------------------------------------------
PERSON DETECTION EVENTS
------------------------------------------------------------
🚶➡️ ENTER at 00:25 (frame 767)
🚶⬅️ EXIT at 00:29 (frame 885)
🚶➡️ ENTER at 00:31 (frame 944)
🚶⬅️ EXIT at 00:33 (frame 1003)
🚶➡️ ENTER at 01:01 (frame 1829)
🚶⬅️ EXIT at 01:02 (frame 1888)

------------------------------------------------------------
FRAME-BY-FRAME SUMMARY
------------------------------------------------------------
[00:00]    Empty
[00:01]    Empty
[00:03]    Empty
[00:05]    Empty
[00:07]    Empty
[00:09]    Empty
[00:11]    Empty
[00:13]    Empty
[00:15]    Empty
[00:17]    Empty
[00:19]    Empty
[00:21]    Empty
[00:23]    Empty
[00:25] 👤 Person
[00:27] 👤 Person
[00:29]    Empty
[00:31] 👤 Person
[00:33]    Empty
[00:35]    Empty
[00:37]    Empty
[00:39]    Empty
[00:41]    Empty
[00:43]    Empty
[00:45]    Empty
[00:47]    Empty
[00:49]    Empty
[00:51]    Empty
[00:53]    Empty
[00:55]    Empty
[00:57]    Empty
[00:59]    Empty
[01:01] 👤 Person
[01:02]    Empty
[01:04]    Empty
[01:06]    Empty
[01:08]    Empty
[01:10]    Empty
[01:12]    Empty
[01:14]    Empty
[01:16]    Empty
[01:18]    Empty
[01:20]    Empty
[01:22]    Empty
[01:24]    Empty
[01:26]    Empty
[01:28]    Empty
[01:30]    Empty
[01:32]    Empty
[01:34]    Empty
[01:36]    Empty
[01:38]    Empty
[01:40]    Empty
[01:42]    Empty
[01:44]    Empty
[01:46]    Empty
[01:48]    Empty
[01:50]    Empty
[01:52]    Empty
[01:54]    Empty
[01:56]    Empty
[01:58]    Empty
[02:00]    Empty

------------------------------------------------------------
STATISTICS
------------------------------------------------------------
Frames with person: 4 (6.5%)
Frames without person: 58 (93.5%)
Total events: 6
============================================================
```

### Event Summary Table

| Time | Event | Frame |
|------|-------|-------|
| **00:25** | 🚶➡️ Person ENTERED | 767 |
| **00:29** | 🚶⬅️ Person EXITED | 885 |
| **00:31** | 🚶➡️ Person ENTERED | 944 |
| **00:33** | 🚶⬅️ Person EXITED | 1003 |
| **01:01** | 🚶➡️ Person ENTERED | 1829 |
| **01:02** | 🚶⬅️ Person EXITED | 1888 |

### Analysis
- **Video Duration**: 2 minutes (120 seconds)
- **Frames Analyzed**: 62 frames (every 2 seconds)
- **Person Detected In**: 4 frames (6.5%)
- **Total Events**: 6 (3 entries, 3 exits)

The surveillance detected a person appearing briefly around:
1. **00:25-00:29**: First appearance (~4 seconds)
2. **00:31-00:33**: Brief appearance (~2 seconds)
3. **01:01-01:02**: Brief appearance (~1 second)

---

## Files

| File | Description |
|------|-------------|
| `vision_chat_agent.py` | Exercise 1: LangGraph vision chat agent |
| `video_surveillance_agent.py` | Exercise 2: Video surveillance with person detection |
| `intro.py` | Simple Ollama LLaVA example |
| `photo.jpg` | Sample image (tree vs graph diagram) |
| `vedio.mp4` | Surveillance test video |
| `requirements.txt` | Python dependencies |
| `surveillance_frames/` | Extracted video frames |
| `surveillance_frames/surveillance_report.txt` | Full surveillance report |

## Troubleshooting

### "Cannot connect to Ollama"
```bash
ollama serve
```

### "Model not found"
```bash
ollama pull llava
```

### Slow processing
- Reduce image resolution: `--resolution=512`
- Increase frame interval: `--interval=5`

# Hinge Opener Bot

An automated Android bot that uses OCR and AI to generate personalized opening messages for dating profiles.

## ⚠️ Important Disclaimers

**READ BEFORE USE:**
- This tool may violate Hinge's Terms of Service and could result in account suspension or ban
- Automated interactions on dating platforms can create inauthentic experiences for other users
- Use responsibly and consider the ethical implications of automation in dating
- This is for educational/research purposes - the authors are not responsible for misuse
- Always respect others' time and genuine connections on dating platforms

## How It Works

1. **Screenshot Capture**: Uses ADB to take screenshots of Hinge profiles on Android
2. **OCR Text Extraction**: Employs Tesseract to extract profile text and details
3. **AI Message Generation**: Sends profile data to OpenAI GPT to craft personalized openers
4. **Automated Interaction**: Executes tap/swipe sequences to send messages

## Features

- **Smart Scrolling**: Automatically scrolls through profiles with overlap detection
- **Duplicate Prevention**: Removes duplicate text lines from OCR extraction
- **Robust Error Handling**: Recovers from ADB disconnections and app crashes
- **Configurable Actions**: JSON-based configuration for tap sequences and timing
- **Multiple Input Methods**: Fallback typing methods (direct input → clipboard → char-by-char)

## Prerequisites

### System Requirements
- **macOS/Linux/Windows** with ADB support
- **Android device** with USB debugging enabled
- **Python 3.7+**
- **Tesseract OCR** installed
- **OpenAI API key**

### Android Setup
1. Enable Developer Options on your Android device
2. Enable USB Debugging
3. Connect device and authorize computer
4. Install Hinge app and log in

## Installation

### 1. Install System Dependencies

**macOS (using Homebrew):**
```bash
brew install android-platform-tools tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install android-tools-adb tesseract-ocr
```

**Windows:**
- Download [Android Platform Tools](https://developer.android.com/studio/releases/platform-tools)
- Download [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)

### 2. Clone Repository
```bash
git clone https://github.com/yourusername/hinge-opener-bot.git
cd hinge-opener-bot
```

### 3. Setup Python Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure Environment
Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
MODEL=gpt-4o-mini
```

## Configuration

### actions.json

Configure the bot behavior by editing `actions.json`:

```json
{
    "device_id": "YOUR_DEVICE_ID",
    "scroll": {
        "start": [540, 1989],
        "end": [540, 800],
        "duration_ms": 450
    },
    "max_scrolls": 40,
    "stop_when_stationary": 3,
    "max_profiles": 10,
    "button_sequence": [
        { "tap_px": [900, 1700], "name": "Like" },
        { "tap_px": [500, 1500], "name": "FocusText" },
        { "type_text": true },
        { "tap_px": [950, 2200], "name": "Done" },
        { "tap_px": [550, 1800], "name": "Send" }
    ]
}
```

#### Configuration Options:
- **device_id**: Your Android device ID (get with `adb devices`)
- **scroll**: Screen coordinates for scrolling through profiles
- **button_sequence**: Sequence of taps/actions to send messages
- **max_scrolls**: Maximum number of scrolls per profile
- **stop_when_stationary**: Stop scrolling after N identical screenshots

### Finding Screen Coordinates

To find the correct tap coordinates for your device:

1. **Take a screenshot:**
   ```bash
   adb shell screencap -p > screen.png
   ```

2. **Open in image editor** and note pixel coordinates for:
   - Like button
   - Text input field
   - Send button
   - Done/Submit button

3. **Test coordinates:**
   ```bash
   adb shell input tap X Y
   ```

## Usage

### Basic Usage
```bash
python hinge_auto.py \
    --pkg co.hinge.app \
    --cfg actions.json \
    --out ocr_dump.txt \
    --prompt "Create witty, personalized openers based on profile details"
```

### Command Line Options
- `--pkg`: App package name (default: `co.hinge.app`)
- `--cfg`: Configuration file path (default: `actions.json`)
- `--out`: OCR output file (default: `ocr_dump.txt`)
- `--prompt`: Custom prompt for AI message generation
- `--sleep_between_cycles`: Pause between profiles (default: 5.0s)

### Example with Custom Prompt
```bash
python hinge_auto.py \
    --prompt "Write humorous, short openers that reference specific profile details. Keep it under 2 lines and avoid generic compliments." \
    --sleep_between_cycles 10.0
```

## How the AI Prompt Works

The bot sends profile information to OpenAI with this system prompt:
- Craft respectful, specific dating openers
- 1-2 short lines maximum
- Reference concrete profile details
- No generic compliments or pickup lines
- Be witty and funny
- Use only ASCII characters (no emojis)

Example generated opener:
```
"Apparently Hinge is showing us most compatible. I see you're into rock climbing - any good spots you'd recommend for someone who's more 'indoor wall' than 'outdoor cliff'? Shall we test this compatibility theory?"
```

## Requirements.txt

```
pillow>=9.0.0
numpy>=1.21.0
opencv-python>=4.5.0
pytesseract>=0.3.8
openai>=1.0.0
python-dotenv>=0.19.0
```

## File Structure

```
hinge-opener-bot/
├── hinge_auto.py          # Main bot script
├── actions.json           # Configuration file
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── .gitignore            # Git ignore file
├── README.md             # This file
└── examples/
    ├── actions_pixel2.json    # Example config for Pixel 2
    └── actions_samsung.json   # Example config for Samsung
```

## Troubleshooting

### Common Issues

**ADB Device Not Found:**
```bash
# Check connected devices
adb devices

# Restart ADB server
adb kill-server
adb start-server
```

**OCR Not Working:**
- Ensure Tesseract is installed and in PATH
- Try different OCR configurations in the code
- Check image quality/resolution

**App Crashes:**
- The bot includes automatic recovery
- Verify coordinates are correct for your device
- Reduce scroll speed or add delays

**OpenAI API Errors:**
- Check your API key is valid
- Ensure you have sufficient credits
- Verify model name is correct

### Device-Specific Setup

Different Android devices may require different coordinates. Create device-specific config files:

```bash
# For different devices
python hinge_auto.py --cfg actions_pixel.json
python hinge_auto.py --cfg actions_samsung.json
```

## Safety Features

- **Error Recovery**: Automatically restarts ADB and app on crashes
- **Duplicate Detection**: Prevents sending identical messages
- **Rate Limiting**: Configurable delays between actions
- **ASCII Sanitization**: Cleans text to prevent ADB injection
- **Multiple Input Methods**: Fallback typing strategies

## Legal and Ethical Considerations

### Terms of Service
- Review Hinge's Terms of Service before use
- Automated interactions may violate platform policies
- Risk of account suspension or permanent ban

### Ethical Use
- Consider impact on other users' experiences
- Avoid mass messaging or spam behavior
- Use for research/educational purposes only
- Respect others seeking genuine connections

### Privacy
- OCR data is stored locally in `ocr_dump.txt`
- OpenAI receives profile text for message generation
- No personal data is stored long-term

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Disclaimer

This project is for educational and research purposes only. The authors are not responsible for any misuse, account bans, or violations of Terms of Service. Use at your own risk and always respect the platforms and people you interact with.

## Support

For issues and questions:
- Check the troubleshooting section
- Review existing GitHub issues
- Create a new issue with detailed information

---

**Remember: Technology should enhance human connections, not replace genuine interaction. Use responsibly.**
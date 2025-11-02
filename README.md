# DeepSeek-OCR CPU (Windows)

A local, CPU-only setup for running DeepSeek-OCR on Windows with a clean, focused CLI. This fork emphasizes stable inference, minimal noise in the console, and practical tooling for single-image and batch/parallel workflows.

## Prerequisites
- Python 3.10+ (Windows)
- Git

Optional (installed automatically via requirements, but if you see ImportErrors):
- torch / torchvision
- addict

## Installation (cmd)
1. Clone the repository:
   ```powershell
   git clone https://github.com/StanTheGorilla/Deepseek-Ocr-Cpu.git
   cd Deepseek-Ocr-Cpu
   ```
2. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   If you encounter import errors:
   ```powershell
   pip install torchvision addict
   ```

## Usage
Run the interactive CLI:
```powershell
python run_cpu.py
```
You can also start with an image path:
```powershell
python run_cpu.py "C:\path\to\your\image.jpg"
```

### Interactive commands
- `load <path>`: Load a single image
- `batch <dir or files>`: Load multiple images for batch/parallel processing
- `parallel <query>`: Run the query across all loaded images
- `compare <query>`: Similar to parallel; intended for side-by-side review
- `info`: Show details for the current image
- `history` or `h`: Preview conversation history
- `save`: Save conversation to `./output/parallel_chat_<timestamp>.txt`
- `clear`: Clear conversation history and batch images
- `help`: Show help
- `quit` or `exit`: Close the app

### Output
- Results and logs are saved under `./output/`
- The CLI prints clean `answer:` lines without progress banners or technical warnings

## Configuration
The script disables noisy progress bars and warnings via:
- Environment variables:
  - `TOKENIZERS_PARALLELISM=false`
  - `TQDM_DISABLE=1`
- Transformers logging set to ERROR and progress bars disabled
- Custom stderr filtering to suppress known harmless warnings and tqdm lines

You can set a default image path or list inside `main()` (look for the comment near the configuration block) in <mcfile name="run_cpu.py" path="c:\Users\cieka\Downloads\cor\run_cpu.py"></mcfile>.

## Troubleshooting
- ImportError: `torchvision` or `addict` not found
  - Install missing packages: `pip install torchvision addict`
- Progress lines like `image: 0it [...]` still appear
  - Make sure you’re running the updated code. The CLI now removes these both from stderr and captured stdout.
- Excessive equals (`===`) banners
  - The CLI collapses stacked separators; if you still see standalone banners, update to the latest commit.

## License
See the LICENSE file for details.

Original project: https://github.com/deepseek-ai/DeepSeek-OCR

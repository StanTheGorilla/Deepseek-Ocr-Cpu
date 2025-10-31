"""
DeepSeek OCR - Parallel Processing Interface
Enhanced with parallel processing for multiple images simultaneously

CONFIGURE YOUR DEFAULT IMAGES:
Look for the "CONFIGURE YOUR DEFAULT IMAGE PATH HERE" section in the main() function
You can set a single image or multiple images for batch processing.
"""

from transformers import AutoModel, AutoTokenizer
import torch
import os
from PIL import Image
import re
from datetime import datetime
import gc
from threading import Lock
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import time
import json
from pathlib import Path
from collections import defaultdict
import multiprocessing

def clean_output(text):
    """Clean up output by removing tags and formatting issues"""
    if not isinstance(text, str):
        text = str(text)
    
    text = re.sub(r'</?image>', '', text)
    text = re.sub(r'</image>.*', '', text, flags=re.DOTALL)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def detect_position_query(user_message):
    """Detect if the user is asking about text position"""
    position_keywords = [
        'position', 'where', 'location', 'layout', 'arrangement',
        'spatial', 'coordinates', 'placement', 'distribution',
        'show text position', 'text position', 'position of text',
        'where is text', 'text location', 'spatial layout',
        'show position', 'arrangement of text', 'text distribution'
    ]
    
    user_lower = user_message.lower()
    return any(keyword in user_lower for keyword in position_keywords)

def create_spatial_text_map(text_content, image_width=800, image_height=600):
    """Create a clean spatial representation of text positions in terminal"""
    if not text_content:
        return "No text detected"
    
    # Parse text content for position information
    lines = text_content.strip().split('\n')
    
    # Create a spatial grid representation
    grid_width = 80  # Terminal width
    grid_height = 30  # Terminal height (reduced for cleaner output)
    
    # Initialize empty grid
    grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]
    
    # Parse text lines and try to assign positions
    text_items = []
    for line in lines:
        line = line.strip()
        if line and len(line) > 0:
            # Determine position based on content type
            if 'top' in line.lower() or 'header' in line.lower() or 'title' in line.lower():
                y_pos = 2
                x_pos = 5
            elif 'bottom' in line.lower() or 'footer' in line.lower():
                y_pos = grid_height - 3
                x_pos = 5
            elif 'center' in line.lower() or 'middle' in line.lower():
                y_pos = grid_height // 2
                x_pos = (grid_width - len(line)) // 2 if len(line) < grid_width else 5
            elif 'right' in line.lower():
                y_pos = len(text_items) + 8
                x_pos = grid_width - len(line) - 5 if len(line) < grid_width - 10 else 5
            elif 'left' in line.lower():
                y_pos = len(text_items) + 8
                x_pos = 5
            else:
                # Regular text placement based on order
                y_pos = len(text_items) * 2 + 5
                x_pos = 10
            
            # Ensure positions are within bounds
            y_pos = min(y_pos, grid_height - 1)
            x_pos = min(x_pos, max(0, grid_width - len(line) - 1))
            
            # Place text in grid
            if y_pos < grid_height:
                for i, char in enumerate(line):
                    if x_pos + i < grid_width:
                        grid[y_pos][x_pos + i] = char
                
                text_items.append(f"  {line}")
    
    # Convert grid to clean string representation
    grid_lines = []
    grid_lines.append("\nText Position Map:")
    grid_lines.append("")
    
    # Only show non-empty rows
    for row in grid:
        row_str = ''.join(row).rstrip()
        if row_str:  # Only include rows with content
            grid_lines.append(row_str)
    
    grid_lines.append("")
    grid_lines.append("Text elements detected:")
    
    # Add detected text items (simplified list)
    for item in text_items[:15]:  # Show up to 15 items
        grid_lines.append(item)
    
    if len(text_items) > 15:
        grid_lines.append(f"  ... and {len(text_items) - 15} more")
    
    return '\n'.join(grid_lines)

# Resource monitoring removed for clean chat interface

def extract_text_parallel(image_path, output_dir):
    """Extract text from image using parallel processing"""
    try:
        # Preprocess image for better results
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Apply enhancement for better text recognition
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.1)
            
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.05)
            
            # Save preprocessed image for model
            enhanced_path = os.path.join(output_dir, f"enhanced_{os.path.basename(image_path)}")
            img.save(enhanced_path)
            return enhanced_path
    except Exception as e:
        print(f"  ⚠️ Image preprocessing failed: {e}")
        return image_path

class ParallelProcessor:
    """Handles parallel processing of multiple images"""
    
    def __init__(self, model, tokenizer, max_workers=None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_workers = max_workers or min(16, multiprocessing.cpu_count())
        self.output_queue = queue.Queue()
        self.processing_lock = Lock()
        self.batch_stats = {
            'processed': 0,
            'total': 0,
            'errors': 0
        }
        
    def process_image_batch(self, image_paths, user_queries=None):
        """Process multiple images in parallel"""
        if not image_paths:
            return []
        
        if user_queries is None:
            user_queries = ["Describe everything you see completely"] * len(image_paths)
        elif len(user_queries) == 1:
            user_queries = [user_queries[0]] * len(image_paths)
        
        # Reset stats
        with self.processing_lock:
            self.batch_stats = {'processed': 0, 'total': len(image_paths), 'errors': 0}
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self._process_single_image, path, query, idx): (path, idx)
                for idx, (path, query) in enumerate(zip(image_paths, user_queries))
            }
            
            # Process completed tasks
            for future in as_completed(future_to_path):
                path, idx = future_to_path[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                    
                    with self.processing_lock:
                        self.batch_stats['processed'] += 1
                        
                except Exception as e:
                    with self.processing_lock:
                        self.batch_stats['errors'] += 1
                    results.append((idx, f"❌ Error processing {path}: {e}"))
        
        # Sort by original index
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    def _process_single_image(self, image_path, user_query, index):
        """Process a single image (worker function)"""
        try:
            # Direct, uncensored prompt
            full_prompt = f"<image>\nDescribe everything you see completely and directly. Extract all text exactly as shown without filtering. Be thorough.\n\n{user_query}"
            
            # Process image with optimized settings
            import io
            import sys
            
            output_dir = './output'
            os.makedirs(output_dir, exist_ok=True)
            
            # Redirect stdout to capture output
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            try:
                # Memory cleanup for each worker
                gc.collect()
                
                result = self.model.infer(
                    self.tokenizer,
                    prompt=full_prompt,
                    image_file=image_path,
                    output_path=output_dir,
                    base_size=1280,
                    image_size=1280,
                    crop_mode=False,
                    save_results=True,
                    test_compress=False
                )
            finally:
                sys.stdout = old_stdout
                captured_text = captured_output.getvalue()
            
            # Extract response
            response_text = self._extract_response(result, captured_text, output_dir)
            
            if not response_text:
                response_text = f"⚠️ No analysis result for {os.path.basename(image_path)}"
            
            return clean_output(response_text)
            
        except Exception as e:
            return f"❌ Error analyzing {os.path.basename(image_path)}: {str(e)}"
    
    def _extract_response(self, result, captured_text, output_dir):
        """Extract response from model output - IMPROVED filtering"""
        response_text = None
        
        # Check direct result
        if result is not None:
            if isinstance(result, dict):
                response_text = result.get('text', result.get('output', result.get('response')))
            elif isinstance(result, str) and len(result.strip()) > 10:
                response_text = result
        
        # Check captured stdout - ONLY filter technical messages, preserve ALL content
        if not response_text and captured_text and len(captured_text.strip()) > 10:
            lines = [line.strip() for line in captured_text.split('\n') if line.strip()]
            
            # MINIMAL filtering - ONLY remove PyTorch/system technical messages
            technical_prefixes = (
                'The attention', 'Setting', 'The `seen', 'UserWarning:', 'FutureWarning:',
                'BASE:', 'NO PATCHES', 'torch.Size', 'warnings.warn', 'DeprecationWarning',
                'You are using', 'Some weights', 'You should probably TRAIN', 'get_max_cache',
                'The attention layers', 'The `seen_tokens`', 'The attention mask',
                'Setting `pad_token_id`', 'User provided device_type'
            )
            
            # Filter ONLY technical PyTorch messages, keep EVERYTHING else
            content_lines = [
                line for line in lines 
                if not any(line.startswith(prefix) for prefix in technical_prefixes)
                and len(line) > 1  # Only exclude empty/single-char lines
            ]
            
            # Use ALL filtered content - no length requirements, no content-based filtering
            if content_lines:
                response_text = '\n'.join(content_lines)
        
        # Check saved files - IMPROVED
        if not response_text:
            try:
                result_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
                for txt_file in sorted(result_files, key=lambda f: os.path.getmtime(os.path.join(output_dir, f)), reverse=True):
                    file_path = os.path.join(output_dir, txt_file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if (len(content) > 50 and 'error' not in content.lower() and
                                'BASE:' not in content and 'torch.Size' not in content):
                                response_text = content
                                break
                    except Exception:
                        continue
            except Exception:
                pass
        
        return response_text

class OptimizedChatSession:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.conversation_history = []
        self.current_image = None
        
        # Initialize parallel processor
        self.parallel_processor = ParallelProcessor(model, tokenizer)
        
        # DIRECT SYSTEM PROMPT - Complete, uncensored analysis
        self.system_prompt = "Describe everything in this image completely and directly. Extract all text exactly as shown without filtering or censoring. Be thorough and accurate."
        
    def load_image(self, image_path):
        """Load and optimize image for processing"""
        if not os.path.exists(image_path):
            return False, f"❌ Image file not found: {image_path}"
        
        try:
            img = Image.open(image_path)
            
            # Optimize image for processing (resize if very large, but maintain quality)
            if img.size[0] > 2048 or img.size[1] > 2048:
                # Maintain aspect ratio while optimizing
                img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                print(f"  📐 Image resized to {img.size[0]}x{img.size[1]} for optimal processing")
            
            self.current_image = image_path
            return True, f"✅ Image loaded: {os.path.basename(image_path)} | Size: {img.size[0]}x{img.size[1]} pixels"
        except Exception as e:
            return False, f"❌ Error loading image: {e}"
    
    def load_multiple_images(self, image_paths):
        """Load multiple images for batch processing"""
        loaded_images = []
        failed_images = []
        
        for path in image_paths:
            if os.path.exists(path):
                try:
                    img = Image.open(path)
                    loaded_images.append(path)
                except Exception as e:
                    failed_images.append((path, str(e)))
            else:
                failed_images.append((path, "File not found"))
        
        if failed_images:
            print(f"⚠️ Failed to load {len(failed_images)} images:")
            for path, error in failed_images:
                print(f"  {os.path.basename(path)}: {error}")
        
        print(f"✅ Successfully loaded {len(loaded_images)} images for parallel processing")
        return loaded_images
    
    def preprocess_image(self, image_path):
        """Preprocess image for optimal model performance"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Apply slight enhancement for better text recognition
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.1)  # Slight contrast boost
                
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.05)  # Slight sharpness boost
                
                return img
        except Exception as e:
            print(f"  ⚠️ Image preprocessing warning: {e}")
            return Image.open(image_path)
    
    def chat(self, user_message):
        """Enhanced chat with maximum accuracy and efficiency"""
        if not self.current_image:
            return "❌ No image loaded. Use 'load <path>' to load an image first."
        
        # Check if this is a position-related query
        is_position_query = detect_position_query(user_message)
        
        # Always use maximum quality settings for best accuracy
        max_settings = {
            "base_size": 1280,
            "image_size": 1280,
            "name": "Maximum Quality (1280x1280)"
        }
        
        # Enhanced prompt for position queries
        if is_position_query:
            full_prompt = f"""<image>
Analyze this image and describe:
- All visible text and their positions (top, middle, bottom, left, right, center)
- Layout and spatial arrangement of elements
- Exact text content without filtering or censoring
- Detailed position information

{user_message}"""
        else:
            # Direct, uncensored prompt for regular queries
            full_prompt = f"<image>\nDescribe everything you see in this image in complete detail. Extract all text exactly as shown, without filtering or censoring any content. Be direct and thorough.\n\n{user_message}"
        
        try:
            # Optimize threading for CPU
            cpu_count = multiprocessing.cpu_count()
            physical_cores = cpu_count // 2
            optimal_threads = min(physical_cores * 2, cpu_count, 16)
            
            torch.set_num_threads(optimal_threads)
            
            # Preprocess image for better results
            processed_img = self.preprocess_image(self.current_image)
            
            # Memory management before inference
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Capture output more reliably
            import io
            import sys
            
            output_dir = './output'
            os.makedirs(output_dir, exist_ok=True)
            
            # Redirect stdout to capture print statements
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            try:
                # Enhanced inference with better error handling
                result = self.model.infer(
                    self.tokenizer,
                    prompt=full_prompt,
                    image_file=self.current_image,
                    output_path=output_dir,
                    base_size=max_settings["base_size"],
                    image_size=max_settings["image_size"],
                    crop_mode=False,
                    save_results=True,
                    test_compress=False
                )
            finally:
                # Restore stdout
                sys.stdout = old_stdout
                captured_text = captured_output.getvalue()
            
            # Enhanced result extraction with multiple fallback methods
            response_text = self.extract_response(result, captured_text, output_dir)
            
            if not response_text:
                return "⚠️ Model returned empty response. Try with a simpler query or restart the application."
            
            cleaned_res = clean_output(response_text)
            
            # Apply spatial formatting for position queries
            if is_position_query:
                spatial_output = create_spatial_text_map(cleaned_res)
                final_response = f"{cleaned_res}\n\n{spatial_output}"
            else:
                final_response = cleaned_res
            
            # Save to history with metadata
            self.conversation_history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "user": user_message,
                "assistant": final_response,
                "quality": max_settings['name'],
                "image_size": processed_img.size if 'processed_img' in locals() else "Unknown",
                "query_type": "position" if is_position_query else "regular"
            })
            
            # Memory cleanup after inference
            gc.collect()
            
            return final_response
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def chat_parallel(self, user_message, image_paths=None):
        """Process multiple images in parallel with the same query"""
        if image_paths is None:
            if not self.current_image:
                return "❌ No image loaded. Use 'load <path>' to load an image first."
            image_paths = [self.current_image]
        
        # Use parallel processor
        results = self.parallel_processor.process_image_batch(image_paths, [user_message] * len(image_paths))
        
        # Save to history
        self.conversation_history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "user": user_message,
            "assistant": "BATCH ANALYSIS:\n" + "\n\n".join(results),
            "quality": "Parallel Processing",
            "image_count": len(image_paths)
        })
        
        return "PARALLEL ANALYSIS RESULTS:\n" + "\n\n" + "="*60 + "\n".join([f"Image {i+1}: {os.path.basename(path)}\n{result}" for i, (path, result) in enumerate(zip(image_paths, results))])
    
    def extract_response(self, result, captured_text, output_dir):
        """Enhanced response extraction - UNCENSORED, minimal filtering"""
        response_text = None
        
        # Method 1: Check direct result
        if result is not None:
            if isinstance(result, dict):
                response_text = result.get('text', result.get('output', result.get('response', result.get('generated_text'))))
            elif isinstance(result, str) and len(result.strip()) > 10:
                response_text = result
            elif hasattr(result, '__str__') and len(str(result).strip()) > 10:
                response_text = str(result)
        
        # Method 2: Parse captured stdout - ONLY filter technical PyTorch messages
        if not response_text and captured_text and len(captured_text.strip()) > 10:
            lines = [line.strip() for line in captured_text.split('\n') if line.strip()]
            
            # MINIMAL filtering - ONLY remove PyTorch/system technical messages
            technical_prefixes = (
                'The attention', 'Setting', 'The `seen', 'UserWarning:', 'FutureWarning:',
                'BASE:', 'NO PATCHES', 'torch.Size', 'warnings.warn', 'DeprecationWarning',
                'You are using', 'Some weights', 'You should probably TRAIN', 'get_max_cache',
                'The attention layers', 'The `seen_tokens`', 'The attention mask',
                'Setting `pad_token_id`', 'User provided device_type'
            )
            
            # Filter ONLY technical PyTorch messages, keep EVERYTHING else
            content_lines = [
                line for line in lines 
                if not any(line.startswith(prefix) for prefix in technical_prefixes)
                and len(line) > 1  # Only exclude empty/single-char lines
            ]
            
            # Use ALL filtered content - no length requirements, no content-based filtering
            if content_lines:
                response_text = '\n'.join(content_lines)
        
        # Method 3: Check saved files - UNCENSORED
        if not response_text:
            try:
                file_patterns = ['result_', 'output_', 'generation_']
                result_files = []
                
                for file in os.listdir(output_dir):
                    if file.endswith('.txt') and any(pattern in file for pattern in file_patterns):
                        result_files.append(file)
                
                if result_files:
                    latest_file = max([os.path.join(output_dir, f) for f in result_files], key=os.path.getmtime)
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        file_content = f.read().strip()
                        if len(file_content) > 10 and 'error' not in file_content.lower():
                            # Minimal filtering on file content too
                            lines = [line.strip() for line in file_content.split('\n') if line.strip()]
                            cleaned_lines = [
                                line for line in lines 
                                if not any(line.startswith(prefix) for prefix in technical_prefixes)
                                and len(line) > 1
                            ]
                            if cleaned_lines:
                                response_text = '\n'.join(cleaned_lines)
                            else:
                                response_text = file_content
            except Exception:
                pass
        
        # Method 4: Search for any text files in output directory - UNCENSORED
        if not response_text:
            try:
                txt_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
                for txt_file in sorted(txt_files, key=lambda f: os.path.getmtime(os.path.join(output_dir, f)), reverse=True):
                    file_path = os.path.join(output_dir, txt_file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            # Accept ANY content that's not purely technical errors
                            if (len(content) > 10 and 
                                'BASE:' not in content and 'torch.Size' not in content):
                                response_text = content
                                break
                    except Exception:
                        continue
            except Exception:
                pass
        
        # Fallback: Try to extract meaningful text from any source
        if not response_text:
            # Look for any text that doesn't look like technical output
            all_text = captured_text if captured_text else str(result) if result else ""
            lines = [line.strip() for line in all_text.split('\n') if line.strip()]
            
            # Find lines that look like actual text analysis (longer, meaningful content)
            text_lines = []
            for line in lines:
                # Skip technical lines
                if (len(line) > 10 and 
                    not any(line.startswith(prefix) for prefix in technical_prefixes) and
                    not line.startswith('💾') and not line.startswith('CPU:') and
                    'torch.Size' not in line and 'BASE:' not in line and
                    not line.strip().isdigit()):
                    text_lines.append(line)
            
            if text_lines:
                response_text = '\n'.join(text_lines[:10])  # Take first 10 meaningful lines
        
        return response_text

def print_banner():
    """Display enhanced welcome banner"""
    print("\n" + "="*80)
    print("  🔍 DeepSeek OCR - PARALLEL PROCESSING CHAT INTERFACE 💬")
    print("="*80)
    print("  ⚡ OPTIMIZED FOR MAXIMUM ACCURACY & PARALLEL PERFORMANCE")
    print("  🚀 Multi-threaded processing with intelligent batch handling")
    print("="*80)

def print_help():
    """Display enhanced help information"""
    print("\n" + "="*80)
    print("COMMANDS:")
    print("="*80)
    print("  Just type naturally to ask about the image!")
    print()
    print("  load <path>        Load a new image")
    print("  batch <paths...>   Load multiple images for parallel processing")
    print("  parallel <query>   Process all loaded images with same query")
    print("  compare <query>    Compare analysis across multiple images")
    print("  history            View conversation history")
    print("  save               Save conversation to file")
    print("  clear              Clear chat history")
    print("  info               Show current image info")
    print("  help               Show this help")
    print("  quit               Exit")
    print("="*80)
    print("\n🚀 PARALLEL PROCESSING FEATURES:")
    print("  • Process multiple images simultaneously")
    print("  • Intelligent thread pool management")
    print("  • Batch processing with progress tracking")
    print("  • Enhanced memory utilization (18-22GB RAM)")
    print("  • Optimized for your 8-core/16-thread Ryzen 7 5700G")
    print("="*80)
    print("\n📝 CONFIGURATION:")
    print("  • Set default image path in code (see comments in main() function)")
    print("  • Configure multiple images for automatic batch loading")
    print("  • Images load automatically on program startup")
    print("="*80)

# System monitoring functions removed for clean interface

def find_images_in_directory(directory):
    """Find all images in a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'}
    found_images = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                found_images.append(os.path.join(root, file))
    
    return found_images

def main():
    # Load model with enhanced settings
    model_name = 'deepseek-ai/DeepSeek-OCR'
    print_banner()
    
    print("\n⏳ Loading model (optimized for parallel processing)...")
    print("  → Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    print("  → Loading model (CPU-optimized with parallel capabilities)...")
    
    # Enhanced model loading with memory optimization
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.float16,  # Use half precision for memory efficiency
        device_map='cpu',
        low_cpu_mem_usage=True  # Reduce initial memory usage
    )
    
    # Convert to float32 for inference (better accuracy)
    model = model.float()
    model = model.eval()
    
    # Patch device property for compatibility
    cpu_device = torch.device('cpu')
    original_device = type(model).device
    
    @property
    def cpu_device_property(self):
        return cpu_device
    
    type(model).device = cpu_device_property
    
    print("  ✅ Model loaded successfully!")
    print("  🚀 Parallel processing engine ready")
    print("")
    
    # Initialize enhanced chat session
    session = OptimizedChatSession(model, tokenizer)
    
    # =================== IMAGE PATH CONFIGURATION ===================
    # 📝 CONFIGURE YOUR DEFAULT IMAGES HERE:
    # 
    # Option 1: Single image (uncomment and modify the path below)
    default_image = "/workspace/user_input_files/image.png"  # <-- CHANGE THIS PATH
    # 
    # Option 2: Multiple images for batch processing (uncomment the section below)
    # default_images = [
    #     r"C:\Users\YourName\Documents\image1.jpg",
    #     r"C:\Users\YourName\Documents\image2.jpg", 
    #     r"C:\Users\YourName\Documents\image3.jpg"
    # ]
    # 
    # 📋 After configuring, the program will automatically load your images on startup
    # ================================================================
    
    # Load default image(s)
    if 'default_images' in locals() and default_images:
        loaded_images = session.load_multiple_images(default_images)
        if loaded_images:
            print(f"✅ Loaded {len(loaded_images)} default images for parallel processing:")
            for img_path in loaded_images:
                print(f"  📁 {os.path.basename(img_path)}")
    elif os.path.exists(default_image):
        success, msg = session.load_image(default_image)
        print(msg)
    else:
        print("💡 Use 'load <path>' to load an image or 'batch <paths...>' for multiple images")
        print(f"💡 Set your default image path in the code: default_image = \"{default_image}\"")
    
    print("\n💬 Start chatting! Ask anything about your image(s).")
    print("🔍 System automatically uses maximum accuracy settings")
    print("🚀 Use 'parallel' command for batch processing multiple images")
    print(f"📁 Default image: {os.path.basename(default_image)} (change in code if needed)\n")
    
    # Main chat loop
    while True:
        try:
            print("─" * 80)
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Commands
            cmd = user_input.lower()
            
            if cmd in ['quit', 'exit', 'q', 'bye']:
                print("\n👋 Goodbye! Your conversation was saved to memory.")
                break
            
            elif cmd == 'help' or cmd == '?':
                print_help()
                continue
            
            elif cmd.startswith('load '):
                path = user_input[5:].strip().strip('"').strip("'")
                success, msg = session.load_image(path)
                print(f"\n{msg}")
                continue
            
            elif cmd.startswith('batch '):
                # Parse multiple image paths
                paths_part = user_input[6:].strip()
                if paths_part.startswith('[') and paths_part.endswith(']'):
                    # JSON array format
                    try:
                        paths = json.loads(paths_part)
                    except json.JSONDecodeError:
                        paths = paths_part.split()
                else:
                    # Space-separated format
                    paths = paths_part.split()
                
                # Expand directory wildcards
                expanded_paths = []
                for path in paths:
                    if os.path.isdir(path):
                        found = find_images_in_directory(path)
                        expanded_paths.extend(found)
                    else:
                        expanded_paths.append(path)
                
                loaded_images = session.load_multiple_images(expanded_paths)
                if loaded_images:
                    print(f"\n📁 Loaded {len(loaded_images)} images for parallel processing")
                continue
            
            elif cmd.startswith('parallel '):
                query = user_input[9:].strip()
                # Get all loaded images (from batch processing or multiple loads)
                if hasattr(session, 'loaded_images') and session.loaded_images:
                    image_paths = session.loaded_images
                else:
                    # Use current image if only one is loaded
                    if session.current_image:
                        image_paths = [session.current_image]
                    else:
                        print("\n❌ No images loaded for parallel processing. Use 'batch' command first.")
                        continue
                
                print(f"\n🚀 Processing {len(image_paths)} images in parallel...")
                results = session.parallel_processor.process_image_batch(image_paths, [query] * len(image_paths))
                
                print(f"\n🤖 Parallel Analysis Results:")
                print("=" * 80)
                for i, (path, result) in enumerate(zip(image_paths, results)):
                    print(f"\n📸 Image {i+1}: {os.path.basename(path)}")
                    print("-" * 40)
                    print(result)
                    print("\n" + "=" * 80)
                continue
            
            elif cmd.startswith('compare '):
                query = user_input[8:].strip()
                # Similar to parallel but for comparison
                if hasattr(session, 'loaded_images') and session.loaded_images:
                    image_paths = session.loaded_images
                    print(f"\n⚖️ Comparing {len(image_paths)} images with query: {query}")
                    results = session.parallel_processor.process_image_batch(image_paths, [query] * len(image_paths))
                    
                    print(f"\n🔍 Comparison Analysis:")
                    print("=" * 80)
                    for i, (path, result) in enumerate(zip(image_paths, results)):
                        print(f"\n📊 {os.path.basename(path)}")
                        print("-" * 40)
                        print(result)
                else:
                    print("\n❌ No images loaded for comparison. Use 'batch' command first.")
                continue
            
            elif cmd == 'info':
                if session.current_image:
                    img = Image.open(session.current_image)
                    print(f"\n📷 Current Image:")
                    print(f"  Path: {session.current_image}")
                    print(f"  Size: {img.size[0]}x{img.size[1]} pixels")
                    print(f"  Format: {img.format}")
                    print(f"  Mode: {img.mode}")
                    print(f"\n💬 Messages in conversation: {len(session.conversation_history)}")
                    
                    # Show loaded images count if batch processing
                    if hasattr(session, 'loaded_images') and session.loaded_images:
                        print(f"\n📁 Batch images loaded: {len(session.loaded_images)}")
                else:
                    print("\n❌ No image loaded")
                continue
            
            elif cmd == 'history' or cmd == 'h':
                if not session.conversation_history:
                    print("\n💭 No messages yet. Start chatting!")
                else:
                    print(f"\n📜 Conversation ({len(session.conversation_history)} messages):")
                    print("="*80)
                    for i, msg in enumerate(session.conversation_history, 1):
                        print(f"\n[{msg['timestamp']}] #{i}")
                        print(f"You: {msg['user']}")
                        preview = msg['assistant'][:200]
                        if len(msg['assistant']) > 200:
                            preview += "..."
                        print(f"Assistant: {preview}")
                continue
            
            elif cmd == 'save':
                if not session.conversation_history:
                    print("\n💭 No conversation to save yet.")
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"parallel_chat_{timestamp}.txt"
                    os.makedirs('./output', exist_ok=True)
                    
                    with open(f"./output/{filename}", 'w', encoding='utf-8') as f:
                        f.write(f"Parallel Processing Chat Analysis\n")
                        f.write(f"Image: {session.current_image}\n")
                        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"System: AMD Ryzen 7 5700G | 24GB RAM | Parallel Processing\n")
                        f.write("="*80 + "\n\n")
                        
                        for msg in session.conversation_history:
                            f.write(f"[{msg['timestamp']}] Quality: {msg['quality']}\n")
                            f.write(f"You: {msg['user']}\n\n")
                            f.write(f"Assistant:\n{msg['assistant']}\n\n")
                            f.write("-"*80 + "\n\n")
                    
                    print(f"\n✅ Saved to: ./output/{filename}")
                continue
            
            elif cmd == 'clear':
                session.conversation_history = []
                if hasattr(session, 'loaded_images'):
                    session.loaded_images = []
                print("\n✅ Chat history and batch images cleared")
                continue
            
            # System monitoring removed for clean interface
            
            # Direct chat - always maximum quality
            print(f"\n🧠 Analyzing with maximum accuracy...")
            response = session.chat(user_input)
            
            print(f"\n🤖 Enhanced Analysis:\n{response}")
            
            # Enhanced statistics
            words = len(response.split())
            chars = len(response)
            print(f"\n📊 Analysis Stats: {words} words, {chars} characters")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Type 'quit' to exit or press Enter to continue...")
            try:
                cont = input().strip().lower()
                if cont in ['quit', 'exit', 'q']:
                    break
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("💡 Try restarting the application or use 'load' to load a new image")
    
    # Cleanup
    try:
        type(model).device = original_device
    except:
        pass
    
    print("\n🔧 Cleaning up resources...")
    gc.collect()

if __name__ == "__main__":
    main()
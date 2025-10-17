#!/usr/bin/env python3
"""
Startup script for OCR Translator Backend
Tự động kiểm tra dependencies và khởi động server
"""

import sys
import subprocess
import os
import platform
import torch
from pathlib import Path

def check_python_version():
    """Kiểm tra Python version"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]}")
    return True

def check_cuda():
    """Kiểm tra CUDA availability"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"✅ CUDA available")
            print(f"   GPU: {gpu_name}")
            print(f"   VRAM: {gpu_memory:.1f}GB")
            
            if gpu_memory < 3.0:
                print("⚠️  Warning: Limited VRAM detected (<3GB)")
                print("   Will use optimized settings")
                
        else:
            print("⚠️  CUDA not available - will use CPU mode")
            print("   Performance will be significantly slower")
            
        return cuda_available
        
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_dependencies():
    """Kiểm tra required packages"""
    package_map = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'torch': 'torch',
        'transformers': 'transformers',
        'paddleocr': 'paddleocr',
        'paddlepaddle-gpu': 'paddle',   # khác tên
        'opencv-python': 'cv2',         # khác tên
        'Pillow': 'PIL',                # khác tên
        'numpy': 'numpy'
    }
    
    missing = []
    
    for package, module in package_map.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}")
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def check_models():
    """Kiểm tra model files"""
    print("\n📦 Checking translation models...")
    
    try:
        from transformers import M2M100Tokenizer
        
        model_name = "facebook/m2m100_418M"
        print(f"   Loading {model_name}...")
        
        # This will download if not cached
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        print(f"✅ Model ready: {model_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("   Models will be downloaded on first run")
        return False

def create_directories():
    """Tạo các thư mục cần thiết"""
    dirs = ['logs', 'cache', 'debug_images']
    
    for dirname in dirs:
        Path(dirname).mkdir(exist_ok=True)
        
    print("✅ Directories created")

def get_system_info():
    """Hiển thị system info"""
    print("\n💻 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   CPU cores: {os.cpu_count()}")
    
    # Memory info (approximate)
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"   RAM: {memory.total / 1e9:.1f}GB ({memory.percent}% used)")
    except ImportError:
        print("   RAM: Unable to detect (install psutil for details)")

def optimize_for_rtx3050():
    """Tạo config file tối ưu cho RTX3050"""
    config_content = '''# Optimized config for RTX3050 + i5-11400H + 16GB RAM
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HARDWARE_CONFIG = {
    "max_workers": 2,
    "batch_size": 4,
    "cache_size": 400,  # Reduced for RTX3050
    "model_precision": "fp16",  # Half precision
}

# Use smaller model for RTX3050
CURRENT_MODEL = {
    "name": "facebook/m2m100_418M",
    "type": "m2m100",
}

# OCR optimized for RTX3050
OCR_CONFIG = {
    "use_gpu": True,
    "gpu_mem": 1500,  # Limit GPU memory for OCR
    "use_textline_orientation": True,
    "lang": "en",
    "show_log": False,
    "use_space_char": True,
    "det_db_thresh": 0.3,
    "det_db_box_thresh": 0.6,
    "det_db_unclip_ratio": 1.5,
    "rec_batch_num": 4,  # Reduced batch size
    "max_text_length": 25,
    "drop_score": 0.5,
}

TRANSLATION_CONFIG = {
    "max_length": 512,
    "num_beams": 2,
    "early_stopping": True,
    "do_sample": False,
    "source_lang": "en",
    "target_lang": "vi",
    "paragraph_distance": 50,
}

API_CONFIG = {
    "host": "127.0.0.1",
    "port": 8000,
    "reload": False,
    "workers": 1,
}
'''
    
    if not os.path.exists('config.py'):
        with open('config.py', 'w') as f:
            f.write(config_content)
        print("✅ Created optimized config.py for RTX3050")
    else:
        print("✅ Config file exists")

def start_server():
    """Khởi động server"""
    print("\n🚀 Starting OCR Translator Server...")
    print("="*50)
    
    try:
        # Sử dụng main_optimized.py nếu có, ngược lại dùng main.py
        if os.path.exists('main_optimized.py'):
            cmd = [sys.executable, 'main_optimized.py']
            print("Using optimized main file")
        else:
            cmd = [sys.executable, 'main.py']
            print("Using standard main file")
            
        # Run server
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server failed to start: {e}")
        return False
    except FileNotFoundError:
        print("❌ Main file not found!")
        return False
    
    return True

def main():
    """Main startup sequence"""
    print("🔧 OCR Translator - Startup Check")
    print("="*50)
    
    # System checks
    if not check_python_version():
        sys.exit(1)
    
    get_system_info()
    
    cuda_available = check_cuda()
    
    print("\n📋 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Setup
    print("\n⚙️  Setting up...")
    create_directories()
    
    if cuda_available:
        optimize_for_rtx3050()
    
    # Model check (optional)
    check_models()
    
    # Performance tips
    if cuda_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory <= 4.5:  # RTX3050 range
            print("\n💡 RTX3050 detected - Performance tips:")
            print("   • Close other GPU applications")
            print("   • Use smaller batch sizes if OOM occurs")
            print("   • Monitor GPU memory usage")
    else:
        print("\n💡 CPU mode - Performance tips:")
        print("   • Close unnecessary applications")
        print("   • Consider using smaller images")
        print("   • Translation will be slower but functional")
    
    print("\n✅ All checks passed!")
    
    # Start server
    input("\nPress Enter to start the server (or Ctrl+C to exit)...")
    start_server()

if __name__ == "__main__":
    main()
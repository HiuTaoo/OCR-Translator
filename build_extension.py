#!/usr/bin/env python3
"""
Build script for OCR Translator Chrome Extension
Tự động tạo icons và package extension
"""

import os
import json
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import shutil

def create_icon(size, output_path, text="OCR", bg_color=(0, 124, 186), text_color="white"):
    """Tạo icon đơn giản"""
    # Tạo image với background gradient
    img = Image.new('RGBA', (size, size), bg_color + (255,))
    draw = ImageDraw.Draw(img)
    
    # Vẽ gradient effect (đơn giản)
    for i in range(size):
        alpha = int(255 * (1 - i / size * 0.3))
        color = bg_color + (alpha,)
        draw.line([(0, i), (size, i)], fill=color)
    
    # Vẽ text
    if size >= 32:
        try:
            # Thử dùng font hệ thống
            if os.name == 'nt':  # Windows
                font_path = "C:/Windows/Fonts/arial.ttf"
            else:  # macOS/Linux
                font_path = "/System/Library/Fonts/Arial.ttf"
            
            if os.path.exists(font_path):
                font_size = max(size // 4, 8)
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Center text
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (size - text_width) // 2
        y = (size - text_height) // 2
        
        # Vẽ shadow
        draw.text((x + 1, y + 1), text, fill=(0, 0, 0, 128), font=font)
        # Vẽ text chính
        draw.text((x, y), text, fill=text_color, font=font)
    
    # Vẽ border
    draw.rectangle([0, 0, size-1, size-1], outline=(255, 255, 255, 100), width=1)
    
    # Save
    img.save(output_path, 'PNG')
    print(f"Created {output_path} ({size}x{size})")

def create_all_icons(icons_dir):
    """Tạo tất cả các icons cần thiết"""
    icons_dir = Path(icons_dir)
    icons_dir.mkdir(exist_ok=True)
    
    sizes = [16, 48, 128]
    
    for size in sizes:
        icon_path = icons_dir / f"icon{size}.png"
        create_icon(size, icon_path)
    
    print(f"✅ Created {len(sizes)} icons in {icons_dir}")

def validate_manifest(manifest_path):
    """Validate manifest.json"""
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        required_fields = ['manifest_version', 'name', 'version']
        for field in required_fields:
            if field not in manifest:
                print(f"❌ Missing required field: {field}")
                return False
        
        if manifest['manifest_version'] != 3:
            print("❌ Manifest version must be 3 for Chrome Extensions MV3")
            return False
        
        print("✅ Manifest validation passed")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in manifest: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ Manifest file not found: {manifest_path}")
        return False

def update_manifest_permissions(manifest_path):
    """Cập nhật permissions cho localhost"""
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        # Ensure required permissions
        required_permissions = ['activeTab', 'scripting', 'storage']
        if 'permissions' not in manifest:
            manifest['permissions'] = []
        
        for perm in required_permissions:
            if perm not in manifest['permissions']:
                manifest['permissions'].append(perm)
        
        # Ensure host permissions for localhost
        required_hosts = ['http://127.0.0.1:8000/*', 'http://localhost:8000/*']
        if 'host_permissions' not in manifest:
            manifest['host_permissions'] = []
        
        for host in required_hosts:
            if host not in manifest['host_permissions']:
                manifest['host_permissions'].append(host)
        
        # Save updated manifest
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        print("✅ Updated manifest permissions")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update manifest: {e}")
        return False

def minify_js_basic(js_content):
    """Basic JS minification (remove comments and extra spaces)"""
    lines = js_content.split('\n')
    minified_lines = []
    
    for line in lines:
        # Remove single-line comments
        if '//' in line and not line.strip().startswith('//'):
            line = line.split('//')[0]
        
        # Skip empty lines and comment-only lines
        stripped = line.strip()
        if stripped and not stripped.startswith('//') and not stripped.startswith('/*'):
            minified_lines.append(stripped)
    
    return ' '.join(minified_lines)

def build_extension(source_dir, output_dir, minify=False):
    """Build extension package"""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return False
    
    output_dir.mkdir(exist_ok=True)
    build_dir = output_dir / 'build'
    
    # Clean build directory
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()
    
    print(f"📦 Building extension from {source_dir} to {build_dir}")
    
    # Files to copy
    extension_files = [
        'manifest.json',
        'popup.html',
        'content.js',
        'content.css',
        'background.js',
        'popup.js',
    ]
    
    # Copy files
    copied_files = []
    for filename in extension_files:
        src_file = source_dir / filename
        dst_file = build_dir / filename
        
        if src_file.exists():
            if filename.endswith('.js') and minify:
                # Basic minification for JS files
                with open(src_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                minified_content = minify_js_basic(content)
                
                with open(dst_file, 'w', encoding='utf-8') as f:
                    f.write(minified_content)
                
                print(f"✅ Minified and copied {filename}")
            else:
                shutil.copy2(src_file, dst_file)
                print(f"✅ Copied {filename}")
            
            copied_files.append(filename)
        else:
            print(f"⚠️  File not found: {filename}")
    
    # Copy icons directory
    icons_src = source_dir / 'icons'
    icons_dst = build_dir / 'icons'
    
    if icons_src.exists():
        shutil.copytree(icons_src, icons_dst)
        print("✅ Copied icons directory")
    else:
        print("⚠️  Icons directory not found, creating default icons")
        create_all_icons(icons_dst)
    
    # Validate manifest in build directory
    manifest_path = build_dir / 'manifest.json'
    if not validate_manifest(manifest_path):
        return False
    
    # Update manifest for localhost
    update_manifest_permissions(manifest_path)
    
    print(f"✅ Extension built successfully in {build_dir}")
    print(f"📁 Files: {', '.join(copied_files)}")
    
    return build_dir

def create_zip_package(build_dir, output_dir, version=None):
    """Tạo ZIP package cho Chrome Web Store"""
    build_dir = Path(build_dir)
    output_dir = Path(output_dir)
    
    # Get version from manifest
    manifest_path = build_dir / 'manifest.json'
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        version = version or manifest.get('version', '1.0.0')
    except:
        version = version or '1.0.0'
    
    # Create ZIP file
    zip_name = f"ocr-translator-v{version}.zip"
    zip_path = output_dir / zip_name
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in build_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(build_dir)
                zipf.write(file_path, arcname)
                
    file_size = zip_path.stat().st_size / 1024  # KB
    print(f"✅ Created ZIP package: {zip_name} ({file_size:.1f}KB)")
    
    return zip_path

def main():
    """Main build process"""
    print("🔧 OCR Translator Extension Builder")
    print("="*50)
    
    # Paths
    current_dir = Path.cwd()
    extension_dir = current_dir / 'extension'
    output_dir = current_dir / 'dist'
    
    # Check if extension directory exists
    if not extension_dir.exists():
        print(f"❌ Extension directory not found: {extension_dir}")
        print("Please make sure you have the extension files in the 'extension' directory")
        return
    
    # Create icons if missing
    icons_dir = extension_dir / 'icons'
    if not icons_dir.exists() or not list(icons_dir.glob('*.png')):
        print("📸 Creating default icons...")
        create_all_icons(icons_dir)
    
    # Build extension
    print("\n📦 Building extension...")
    build_dir = build_extension(extension_dir, output_dir, minify=True)
    
    if not build_dir:
        print("❌ Build failed!")
        return
    
    # Create ZIP package
    print("\n📦 Creating ZIP package...")
    zip_path = create_zip_package(build_dir, output_dir)
    
    print("\n✅ Build completed successfully!")
    print(f"📁 Build directory: {build_dir}")
    print(f"📦 ZIP package: {zip_path}")
    
    print("\n🚀 Next steps:")
    print("1. Load the extension in Chrome:")
    print(f"   - Open chrome://extensions/")
    print(f"   - Enable 'Developer mode'")
    print(f"   - Click 'Load unpacked' and select: {build_dir}")
    print("\n2. Or upload to Chrome Web Store:")
    print(f"   - Use the ZIP file: {zip_path}")
    
    print("\n💡 Testing tips:")
    print("- Make sure the backend server is running on localhost:8000")
    print("- Test on a webpage with images/text")
    print("- Check browser console for any errors")

if __name__ == "__main__":
    main()
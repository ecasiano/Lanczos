"""Generate the Lanczos ED app icon as a .icns file for macOS."""
import subprocess
import tempfile
import os

def create_icon():
    """Create a 1024x1024 icon and convert to .icns via iconutil."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow not installed. Installing...")
        subprocess.check_call(["pip", "install", "Pillow",
                               "--break-system-packages"])
        from PIL import Image, ImageDraw, ImageFont

    size = 1024
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Dark blue-purple gradient background (circle)
    cx, cy, r = size//2, size//2, size//2 - 40
    for ri in range(r, 0, -1):
        frac = ri / r
        red = int(25 + 30 * (1 - frac))
        green = int(25 + 15 * (1 - frac))
        blue = int(80 + 120 * (1 - frac))
        draw.ellipse([cx-ri, cy-ri, cx+ri, cy+ri],
                     fill=(red, green, blue, 255))

    # Draw a stylized "L" with a wavefunction-like curve
    # The "L" shape
    lw = 60
    draw.rectangle([280, 200, 280+lw, 750], fill=(100, 200, 255, 255))
    draw.rectangle([280, 750-lw, 700, 750], fill=(100, 200, 255, 255))

    # Wavefunction squiggle on top
    import math
    points = []
    for i in range(200):
        x = 300 + i * 2.2
        y = 400 + 80 * math.sin(i * 0.08) * math.exp(-((i-100)**2) / 3000)
        points.append((x, y))
    if len(points) > 2:
        draw.line(points, fill=(255, 200, 80, 255), width=8)

    # "ED" text
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 180)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((420, 520), "ED", fill=(255, 255, 255, 220), font=font)

    # Save as PNG first
    icon_dir = os.path.join(os.path.dirname(__file__), 'icons')
    os.makedirs(icon_dir, exist_ok=True)
    png_path = os.path.join(icon_dir, 'icon_1024.png')
    img.save(png_path)
    print(f"Saved {png_path}")

    # Create .iconset directory with required sizes
    iconset = os.path.join(icon_dir, 'LanczosED.iconset')
    os.makedirs(iconset, exist_ok=True)

    sizes = [16, 32, 64, 128, 256, 512, 1024]
    for s in sizes:
        resized = img.resize((s, s), Image.LANCZOS)
        resized.save(os.path.join(iconset, f'icon_{s}x{s}.png'))
        if s <= 512:
            double = img.resize((s*2, s*2), Image.LANCZOS)
            double.save(os.path.join(iconset, f'icon_{s}x{s}@2x.png'))

    # Convert to .icns using iconutil (macOS only)
    icns_path = os.path.join(icon_dir, 'LanczosED.icns')
    try:
        subprocess.run(['iconutil', '-c', 'icns', iconset, '-o', icns_path],
                       check=True)
        print(f"Created {icns_path}")
    except FileNotFoundError:
        print("iconutil not found (not on macOS?). Using PNG fallback.")
        icns_path = png_path

    return icns_path


if __name__ == '__main__':
    create_icon()

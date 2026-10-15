"""
Script to generate modern, crisp PNG icons for the Chrome Extension.
Generates 16x16, 48x48, and 128x128 icons with rounded corners and a sleek checkmark.
"""

import os
from PIL import Image, ImageDraw

def create_icon(size: int, output_path: str):
    # Create RGBA image with transparent background
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Padding and dimensions
    pad = int(size * 0.08)
    rect = [pad, pad, size - pad, size - pad]
    radius = int(size * 0.24)

    # Gradient-like modern dark indigo / violet fill (#4F46E5)
    bg_color = (79, 70, 229, 255)
    draw.rounded_rectangle(rect, radius=radius, fill=bg_color)

    # Draw checkmark in pure white
    # Points scaled based on size
    # Checkmark coords inside the box
    stroke_w = max(2, int(size * 0.10))
    
    # Checkmark coordinates: start, corner, end
    pt1 = (size * 0.28, size * 0.52)
    pt2 = (size * 0.44, size * 0.68)
    pt3 = (size * 0.74, size * 0.36)

    draw.line([pt1, pt2], fill=(255, 255, 255, 255), width=stroke_w)
    draw.line([pt2, pt3], fill=(255, 255, 255, 255), width=stroke_w)
    
    # Smooth joint dots
    r_cap = stroke_w // 2
    for pt in [pt1, pt2, pt3]:
        draw.ellipse([pt[0] - r_cap, pt[1] - r_cap, pt[0] + r_cap, pt[1] + r_cap], fill=(255, 255, 255, 255))

    img.save(output_path, "PNG")
    print(f"Created {output_path} ({size}x{size})")

if __name__ == "__main__":
    icons_dir = os.path.join(os.path.dirname(__file__), "icons")
    os.makedirs(icons_dir, exist_ok=True)
    
    for s in [16, 48, 128]:
        create_icon(s, os.path.join(icons_dir, f"icon{s}.png"))

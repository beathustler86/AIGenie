from PIL import Image
import os

# Create a simple test image
img = Image.new('RGB', (256, 256), color='navy')
test_path = os.path.join(os.getcwd(), 'pillow_test_output.png')
img.save(test_path)

print(f"✅ Pillow test image saved at: {test_path}")
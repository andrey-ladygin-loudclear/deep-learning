import random

from PIL import Image


def generate_image(block_size = 32, x_blocks = 140, y_blocks = 140, output_path = "assets/image.png"):
    images = ['assets/background/177.jpg', 'assets/background/178.jpg', 'assets/background/179.jpg',
              'assets/background/180.jpg', 'assets/background/181.jpg']

    out = Image.new('RGBA', (block_size*x_blocks, block_size*y_blocks))
    img = [Image.open(src) for src in images]

    for i in range(x_blocks):
        for j in range(y_blocks):
            x = i * 32
            y = j * 32
            out.paste(random.choice(img), (x, y))

    out.save(output_path)
    out.show()
import json

from PIL import Image

with open('exportMap.json', 'r') as f:
    read_data = f.read()

data = json.loads(read_data)
arr = []
mw = 0
mh = 0

for item in data:
    if item.get('type') == 'background':
        arr.append(item)
        x, y = item.get('position')
        mw = max(mw, x)
        mh = max(mh, y + 64)

out = Image.new('RGBA', (mw, mh))

for img in arr:
    im = Image.open(img.get('src'))
    x, y = img.get('position')
    out.paste(im, (x, 995-y))


out.save("backgrounds/first.png")
out.show()
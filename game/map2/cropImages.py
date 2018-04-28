from PIL import Image

im = Image.open("03125630010daeacd577b7aae6dc964e.png")
w, h = im.size

dx = 0
dy = 0

for j in range(20):
    for i in range(19):
        if i == 0 or i == 19:
            dx = 0
        else:
            dx = 1 * i

        if j == 0 or j == 20:
            dy = 0
        else:
            dy = 1 * j

        x = i * 32 + dx
        y = j * 32 + dy
        im.crop((x, y, x + 32, y + 32)).save('assets/'+str(j)+"x"+str(i)+".jpg")
from flask import json
from simplejson import JSONDecodeError

from sprites.destroyableObject import destroyableObject


class ButtonsProvider:

    def exportToFile(self, walls):
        data = []

        minx = 0
        miny = 0

        for wall in walls:
            x, y = wall.position
            minx = min(x, minx)
            miny = min(y, miny)


        for wall in walls:
            x, y = wall.position

            x += 16
            y += 16

            data.append({
                'position': (x + abs(minx), y + abs(miny)),
                'scale': wall.scale,
                'type': wall.type,
                'src': wall.src,
            })

        with open('exportMap.json', 'w') as file_:
            file_.write(json.dumps(data))

    def getMap(self):
        try:
            with open('exportMap.json', 'r') as f:
                 read_data = f.read()

            return json.loads(read_data)
        except Exception:
            pass

        return [{
            'src': 'backgrounds/fill.png',
            'position': [0,0],
            'type': 0,
        }]


    def toggleDestoyableObjects(self, walls, layer, collision):
        for wall in walls:
            if isinstance(wall, destroyableObject):
                if wall in layer:
                    layer.remove(wall)
                    if wall in collision.objs: collision.remove_tricky(wall)
                else:
                    layer.add(wall)
                    collision.add(wall)
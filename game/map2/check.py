from ButtonsProvider import ButtonsProvider

provider = ButtonsProvider()
map = provider.getMap()
for block in map:
    print block
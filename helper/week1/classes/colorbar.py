from tqdm import trange
from time import sleep
import colorama
t = trange(100, desc='Bar desc', leave=True)
for i in t:
    c = colorama.Fore.BLUE

    if i > 20: c = colorama.Fore.BLUE
    if i > 30: c = colorama.Fore.RED
    if i > 40: c = colorama.Fore.GREEN
    if i > 60: c = colorama.Fore.YELLOW
    if i > 70: c = colorama.Fore.CYAN
    if i > 80: c = colorama.Fore.MAGENTA
    if i > 90: c = colorama.Fore.GREEN

    t.set_description(c + "Bar desc (file %i)" % i)

    #print(colorama.Fore.GREEN + "\nLearning BPE" + colorama.Fore.RESET)
    t.refresh() # to show immediately the update
    sleep(0.06)
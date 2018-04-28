BLOCK_SIZE = 8000
number = "{0}: ".format(sys.argv[1]) if len(sys.argv) == 2 else ""
word = sys.stdin.readline().rstrip()


for filename in sys.stdin:
    filename = filename.rstrip()
    previous = ""
    try:
        with open(filename, "rb") as fh:
            while True:
                current = fh.read(BLOCK_SIZE)
                if not current:
                    break
                current = current.decode("utf8", "ignore")

                if (word in current or word in previous[-len(word):] + current[:len(word)]):
                    print("{0}{1}".format(number, filename))
                    break
                if len(current) != BLOCK_SIZE:
                    break
                previous = current
    except EnvironmentError as err:
        print("{0}{1}".format(number, err))














def main():
    opts, word, args = parse_options()
    filelist = get_files(args, opts.recurse)
    work_queue = queue.Queue()
    for i in range(opts.count):
        number = "{0}: ".format(i + 1) if opts.debug else ""
    worker = Worker(work_queue, word, number)
    worker.daemon = True
    worker.start()
    for filename in filelist:
        work_queue.put(filename)
    work_queue.join()

class Worker(threading.Thread):
    def __init__(self, work_queue, word, number):
        super().__init__()
        self.work_queue = work_queue
        self.word = word
        self.number = number
        
    def run(self):
        while True:
            try:
                filename = self.work_queue.get()
                self.process(filename)
            finally:
                self.work_queue.task_done()
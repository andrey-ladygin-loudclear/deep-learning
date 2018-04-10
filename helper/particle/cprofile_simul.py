from simul import benchmark
import cProfile


#cProfile.run("benchmark()")

pr = cProfile.Profile()
pr.enable()
benchmark()
pr.disable()
pr.print_stats()
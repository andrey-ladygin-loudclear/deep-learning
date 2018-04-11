from random import uniform

from matplotlib import pyplot as plt
from matplotlib import animation

import timeit

from Particle import Particle
from ParticleSimulator import ParticleSimulator


def test_evolve():
    particles = [Particle( 0.3,  0.5,  +1),
                 Particle( 0.0, -0.5, -1),
                 Particle(-0.1, -0.4, +3)]

    simulator = ParticleSimulator(particles)
    simulator.evolve(0.1)

    p0, p1, p2 = particles

    def fequal(a, b, eps=1e-5):
        return abs(a - b) < eps

    assert fequal(p0.x, 0.210269)
    assert fequal(p0.y, 0.543863)

    assert fequal(p1.x, -0.099334)
    assert fequal(p1.y, -0.490034)

    assert fequal(p2.x, 0.191358)
    assert fequal(p2.y, -0.365227)


def benchmark():
    particles = [Particle(uniform(-1.0, 1.0),
                          uniform(-1.0, 1.0),
                          uniform(-1.0, 1.0))
                 for i in range(1000)]

    simulator = ParticleSimulator(particles)
    simulator.evolve(0.1)
    # to run
    # %load_ext line_profiler
    # from simul import benchmark, ParticlesSimulator
    # %lprun -f ParticlesSimulator.evolve benchmark()
    # kernprof -l -v simul.py


def benchmark_memory():
    particles = [Particle(uniform(-1.0, 1.0),
                          uniform(-1.0, 1.0),
                          uniform(-1.0, 1.0))
                 for i in range(100000)]
    simulator = ParticleSimulator(particles)
    simulator.evolve(0.001)
    # 1 MiB (mebibyte) is equivalent to 1,048,576 bytes. It is different from 1 MB
    # (megabyte), which is equivalent to 1,000,000 bytes.
    # to run
    # %load_ext memory_profiler
    # from simul import benchmark_memory
    # mprun -f benckmark_memory benchmark_memory()
    # mprof run


if __name__ == '__main__':
    benchmark()
    # %timeit benchmark()
    # result = timeit.timeit('benchmark()', setup='from __main__ import benchmark', number=10)
    # python -m cProfile simul.py
    # python -m cProfile -s tottime simul.py
    # python -m cProfile -o prof.out simul.py

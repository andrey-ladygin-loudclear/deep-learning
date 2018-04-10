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



if __name__ == '__main__':
    benchmark()
    #%timeit benchmark()
    #result = timeit.timeit('benchmark()', setup='from __main__ import benchmark', number=10)
    #python -m cProfile simul.py
    #python -m cProfile -s tottime simul.py
    #python -m cProfile -o prof.out simul.py

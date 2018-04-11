#kernprof -l -v FILE.py
@profile
def evolve(self, dt):
    timestep = 0.00001
    nsteps = int(dt/timestep)

    for i in range(nsteps):
        for p in self.particles:
            norm = (p.x**2 + p.y**2) **0.5
            v_x = -p.y/norm
            v_y = p.x/norm

            d_x = timestep * p.ang_vel * v_x
            d_y = timestep * p.ang_vel * v_y

            p.x += d_x
            p.y += d_y
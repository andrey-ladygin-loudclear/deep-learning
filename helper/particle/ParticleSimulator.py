class ParticleSimulator:
    def __init__(self, particles):
        self.particles = particles

    @profile
    def evolve(self, dt):
        timestep = 0.00001
        nsteps = int(dt/timestep)

        # will be better to use sin and cos for performance
        # x = r * cos(alpha)
        # y = r * sin(alpha)

        for i in range(nsteps):
            for p in self.particles:
                norm = (p.x**2 + p.y**2) **0.5

                # p.x, p.y = p.x - t_x_ang*p.y/norm, p.y + t_x_ang * p.x/norm
                v_x = -p.y/norm
                v_y = p.x/norm

                d_x = timestep * p.ang_vel * v_x
                d_y = timestep * p.ang_vel * v_y

                p.x += d_x
                p.y += d_y

    def evolve_fast(self, dt):
        timestep = 0.00001
        nsteps = int(dt/timestep)

        # Loop order is changed
        for p in self.particles:
            t_x_ang = timestep * p.ang_vel
            for i in range(nsteps):
                norm = (p.x**2 + p.y**2)**0.5
                p.x, p.y = (p.x - t_x_ang * p.y/norm,
                            p.y + t_x_ang * p.x/norm)

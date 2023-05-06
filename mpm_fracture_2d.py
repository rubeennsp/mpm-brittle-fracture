import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)
# ti.init(arch=ti.cuda)

quality = 1  # Use a larger value for higher-res simulations
n_particles, n_grid = 9000 * quality**2, 128 * quality
# n_particles = 100
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters

x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=float,
                    shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float,
                    shape=n_particles)  # deformation gradient
material = ti.field(dtype=int, shape=n_particles)  # material id
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
grid_v = ti.Vector.field(2, dtype=float,
                         shape=(n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass
gravity = ti.Vector.field(2, dtype=float, shape=())
attractor_strength = ti.field(dtype=float, shape=())
attractor_pos = ti.Vector.field(2, dtype=float, shape=())

group_size = n_particles // 3
water = ti.Vector.field(2, dtype=float, shape=group_size)  # position
jelly = ti.Vector.field(2, dtype=float, shape=group_size)  # position
snow = ti.Vector.field(2, dtype=float, shape=group_size)  # position
mouse_circle = ti.Vector.field(2, dtype=float, shape=(1, ))


@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:  # Particle state update and scatter to grid (P2G)
        # Base grid coords (integer)
        base = ti.floor(x[p] * inv_dx - 0.5).cast(int)

        # float offset from base grid coords # TODO: Find the range of this fx
        fx = x[p] * inv_dx - base.cast(float)
        # print('Uncasted: ', (x[p] * inv_dx - 0.5), '\tbase = ', base, '\tfx = ', fx) 

        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [
            0.5 * (1.5 - fx)**2, # N(fx  ).    fx     is between 0.5 and 1.5
            0.75 - (fx - 1)**2,  # N(fx-1).    (fx-1) is between -0.5 and 0.5
            0.5 * (fx - 0.5)**2  # N(fx-2).    (fx-2) is between -1.5 and 0.5
        ]

        # deformation gradient update
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p] # Multiply F by (I + dt * C)
        # Hardening coefficient: Jelly
        h = 1
        mu, la = mu_0 * h, lambda_0 * h # TODO: What are these Lame parameters mu and lambda?

        # Singular value decomposition
        U, sig, V = ti.svd(F[p]) # TODO: Is this the right polar svd?
        J = 1.0

        for d in ti.static(range(2)):
            J *= sig[d, d]
        # J is probably equal to determinant. TODO: Test this out

        # Fixed corotated constitutive model seen in SIGGRAPH 2018 MPM Tutorial section 6.3.
        # TODO: Change to Neo-Hookean from section 6.2? Change to other Neo-Hookean energies?
        stress_J = (
            2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() # 2 mu (F-R) times F^T
            + ti.Matrix.identity(float, 2) * la * J * (J - 1) # lambda J (J-1) F^-T times F^T
        )

        # FLOP optimization from the starter code. Refer to Q_p in section 6 equation (29) of the MLS-MPM paper.
        # stress_J = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress_J
        # affine = stress_J + p_mass * C[p] # Q_p
        # # Later, in the gridnode loop, we get
        # # mv_contribution = (p_mass * v[p] + affine @ dpos)
        # # which saves at least one matmul
        # # But we're not using this. For clarity.

        # (unweighted contribution of) internal force * dt
        M_inv = (4 * inv_dx * inv_dx) # 4 / h^2
        fint_dt = - dt * p_vol * M_inv * stress_J # to be multiplied by weight and dpos, later. In other words,
                                                  # internal force * dt = fint_dt @ (xi - xp)

        # Loop over 3x3 grid node neighborhood around particle.
        for i, j in ti.static(ti.ndrange(3, 3)):
            # Compute delta x from particle to grid. (xi - xp)
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx

            # Compute particle's momentum contribution to grid node
            mv_contribution = v[p]              # start with linear velocity
            mv_contribution += C[p] @ dpos      # include affine velocity
            mv_contribution *= p_mass           # momentum = mass * velocity
            mv_contribution += fint_dt @ dpos   # include internal force * dt

            # Accumulate weighted contribution on gridnode.
            weight = w[i][0] * w[j][1] # quadratic kernel weight product over dimensions
            grid_v[base + offset] += weight * mv_contribution
            grid_m[base + offset] += weight * p_mass

    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            grid_v[i,
                   j] = (1 / grid_m[i, j]) * grid_v[i,
                                                    j]  # Momentum to velocity
            grid_v[i, j] += dt * gravity[None] * 30  # gravity
            dist = attractor_pos[None] - dx * ti.Vector([i, j])
            grid_v[i, j] += dist / (
                0.01 + dist.norm()) * attractor_strength[None] * dt * 100
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0  # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j][0] > 0:
                grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0:
                grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0
    for p in x:  # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        # loop over 3x3 grid node neighborhood
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection


@ti.kernel
def reset():
    # i: [0, ..., n-1]
    # group_size = n / 3
    # (i // group_size): [0, 0, ..., 0, 1, 1, ..., 1, 2, 2, ..., 2] == material
    start_pos = ti.Vector([0.3, 0.05])
    for i in range(n_particles):
        x[i] = [
            ti.random() * 0.2 + start_pos[0] + 0.10 * (i // group_size),
            ti.random() * 0.2 + start_pos[1] + 0.32 * (i // group_size)
        ]
        material[i] = i // group_size  # 0: fluid 1: jelly 2: snow
        v[i] = [0, 0]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1
        C[i] = ti.Matrix.zero(float, 2, 2)


@ti.kernel
def render():
    for i in range(group_size):
        water[i] = x[i]
        jelly[i] = x[i + group_size]
        snow[i] = x[i + 2 * group_size]


def main():
    print(
        "[Hint] Use WSAD/arrow keys to control gravity. Use left/right mouse buttons to attract/repel. Press R to reset."
    )

    res = (512, 512)
    window = ti.ui.Window("Taichi MLS-MPM-128", res=res, vsync=True)
    canvas = window.get_canvas()
    radius = 0.003

    reset()
    gravity[None] = [0, -1]

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == 'r':
                reset()
            elif window.event.key in [ti.ui.ESCAPE]:
                break
        if window.event is not None:
            gravity[None] = [0, 0]  # if had any event
        if window.is_pressed(ti.ui.LEFT, 'a'):
            gravity[None][0] = -1
        if window.is_pressed(ti.ui.RIGHT, 'd'):
            gravity[None][0] = 1
        if window.is_pressed(ti.ui.UP, 'w'):
            gravity[None][1] = 1
        if window.is_pressed(ti.ui.DOWN, 's'):
            gravity[None][1] = -1
        mouse = window.get_cursor_pos()
        mouse_circle[0] = ti.Vector([mouse[0], mouse[1]])
        canvas.circles(mouse_circle, color=(0.2, 0.4, 0.6), radius=0.05)
        attractor_pos[None] = [mouse[0], mouse[1]]
        attractor_strength[None] = 0
        if window.is_pressed(ti.ui.LMB):
            attractor_strength[None] = 1
        if window.is_pressed(ti.ui.RMB):
            attractor_strength[None] = -1

        for s in range(int(2e-3 // dt)):
            substep()
            # quit()
        render()
        canvas.set_background_color((0.067, 0.184, 0.255))
        canvas.circles(water, radius=radius, color=(0, 0.5, 0.5))
        canvas.circles(jelly, radius=radius, color=(0.93, 0.33, 0.23))
        canvas.circles(snow, radius=radius, color=(1, 1, 1))
        window.show()


if __name__ == '__main__':
    main()

import taichi as ti
from taichi_glsl.scalar import isnan, isinf
import matplotlib.cm
from math import sqrt
import numpy as np

import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)
# ti.init(arch=ti.cuda)
# ti.init(arch=ti.cuda, debug=True)
# ti.init(arch=ti.cpu, debug=False)

#dim, n_grid, steps, dt = 2, 128, 20, 2e-4
#dim, n_grid, steps, dt = 2, 256, 32, 1e-4
#dim, n_grid, steps, dt = 3, 32, 25, 4e-4
dim, n_grid, steps, dt = 3, 64, 25, 2e-4 / 4
# dim, n_grid, steps, dt = 3, 96, 1, 2e-4 / 10
#dim, n_grid, steps, dt = 3, 128, 5, 1e-4

n_particles = n_grid**dim // 2**(dim - 1)

print(f'n_particles: {n_particles}')

dx = 1 / n_grid

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
GRAVITY = [0, -9.8, 0]
bound = 3
E = 5e4  # Young's modulus
nu = 0.2  #  Poisson's ratio
Gf = 200 # Mode 1 fracture energy
sigma_f = 150 # principal failure stress. See Section 4.2 of paper.
l_ch = dx * sqrt(dim) # Characteristic length: grid-cell diagonal
H_bar = sigma_f ** 2 / (2 * E * Gf)
H = H_bar * l_ch / (1 - H_bar * l_ch)  # Brittleness factor.   See Section 4.2 of paper, under eq (7), for definition.
                                       #                       See Table 3 for actual parameter values.
H = 40 # override "properly-calculated" brittleness factor :( because it didn't work very well.
       # TODO: Investigate these parameter settings. They might be scale dependent.
print(f'H: {H}')
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters

F_x = ti.Vector.field(dim, float, n_particles)
F_v = ti.Vector.field(dim, float, n_particles)
F_damage = ti.field(dtype=float, shape=n_particles)  # material damage (c in equations)
F_C = ti.Matrix.field(dim, dim, float, n_particles)
F_dg = ti.Matrix.field(dim, dim, dtype=float,
                       shape=n_particles)  # deformation gradient
F_Jp = ti.field(float, n_particles)

F_colors = ti.Vector.field(4, float, n_particles)
F_colors_random = ti.Vector.field(4, float, n_particles)
F_materials = ti.field(int, n_particles)
F_grid_v = ti.Vector.field(dim, float, (n_grid, ) * dim)
F_grid_m = ti.field(float, (n_grid, ) * dim)
F_used = ti.field(int, n_particles)
colormap = matplotlib.cm.get_cmap('jet') # Can try other colormaps

neighbour = (3, ) * dim

WATER = 0
JELLY = 1
SNOW = 2

@ti.func
def my_sym_eig_3x3(A):
    Q, sig_mat, Q2 = ti.svd(A)
    eig = ti.Vector.zero(dt=float, n=3)
    for j in ti.static(range(3)):
        mult = 1.0
        if sig_mat[j, j] == 0.0:
            mult = 1.0
        else:
            divresult = (A @ Q[:, j]) / (sig_mat[j, j] * Q[:, j])
            print(f"divresult {j}: {divresult}")
            for i in ti.static(range(3)):
                # Guard for inf or nan
                # or really for anything other than +/- 1
                if ti.abs(ti.abs(divresult[i]) - 1) < 1e-5:
                    mult = divresult[i]

        eig[j] = sig_mat[j, j] * mult

    return eig, Q


@ti.kernel
def substep(g_x: float, g_y: float, g_z: float):
    for I in ti.grouped(F_grid_m):
        F_grid_v[I] = ti.zero(F_grid_v[I])
        F_grid_m[I] = 0
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_used[p] == 0:
            continue
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

        F_dg[p] = (ti.Matrix.identity(float, dim) +
                   dt * F_C[p]) @ F_dg[p]  # deformation gradient update

        # Hardening coefficient (snow gets harder when compressed. We don't need snow in our sim)
        h = 1
        mu, la = mu_0 * h, lambda_0 * h

        U, sig, V = ti.svd(F_dg[p])
        J = ti.Matrix.determinant(sig)

        # Fixed corotated constitutive model seen in SIGGRAPH 2018 MPM Tutorial section 6.3.
        # TODO: Change to Neo-Hookean from section 6.2? Change to other Neo-Hookean energies?
        effective_stress = (
            2 * mu / J * (F_dg[p] - U @ V.transpose()) @ F_dg[p].transpose() # 2 mu (F-R) times F^T
            + ti.Matrix.identity(float, dim) * la * (J - 1) # lambda J (J-1) F^-T times F^T
        )

        # Force the symmetry. It should be pretty close to symmetric, but err on safety.
        effective_stress = (effective_stress + effective_stress.transpose()) / 2.
        assert not isnan(effective_stress[0, 0])
        sigbar, Q = my_sym_eig_3x3(effective_stress) # Q's columns are normalized eigenvalues

        max_effective_stress = ti.max(1e-5, *sigbar) # scalar

        # Update damage from max stress
        unclamped_damage = (1 + H) * (1 - sigma_f / max_effective_stress)
        clamped_damage = ti.math.clamp(unclamped_damage, 0, 1) # Clamp damage between 0 and 1
        c = F_damage[p] = ti.max(F_damage[p], clamped_damage)


        weakened_sigbarmat = ti.Matrix.zero(float, dim, dim)
        sigbarmat = ti.Matrix.zero(float, dim, dim)
        for i in ti.static(range(dim)):
            weakening_multiplier = (1 - c) if sigbar[i] > 0 else 1
            weakened_sigbarmat[i, i] = sigbar[i] * weakening_multiplier
            sigbarmat[i, i] = sigbar[i]
        weakened_stress = Q @ weakened_sigbarmat @ Q.transpose()
        # reconstructed_stress = Q @ sigbarmat @ Q.transpose()

        stress_J = weakened_stress * J

        M_inv = (4 / dx / dx) # 4 / h^2
        fint_dt = - dt * p_vol * M_inv * stress_J # to be multiplied by weight and dpos, later. In other words,
                                                  # internal force * dt = fint_dt @ (xi - xp)

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]

            # Compute particle's momentum contribution to grid node
            mv_contribution = F_v[p]            # start with linear velocity
            mv_contribution += F_C[p] @ dpos    # include affine velocity
            mv_contribution *= p_mass           # momentum = mass * velocity
            mv_contribution += fint_dt @ dpos   # include internal force * dt

            # F_grid_v[base + offset] += weight * (p_mass * F_v[p] + affine @ dpos)
            F_grid_v[base + offset] += weight * mv_contribution
            F_grid_m[base + offset] += weight * p_mass
    for I in ti.grouped(F_grid_m):
        if F_grid_m[I] > 0:
            F_grid_v[I] /= F_grid_m[I]
        F_grid_v[I] += dt * ti.Vector([g_x, g_y, g_z])
        cond = (I < bound) & (F_grid_v[I] < 0) | \
               (I > n_grid - bound) & (F_grid_v[I] > 0)
        F_grid_v[I] = ti.select(cond, 0, F_grid_v[I])
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_used[p] == 0:
            continue
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.zero(F_v[p])
        new_C = ti.zero(F_C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = F_grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        F_v[p] = new_v
        F_x[p] += dt * F_v[p]
        F_C[p] = new_C


class CubeVolume:
    def __init__(self, minimum, size, material):
        self.minimum = minimum
        self.size = size
        self.volume = self.size.x * self.size.y * self.size.z
        self.material = material


@ti.kernel
def init_cube_vol(first_par: int, last_par: int, x_begin: float,
                  y_begin: float, z_begin: float, x_size: float, y_size: float,
                  z_size: float, material: int):
    for i in range(first_par, last_par):
        F_x[i] = ti.Vector([ti.random() for i in range(dim)]) * ti.Vector(
            [x_size, y_size, z_size]) + ti.Vector([x_begin, y_begin, z_begin])
        F_Jp[i] = 1
        F_dg[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_v[i] = ti.Vector([0.0, -2.0, 0.0])
        F_damage[i] = 0
        F_materials[i] = material
        F_colors_random[i] = ti.Vector(
            [ti.random(), ti.random(),
             ti.random(), ti.random()])
        F_used[i] = 1


@ti.kernel
def set_all_unused():
    for p in F_used:
        F_used[p] = 0
        # basically throw them away so they aren't rendered
        F_x[p] = ti.Vector([533799.0, 533799.0, 533799.0])
        F_Jp[p] = 1
        F_dg[p] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_C[p] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        F_v[p] = ti.Vector([0.0, 0.0, 0.0])


def init_vols(vols):
    set_all_unused()
    total_vol = 0
    for v in vols:
        total_vol += v.volume

    next_p = 0
    for i, v in enumerate(vols):
        v = vols[i]
        if isinstance(v, CubeVolume):
            par_count = int(v.volume / total_vol * n_particles)
            if i == len(
                    vols
            ) - 1:  # this is the last volume, so use all remaining particles
                par_count = n_particles - next_p
            init_cube_vol(next_p, next_p + par_count, *v.minimum, *v.size,
                          v.material)
            next_p += par_count
        else:
            raise Exception("???")


@ti.kernel
def set_color_by_material(mat_color: ti.types.ndarray()):
    for i in range(n_particles):
        mat = F_materials[i]
        F_colors[i] = ti.Vector(
            [mat_color[mat, 0], mat_color[mat, 1], mat_color[mat, 2], 1.0])


print("Loading presets...this might take a minute")

presets = [[
    CubeVolume(ti.Vector([0.55, 0.05, 0.55]), ti.Vector([0.4, 0.4, 0.4]),
               WATER),
],
           [
               CubeVolume(ti.Vector([0.05, 0.05, 0.05]),
                          ti.Vector([0.3, 0.4, 0.3]), WATER),
               CubeVolume(ti.Vector([0.65, 0.05, 0.65]),
                          ti.Vector([0.3, 0.4, 0.3]), WATER),
           ],
           [
               CubeVolume(ti.Vector([0.6, 0.05, 0.6]),
                          ti.Vector([0.25, 0.25, 0.25]), WATER),
               CubeVolume(ti.Vector([0.35, 0.35, 0.35]),
                          ti.Vector([0.25, 0.25, 0.25]), SNOW),
               CubeVolume(ti.Vector([0.05, 0.6, 0.05]),
                          ti.Vector([0.25, 0.25, 0.25]), JELLY),
           ]]
preset_names = [
    "Single Dam Break",
    "Double Dam Break",
    "Water Snow Jelly",
]

curr_preset_id = 0

paused = False

use_random_colors = False
particles_radius = 0.002

material_colors = [(0.1, 0.6, 0.9), (0.93, 0.33, 0.23), (1.0, 1.0, 1.0)]


def init():
    global paused
    init_vols(presets[curr_preset_id])


init()

res = (1080, 720)
window = ti.ui.Window("Real MPM 3D", res, vsync=True)

canvas = window.get_canvas()
gui = window.get_gui()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(55)


def show_options():
    global use_random_colors
    global paused
    global particles_radius
    global curr_preset_id

    with gui.sub_window("Presets", 0.05, 0.1, 0.2, 0.15) as w:
        old_preset = curr_preset_id
        for i in range(len(presets)):
            if w.checkbox(preset_names[i], curr_preset_id == i):
                curr_preset_id = i
        if curr_preset_id != old_preset:
            init()
            paused = True

    with gui.sub_window("Gravity", 0.05, 0.3, 0.2, 0.1) as w:
        GRAVITY[0] = w.slider_float("x", GRAVITY[0], -10, 10)
        GRAVITY[1] = w.slider_float("y", GRAVITY[1], -10, 10)
        GRAVITY[2] = w.slider_float("z", GRAVITY[2], -10, 10)

    with gui.sub_window("Options", 0.05, 0.45, 0.2, 0.4) as w:
        use_random_colors = w.checkbox("use_random_colors", use_random_colors)
        if not use_random_colors:
            material_colors[WATER] = w.color_edit_3("water color",
                                                    material_colors[WATER])
            material_colors[SNOW] = w.color_edit_3("snow color",
                                                   material_colors[SNOW])
            material_colors[JELLY] = w.color_edit_3("jelly color",
                                                    material_colors[JELLY])
            set_color_by_material(np.array(material_colors, dtype=np.float32))
        particles_radius = w.slider_float("particles radius ",
                                          particles_radius, 0, 0.005)
        if w.button("restart"):
            init()
        if paused:
            if w.button("Continue"):
                paused = False
        else:
            if w.button("Pause"):
                paused = True

def lerp(a, b, t):
    return a * (1 - t) + b * t

def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))

    damage_numpy = F_damage.to_numpy(float)
    # damage_numpy[damage_numpy < 1] = 0 # Ignore partial damage. Only fully damaged particles are colored.
    alpha = lerp(1, 0.1, damage_numpy)
    color_numpy = colormap(damage_numpy)
    color_numpy[:, 3] = alpha
    F_colors.from_numpy(color_numpy)
    colors_used = F_colors_random if use_random_colors else F_colors
    scene.particles(F_x, per_vertex_color=colors_used, radius=particles_radius)

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)


def main():
    frame_id = 0

    while window.running:
        #print("heyyy ",frame_id)
        frame_id += 1
        frame_id = frame_id % 256

        if not paused:
            for _ in range(steps):
                substep(*GRAVITY)

        render()
        show_options()
        window.show()


if __name__ == '__main__':
    main()

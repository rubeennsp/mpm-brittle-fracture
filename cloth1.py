import taichi as ti
import matplotlib.pyplot as plt
import numpy as np

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

@ti.func
def lerp(a, b, t):
    return a + (b-a) * t


### Simulation parameters and data ###

gravity = ti.Vector([0, -9.8, 0])

n = 100 # cloth resolution: number of nodes each side
quad_size = 1.0 / (n-1)
step_duration = 1/60 # Hopefully it's 60 fps
substeps = 100

spring_types = STRETCH, SHEAR, BENDING = 'STRETCH', 'SHEAR', 'BENDING'

spring_offsets_dict = {
    STRETCH: [
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
    ],
    SHEAR: [
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
    ],
    BENDING: [
        [-2, 0],
        [2, 0],
        [0, -2],
        [0, 2],
    ],
}

initial_spring_constants = {
    STRETCH: 5e5,
    SHEAR: 3e5,
    BENDING: 1e5,
}

spring_constants = {
    spring_type: ti.field(float, shape=())
    for spring_type in spring_types
}
for spring_type in spring_types:
    spring_constants[spring_type][None] = initial_spring_constants[spring_type]

spring_restlengths_dict = {
    STRETCH: quad_size,
    SHEAR: quad_size * ti.sqrt(2),
    BENDING: quad_size * 2,
}


spring_damping_constant = ti.field(float, shape=())
spring_damping_constant[None] = 10

drag_damping = ti.field(float, shape=())
drag_damping[None] = 5

# Ball

ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_velocity = ti.Vector.field(3, dtype=float, shape=(1, ))

ball_radius = 0.1
ball_sticky_slowdown = ti.field(float, shape=())
ball_sticky_slowdown[None] = 0.6
ball_center[0] = [0, -0.5, 0]
ball_velocity[0] = [0, 0, 0]

floor_height = -0.5
floor_sticky_slowdown = ti.field(float, shape=())
floor_sticky_slowdown[None] = 0.003

# Cloth data

x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))


### Visualization parameters and data ###

num_triangles = (n - 1) * (n - 1) * 2

triangle_indices = ti.field(int, shape=num_triangles * 3)
triangle_vertices = ti.Vector.field(3, dtype=float, shape=n * n)
triangle_colors = ti.Vector.field(3, dtype=float, shape=n * n)

bending_springs = True

floor_vertices = ti.Vector.field(3, dtype=float, shape=6)
visual_floor_height = floor_height - 5e-3 # Render the floor slightly lower to avoid visual penetration
for i, floor_vertex in enumerate([
    [-1000, visual_floor_height, -1000],
    [-1000, visual_floor_height, 1000],
    [1000, visual_floor_height, -1000],
    [1000, visual_floor_height, 1000],
    [1000, visual_floor_height, -1000],
    [-1000, visual_floor_height, 1000],
]):
    floor_vertices[i] = floor_vertex

floor_indices = ti.field(int, shape=6)
for i in range(6):
    floor_indices[i] = i

# num_lines = ?
# line_indices = ti.field(int, shape=num_lines * 2)

@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1

    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5 + random_offset[0],
            -0.3,
            j * quad_size - 0.5 + random_offset[1]
        ]
        v[i, j] = [0, 0, 0]


@ti.kernel
def initialize_mesh():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # 1st triangle of the square
        triangle_indices[quad_id * 6 + 0] = i * n + j
        triangle_indices[quad_id * 6 + 1] = (i + 1) * n + j
        triangle_indices[quad_id * 6 + 2] = i * n + (j + 1)
        # 2nd triangle of the square
        triangle_indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        triangle_indices[quad_id * 6 + 4] = i * n + (j + 1)
        triangle_indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        triangle_colors[i * n + j] = (lerp(0.3, 1, i/n), lerp(0.3, 1, j/n), 0.2)

initialize_mesh()


@ti.kernel
def substep(dt: float):
    for i in ti.grouped(x):
        v[i] += gravity * dt

    # Spring forces
    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])

        # For every kind of spring (stretch, shear, bending)
        for spring_type in ti.static(spring_types):
            spring_offsets = spring_offsets_dict[spring_type]
            Ks = spring_constants[spring_type][None]
            restlength = spring_restlengths_dict[spring_type]
            Kd = spring_damping_constant[None]
            # Accumulate the forces on the particle
            for spring_offset in ti.static(spring_offsets):
                j = i + spring_offset
                if 0 <= j[0] < n and 0 <= j[1] < n:
                    x_ij = x[i] - x[j]
                    v_ij = v[i] - v[j]
                    d = x_ij.normalized()
                    current_dist = x_ij.norm()
                    # Spring force
                    force += - d * (                # Force direction
                        Ks * (current_dist - restlength)    # Spring force magnitude
                        + Kd * v_ij.dot(d)          # Spring damping force magnitude
                    )
                    # force = ti.Vector([0, 0, 0])
                    # force += -Ks * d * (current_dist / restlength - 1)
                    # # Dashpot damping
                    # force += -v_ij.dot(d.normalized()) * d.normalized() * spring_damping_constant[None] * quad_size

        v[i] += force * dt

    for i in ti.grouped(x):
        # Damping
        v[i] *= ti.exp(-drag_damping[None] * dt)

        # Contact with ball
        offset_to_center = x[i] - ball_center[0]
        normal = offset_to_center.normalized()
        if offset_to_center.norm() <= ball_radius:
            # Velocity projection
            relative_velocity = v[i] - ball_velocity[0]
            normal_velocity_component_magnitude = relative_velocity.dot(normal)
            if (normal_velocity_component_magnitude < 0):
                normal_velocity_component = normal_velocity_component_magnitude * normal
                projected_relative_velocity = relative_velocity - normal_velocity_component
                projected_relative_velocity *= (ball_sticky_slowdown[None] ** dt) # non-physical friction
                # if projected_relative_velocity.norm() < 1:
                #     projected_relative_velocity = [0, 0, 0]
                v[i] = ball_velocity[0] + projected_relative_velocity
                
            x[i] = ball_center[0] + normal * ball_radius # protrude out to ball. TODO: consider ball velocity

        # Contact with floor
        if x[i][1] <= floor_height:
            x[i][1] = floor_height # Project back out of floor
            if (v[i][1] < 0): # If moving into floor,
                v[i][1] = 0                                  # zero out that Y component (floor-normal component).
                v[i] *= (floor_sticky_slowdown[None] ** dt)  # And slow down. This is a non-physical interpretation of friction.

        x[i] += dt * v[i]


@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        triangle_vertices[i * n + j] = x[i, j]

def update_ball(timestep):
    ball_center[0] += ball_velocity[0] * timestep
    slowdown_factor_per_unit_time = 0.2
    ball_velocity[0] *= slowdown_factor_per_unit_time ** timestep

# screen:
"""
(0, 1)    (1, 1)

(0, 0)    (1, 0)
"""
def control_ball_with_cursor(cursor_pos, proj, view, timestep):
    """
    Control the ball

    Params:
    - `cursor_pos`: Position of cursor where bottom-left is [0, 0] and top-right is [1, 1]
    - `proj`, `view`: The projection and view matrices of the camera. (See glm::perspective and glm::lookat)

    Side effects:
    - Modifies `ball_center`
    - TODO: track ball velocity
    """
    P = view @ proj
    cursor_pos = np.array(cursor_pos)
    cursor_pos = cursor_pos * 2 - 1 # convert cursor pos to NDC
    base_pos = P @ [0, 0, 0, 1] # base position: The ball will be set at the same "depth" as this position.
                                #                There is a depth ambiguity and this fixes a choice.
                                #                Consider setting this base_position as a function argument.
                                #                What makes sense other than the origin?
                                #                  - Fixed distance from camera
                                #                  - 
    projected_base_depth = base_pos[2] / base_pos[3] # projected screen-space depth of base position
    cursor_pos_hom = np.array((cursor_pos[0], cursor_pos[1], projected_base_depth, 1))
    unprojected = np.linalg.inv(P) @ cursor_pos_hom
    target = unprojected[:3] / unprojected[3]

    # Move ball towards target
    factor = (0.05 ** timestep) # Every unit of time, the distance from ball to target is scaled by 0.05
    prev_pos = np.array(ball_center[0])
    disp = (prev_pos - target)
    disp = factor * disp
    next_pos = target + factor * disp
    # ball_center[0] = next_pos
    ball_velocity[0] = (next_pos - prev_pos) / timestep
    return


window = ti.ui.Window("Cloth and Sphere", (768, 768),
                        vsync=True)

gui = window.get_gui()

canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0
initialize_mass_points()

while window.running:

    with gui.sub_window("Parameters", 0.05, 0.05, 0.9, 0.5) as w:
        w.text(f"simulation time: {current_t}")
        w.text("Press R to reset cloth")
        substeps = w.slider_int("Substeps per frame", substeps, 5, 200)
        w.text("Hold SPACEBAR to control ball with mouse movement")
        w.text("\nContact slowdown multipliers (non-physical \"friction\"):"
            "\nDuring contact, tangential velocity gets scaled by this factor per unit time.")
        floor_sticky_slowdown[None] = w.slider_float("Floor slowdown multiplier", floor_sticky_slowdown[None], 0.00001, 1)
        ball_sticky_slowdown[None] = w.slider_float("Ball slowdown multiplier", ball_sticky_slowdown[None], 0.00001, 1)
        w.text("\nSpring constants")
        for spring_type in spring_types:
            spring_constants[spring_type][None] = w.slider_float(spring_type, spring_constants[spring_type][None], 0, 5e6)
        w.text("\n Damping")
        spring_damping_constant[None] = w.slider_float("Spring damping", spring_damping_constant[None], 0, 500)
        drag_damping[None] = w.slider_float("Air drag damping", drag_damping[None], 0, 20)

    for e in window.get_events(ti.ui.PRESS):
        if e.key == "r":
            # Reset
            initialize_mass_points()
            current_t = 0

        if e.key == "t":
            # Reset
            show_param_window = not show_param_window
            

    if window.is_pressed(ti.ui.SPACE):
        cursor_pos = window.get_cursor_pos()
        width, height = window.get_window_shape()
        aspect = width / height
        proj, view = camera.get_projection_matrix(aspect), camera.get_view_matrix()
        control_ball_with_cursor(cursor_pos, proj, view, step_duration)
    update_ball(step_duration)

    dt = step_duration / float(substeps)
    for i in range(substeps):
        substep(dt)
        current_t += dt
    update_vertices()

    camera.position(0.0, 0.0, 2)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 0, 3), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(triangle_vertices,
                indices=triangle_indices,
                per_vertex_color=triangle_colors,
                two_sided=True)
    scene.mesh(floor_vertices, indices=floor_indices, color=(.5, .5, .5), two_sided=True)
    # scene.lines(triangle_vertices, )

    # Draw a smaller ball to avoid visual penetration
    scene.particles(ball_center,
                    radius=ball_radius * 0.9,
                    color=(0.5, 0.7, 0.2))
    canvas.scene(scene)
    window.show()

#TODO: include self-collision handling
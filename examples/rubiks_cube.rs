//! [gallery]
//! name = "Rubik Mini Game"
//! category = "Post Effects"
//! description = "Playable 3x3 Rubik's cube with keyboard turns and orbit camera controls."
//! instructions = "U D L R F B: turn a face clockwise\nHold Shift: inverse turn\nSpace: scramble from solved state\nEnter: reset to solved state"
//! order = 458
//!

use std::collections::VecDeque;
use std::f32::consts::FRAC_PI_2;

use glam::IVec3;
use myth::prelude::*;
use myth::resources::Key;

const TURN_DURATION: f32 = 0.16;
const SCRAMBLE_TURNS: usize = 20;
const CUBIE_STEP: f32 = 1.08;
const CUBIE_SIZE: f32 = 0.92;
const STICKER_SIZE: f32 = 0.72;
const STICKER_THICKNESS: f32 = 0.06;
const STICKER_OFFSET: f32 = (CUBIE_SIZE * 0.5) + (STICKER_THICKNESS * 0.55);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Axis {
    X,
    Y,
    Z,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Face {
    Up,
    Down,
    Left,
    Right,
    Front,
    Back,
}

impl Face {
    const ALL: [Self; 6] = [
        Self::Up,
        Self::Down,
        Self::Left,
        Self::Right,
        Self::Front,
        Self::Back,
    ];

    fn key(self) -> Key {
        match self {
            Self::Up => Key::U,
            Self::Down => Key::D,
            Self::Left => Key::L,
            Self::Right => Key::R,
            Self::Front => Key::F,
            Self::Back => Key::B,
        }
    }

    fn axis(self) -> Axis {
        match self {
            Self::Left | Self::Right => Axis::X,
            Self::Down | Self::Up => Axis::Y,
            Self::Back | Self::Front => Axis::Z,
        }
    }

    fn normal(self) -> IVec3 {
        match self {
            Self::Up => IVec3::new(0, 1, 0),
            Self::Down => IVec3::new(0, -1, 0),
            Self::Left => IVec3::new(-1, 0, 0),
            Self::Right => IVec3::new(1, 0, 0),
            Self::Front => IVec3::new(0, 0, 1),
            Self::Back => IVec3::new(0, 0, -1),
        }
    }

    fn material(self, materials: &MaterialCache) -> MaterialHandle {
        match self {
            Self::Up => materials.up,
            Self::Down => materials.down,
            Self::Left => materials.left,
            Self::Right => materials.right,
            Self::Front => materials.front,
            Self::Back => materials.back,
        }
    }

    fn is_visible_on(self, coord: IVec3) -> bool {
        let normal = self.normal();
        match self.axis() {
            Axis::X => coord.x == normal.x,
            Axis::Y => coord.y == normal.y,
            Axis::Z => coord.z == normal.z,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TurnCommand {
    face: Face,
    clockwise: bool,
}

impl TurnCommand {
    fn rotation_axis(self) -> Vec3 {
        ivec3_to_vec3(self.face.normal())
    }

    fn final_angle(self) -> f32 {
        if self.clockwise {
            -FRAC_PI_2
        } else {
            FRAC_PI_2
        }
    }

    fn angle_at(self, progress: f32) -> f32 {
        self.final_angle() * progress
    }

    fn matches(self, coord: IVec3) -> bool {
        let normal = self.face.normal();
        match self.face.axis() {
            Axis::X => coord.x == normal.x,
            Axis::Y => coord.y == normal.y,
            Axis::Z => coord.z == normal.z,
        }
    }
}

struct GeometryCache {
    unit_box: GeometryHandle,
}

struct MaterialCache {
    body: MaterialHandle,
    pedestal: MaterialHandle,
    trim: MaterialHandle,
    up: MaterialHandle,
    down: MaterialHandle,
    left: MaterialHandle,
    right: MaterialHandle,
    front: MaterialHandle,
    back: MaterialHandle,
}

impl GeometryCache {
    fn register(engine: &Engine) -> Self {
        Self {
            unit_box: engine
                .assets
                .geometries
                .add(Geometry::new_box(1.0, 1.0, 1.0)),
        }
    }
}

impl MaterialCache {
    fn register(engine: &Engine) -> Self {
        Self {
            body: engine
                .assets
                .materials
                .add(Material::new_unlit(Vec4::new(0.10, 0.11, 0.14, 1.0))),
            pedestal: engine
                .assets
                .materials
                .add(Material::new_unlit(Vec4::new(0.18, 0.20, 0.24, 1.0))),
            trim: engine
                .assets
                .materials
                .add(Material::new_unlit(Vec4::new(0.30, 0.33, 0.40, 1.0))),
            up: engine
                .assets
                .materials
                .add(Material::new_unlit(Vec4::new(0.96, 0.97, 0.99, 1.0))),
            down: engine
                .assets
                .materials
                .add(Material::new_unlit(Vec4::new(0.98, 0.83, 0.18, 1.0))),
            left: engine
                .assets
                .materials
                .add(Material::new_unlit(Vec4::new(0.98, 0.48, 0.12, 1.0))),
            right: engine
                .assets
                .materials
                .add(Material::new_unlit(Vec4::new(0.84, 0.20, 0.19, 1.0))),
            front: engine
                .assets
                .materials
                .add(Material::new_unlit(Vec4::new(0.13, 0.63, 0.24, 1.0))),
            back: engine
                .assets
                .materials
                .add(Material::new_unlit(Vec4::new(0.17, 0.39, 0.86, 1.0))),
        }
    }
}

struct Cubie {
    root: NodeHandle,
    home_coord: IVec3,
    coord: IVec3,
    orientation: Quat,
}

struct TurningPiece {
    index: usize,
    start_coord: IVec3,
    start_position: Vec3,
    start_rotation: Quat,
}

struct ActiveTurn {
    command: TurnCommand,
    elapsed: f32,
    pieces: Vec<TurningPiece>,
}

struct RubikMiniGame {
    controls: OrbitControls,
    cubies: Vec<Cubie>,
    turn_queue: VecDeque<TurnCommand>,
    active_turn: Option<ActiveTurn>,
    is_scrambling: bool,
    game_started: bool,
    move_count: u32,
    scramble_rng: u64,
}

impl RubikMiniGame {
    fn spawn_scene(
        scene: &mut Scene,
        geometry: &GeometryCache,
        materials: &MaterialCache,
    ) -> Vec<Cubie> {
        let pedestal = scene.add_mesh(Mesh::new(geometry.unit_box, materials.pedestal));
        scene
            .edit_node(pedestal)
            .set_position(0.0, -2.25, 0.0)
            .set_scale_xyz(4.8, 0.28, 4.8);

        let trim = scene.add_mesh(Mesh::new(geometry.unit_box, materials.trim));
        scene
            .edit_node(trim)
            .set_position(0.0, -1.95, 0.0)
            .set_scale_xyz(3.0, 0.12, 3.0);

        let mut cubies = Vec::new();

        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    let coord = IVec3::new(x, y, z);
                    if coord == IVec3::ZERO {
                        continue;
                    }

                    let root = scene.create_node_with_name(&format!("Cubie_{x}_{y}_{z}"));
                    scene.push_root_node(root);

                    let body = scene
                        .add_mesh_to_parent(Mesh::new(geometry.unit_box, materials.body), root);
                    scene.edit_node(body).set_scale(CUBIE_SIZE);

                    for face in Face::ALL {
                        if !face.is_visible_on(coord) {
                            continue;
                        }

                        let sticker = scene.add_mesh_to_parent(
                            Mesh::new(geometry.unit_box, face.material(materials)),
                            root,
                        );
                        let offset = ivec3_to_vec3(face.normal()) * STICKER_OFFSET;
                        let scale = sticker_scale(face.axis());
                        scene
                            .edit_node(sticker)
                            .set_position_vec(offset)
                            .set_scale_xyz(scale.x, scale.y, scale.z);
                    }

                    let cubie = Cubie {
                        root,
                        home_coord: coord,
                        coord,
                        orientation: Quat::IDENTITY,
                    };

                    Self::write_cubie_transform(
                        scene,
                        cubie.root,
                        coord_to_position(cubie.coord),
                        cubie.orientation,
                    );
                    cubies.push(cubie);
                }
            }
        }

        cubies
    }

    fn print_controls() {
        println!("========================================");
        println!("Rubik Mini Game");
        println!("Mouse drag: orbit camera");
        println!("Mouse wheel: zoom");
        println!("U D L R F B: turn a face clockwise");
        println!("Hold Shift: inverse turn");
        println!("Space: scramble from solved state");
        println!("Enter: reset to solved state");
        println!("========================================");
    }

    fn write_cubie_transform(
        scene: &mut Scene,
        handle: NodeHandle,
        position: Vec3,
        rotation: Quat,
    ) {
        if let Some(node) = scene.node_mut(handle) {
            node.transform.position = position;
            node.transform.rotation = rotation;
        }
    }

    fn restore_solved(&mut self, scene: &mut Scene) {
        self.turn_queue.clear();
        self.active_turn = None;
        self.is_scrambling = false;
        self.game_started = false;
        self.move_count = 0;

        for cubie in &mut self.cubies {
            cubie.coord = cubie.home_coord;
            cubie.orientation = Quat::IDENTITY;
            Self::write_cubie_transform(
                scene,
                cubie.root,
                coord_to_position(cubie.coord),
                cubie.orientation,
            );
        }
    }

    fn queue_scramble(&mut self, scene: &mut Scene) {
        self.restore_solved(scene);
        self.is_scrambling = true;
        self.game_started = true;

        let mut last_face = None;
        for _ in 0..SCRAMBLE_TURNS {
            let mut command = self.random_turn();
            while Some(command.face) == last_face {
                command = self.random_turn();
            }
            self.turn_queue.push_back(command);
            last_face = Some(command.face);
        }

        println!("Scrambling: {} moves queued.", SCRAMBLE_TURNS);
    }

    fn random_turn(&mut self) -> TurnCommand {
        let face_index = (self.next_random_u32() as usize) % Face::ALL.len();
        TurnCommand {
            face: Face::ALL[face_index],
            clockwise: (self.next_random_u32() & 1) == 0,
        }
    }

    fn next_random_u32(&mut self) -> u32 {
        let mut state = self.scramble_rng;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        self.scramble_rng = state;
        state as u32
    }

    fn handle_input(&mut self, scene: &mut Scene, input: &myth::resources::Input) {
        if input.get_key_down(Key::Enter) {
            self.restore_solved(scene);
            println!("Cube reset. Press Space to start a new scramble.");
            return;
        }

        if input.get_key_down(Key::Space)
            && self.active_turn.is_none()
            && self.turn_queue.is_empty()
        {
            self.queue_scramble(scene);
            return;
        }

        if self.is_scrambling {
            return;
        }

        let inverse = input.get_key(Key::ShiftLeft) || input.get_key(Key::ShiftRight);
        let mut queued_this_frame = 0_u32;

        for face in Face::ALL {
            if input.get_key_down(face.key()) {
                self.turn_queue.push_back(TurnCommand {
                    face,
                    clockwise: !inverse,
                });
                queued_this_frame += 1;
            }
        }

        if self.game_started {
            self.move_count += queued_this_frame;
        }
    }

    fn begin_next_turn(&mut self) {
        if self.active_turn.is_some() {
            return;
        }

        let Some(command) = self.turn_queue.pop_front() else {
            return;
        };

        let mut pieces = Vec::with_capacity(9);
        for (index, cubie) in self.cubies.iter().enumerate() {
            if command.matches(cubie.coord) {
                pieces.push(TurningPiece {
                    index,
                    start_coord: cubie.coord,
                    start_position: coord_to_position(cubie.coord),
                    start_rotation: cubie.orientation,
                });
            }
        }

        self.active_turn = Some(ActiveTurn {
            command,
            elapsed: 0.0,
            pieces,
        });
    }

    fn animate_active_turn(&mut self, scene: &mut Scene, dt: f32) {
        let mut finished = false;

        if let Some(active_turn) = &mut self.active_turn {
            active_turn.elapsed += dt;
            let progress = (active_turn.elapsed / TURN_DURATION).min(1.0);
            let rotation = Quat::from_axis_angle(
                active_turn.command.rotation_axis(),
                active_turn.command.angle_at(progress),
            );

            for piece in &active_turn.pieces {
                let position = rotation * piece.start_position;
                let orientation = (rotation * piece.start_rotation).normalize();
                let handle = self.cubies[piece.index].root;
                Self::write_cubie_transform(scene, handle, position, orientation);
            }

            finished = progress >= 1.0;
        }

        if finished {
            self.finish_active_turn(scene);
        }
    }

    fn finish_active_turn(&mut self, scene: &mut Scene) {
        let Some(active_turn) = self.active_turn.take() else {
            return;
        };

        let final_rotation = Quat::from_axis_angle(
            active_turn.command.rotation_axis(),
            active_turn.command.final_angle(),
        );

        for piece in active_turn.pieces {
            let cubie = &mut self.cubies[piece.index];
            cubie.coord = snap_coord(final_rotation * ivec3_to_vec3(piece.start_coord));
            cubie.orientation = snap_orientation(final_rotation * piece.start_rotation);
            Self::write_cubie_transform(
                scene,
                cubie.root,
                coord_to_position(cubie.coord),
                cubie.orientation,
            );
        }

        if self.is_scrambling && self.turn_queue.is_empty() {
            self.is_scrambling = false;
            println!("Scramble ready. Solve it with U D L R F B (Shift for inverse).");
        }

        if self.game_started
            && !self.is_scrambling
            && self.turn_queue.is_empty()
            && self.is_solved()
        {
            println!(
                "Solved in {} moves. Press Space to scramble again.",
                self.move_count
            );
            self.game_started = false;
        }
    }

    fn is_solved(&self) -> bool {
        self.cubies
            .iter()
            .all(|cubie| cubie.coord == cubie.home_coord && quat_is_identity(cubie.orientation))
    }
}

impl AppHandler for RubikMiniGame {
    fn init(engine: &mut Engine, _window: &dyn Window) -> Self {
        Self::print_controls();

        let geometry = GeometryCache::register(engine);
        let materials = MaterialCache::register(engine);

        let scene = engine.scene_manager.create_active();
        let cubies = Self::spawn_scene(scene, &geometry, &materials);

        let camera_position = Vec3::new(6.0, 5.5, 7.5);
        let camera = scene.add_camera(Camera::new_perspective(45.0, 1280.0 / 720.0, 0.1));
        scene
            .edit_node(camera)
            .set_position_vec(camera_position)
            .look_at(Vec3::ZERO);
        scene.active_camera = Some(camera);

        let mut controls = OrbitControls::new(camera_position, Vec3::ZERO);
        controls.enable_pan = false;
        controls.min_distance = 5.5;
        controls.max_distance = 16.0;
        controls.rotate_speed = 0.55;
        controls.zoom_speed = 0.7;

        Self {
            controls,
            cubies,
            turn_queue: VecDeque::new(),
            active_turn: None,
            is_scrambling: false,
            game_started: false,
            move_count: 0,
            scramble_rng: 0xC0FF_EE12_3456_789A,
        }
    }

    fn update(&mut self, engine: &mut Engine, _window: &dyn Window, frame: &FrameState) {
        let Some(scene) = engine.scene_manager.active_scene_mut() else {
            return;
        };

        self.handle_input(scene, &engine.input);

        let _ = scene.edit_main_camera_bundle(|transform, camera| {
            self.controls
                .update(transform, &engine.input, camera.fov(), frame.dt);
        });

        self.begin_next_turn();
        self.animate_active_turn(scene, frame.dt);
        self.begin_next_turn();
    }
}

fn sticker_scale(axis: Axis) -> Vec3 {
    match axis {
        Axis::X => Vec3::new(STICKER_THICKNESS, STICKER_SIZE, STICKER_SIZE),
        Axis::Y => Vec3::new(STICKER_SIZE, STICKER_THICKNESS, STICKER_SIZE),
        Axis::Z => Vec3::new(STICKER_SIZE, STICKER_SIZE, STICKER_THICKNESS),
    }
}

fn coord_to_position(coord: IVec3) -> Vec3 {
    ivec3_to_vec3(coord) * CUBIE_STEP
}

fn ivec3_to_vec3(coord: IVec3) -> Vec3 {
    Vec3::new(coord.x as f32, coord.y as f32, coord.z as f32)
}

fn snap_coord(position: Vec3) -> IVec3 {
    IVec3::new(
        position.x.round() as i32,
        position.y.round() as i32,
        position.z.round() as i32,
    )
}

fn snap_axis(direction: Vec3) -> Vec3 {
    let axes = [
        Vec3::X,
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::Y,
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::Z,
        Vec3::new(0.0, 0.0, -1.0),
    ];

    let mut best_axis = Vec3::X;
    let mut best_dot = f32::NEG_INFINITY;
    for axis in axes {
        let dot = direction.dot(axis);
        if dot > best_dot {
            best_dot = dot;
            best_axis = axis;
        }
    }
    best_axis
}

fn snap_orientation(rotation: Quat) -> Quat {
    let x_axis = snap_axis(rotation * Vec3::X);
    let y_axis = snap_axis(rotation * Vec3::Y);
    let z_axis = x_axis.cross(y_axis).normalize();
    Quat::from_mat3(&Mat3::from_cols(x_axis, y_axis, z_axis)).normalize()
}

fn quat_is_identity(rotation: Quat) -> bool {
    rotation.dot(Quat::IDENTITY).abs() > 0.9999
}

#[myth::main]
fn main() -> myth::Result<()> {
    App::new().run::<RubikMiniGame>()
}

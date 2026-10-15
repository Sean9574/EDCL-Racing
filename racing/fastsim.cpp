// Compiled simulator (spec §5.7): time trials of network populations and the expert's lap, computed exactly as
// racing/sim.py computes them. Cars never interact, so every pair of map and car is one job: the car is driven from
// the start line until it is done, on its own, and threads take the next job from a shared counter. Every formula
// follows sim.py, track.py, expert.py and actions.py with the same operands in the same order; the build turns off
// fused multiply-add, so every product and sum rounds as it does in numpy.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

extern "C" {

struct Track {
    const double* dist;          // (rows, cols) m from each road cell to the nearest grass cell; 0 on grass
    int64_t rows, cols;
    double origin_x, origin_y, width;
    const double* centerline;    // (m, 2)
    const double* tangent;       // (m, 2)
    const double* curvature;     // (m,)
    int64_t m;
    double spacing;
    double start_x, start_y, start_heading;
    int64_t back, ahead;         // centerline points searched behind and ahead of the last match
    int64_t sight;               // the expert's sight in centerline points
    double time_limit;           // s
};

struct Settings {
    double dt, grip, yaw_min_speed, speed_scale, ray_range, width_scale, ray_min_step, grid_res, stall_time, time_eps;
    double expert_grip, lookahead_base, lookahead_time, grip_margin, speed_band, expert_sight;
    int64_t ray_max_iterations, laps, n_inputs;
    const double* ray_angles;
    int64_t n_rays;
    const double* steer_levels;
    int64_t n_steer;
    const double* throttle_levels;
    int64_t n_throttle;
};

struct Programs {                // network programs of all cars (spec §5.7)
    const int64_t* node_offset;  // (n_cars + 1): car c's nodes are node_offset[c] .. node_offset[c + 1] - 1
    const double* bias;          // per node
    const int64_t* activation;   // per node: 0 ReLU, 1 tanh, 2 identity
    const int64_t* conn_offset;  // (n_nodes + 1): node k's connections are conn_offset[k] .. conn_offset[k + 1] - 1
    const int64_t* source;       // per connection: slot within its car (input i at i, the car's node j at n_inputs + j)
    const double* weight;        // per connection
    const int64_t* outputs;      // (n_cars, n_outputs) slots of the outputs, in action order
    int64_t n_outputs;
};

struct Results {                 // (n_tracks, n_cars), one row per track
    uint8_t* finished;
    uint8_t* crashed;
    double* progress;            // m
    double* finish_time;         // s, nan unless finished
    int64_t* laps_done;
    int64_t* steps;
};

}  // extern "C"

namespace {

int64_t wrap(int64_t i, int64_t m) {             // np.mod: never negative
    int64_t k = i % m;
    return k < 0 ? k + m : k;
}

// track.wall_distance
double wall_distance(const Track& t, const Settings& s, double x, double y) {
    int64_t c = (int64_t)std::floor((x - t.origin_x) / s.grid_res);
    int64_t r = (int64_t)std::floor((y - t.origin_y) / s.grid_res);
    if (r < 0 || r >= t.rows || c < 0 || c >= t.cols) return 0.0;
    return t.dist[r * t.cols + c];
}

// sim.cast_rays for one ray: sphere tracing for at most `cap` iterations
double cast_ray(const Track& t, const Settings& s, double x, double y, double angle, double reach, int64_t cap) {
    double dx = std::cos(angle), dy = std::sin(angle);
    double dist = 0.0;
    for (int64_t iteration = 0; iteration < cap; ++iteration) {
        double d = wall_distance(t, s, x + dx * dist, y + dy * dist);
        if (d <= 0.0) break;
        dist = dist + std::max(d, s.ray_min_step);
        if (!(dist < reach)) break;
    }
    return std::min(dist, reach);
}

// Values of every slot of car `car` for inputs `obs`: the inputs, then each node in order (spec §5.7). A node sums
// source value × weight over its connections from zero, in order, then adds its bias, then applies its activation;
// this is the order of np.einsum for dense populations and of np.bincount for NEAT.
void evaluate(const Programs& p, int64_t car, int64_t n_inputs, const double* obs, std::vector<double>& values) {
    int64_t first = p.node_offset[car], last = p.node_offset[car + 1];
    values.assign(n_inputs + (last - first), 0.0);
    for (int64_t i = 0; i < n_inputs; ++i) values[i] = obs[i];
    for (int64_t k = first; k < last; ++k) {
        double total = 0.0;
        for (int64_t c = p.conn_offset[k]; c < p.conn_offset[k + 1]; ++c) total += values[p.source[c]] * p.weight[c];
        total = total + p.bias[k];
        if (p.activation[k] == 0) total = std::max(total, 0.0);
        else if (p.activation[k] == 1) total = std::tanh(total);
        values[n_inputs + (k - first)] = total;
    }
}

// nets.act / np.argmax: the first largest output
int64_t network_action(const Programs& p, int64_t car, int64_t n_inputs, const double* obs,
                       std::vector<double>& values) {
    evaluate(p, car, n_inputs, obs, values);
    const int64_t* out = p.outputs + car * p.n_outputs;
    int64_t best = 0;
    for (int64_t j = 1; j < p.n_outputs; ++j)
        if (values[out[j]] > values[out[best]]) best = j;
    return best;
}

// actions.controls_to_class for one level list: the first nearest level
int64_t nearest(const double* levels, int64_t n, double value) {
    int64_t best = 0;
    for (int64_t k = 1; k < n; ++k)
        if (std::fabs(value - levels[k]) < std::fabs(value - levels[best])) best = k;
    return best;
}

// expert.expert_controls and expert.target_speed, quantized to an action class (expert.expert_policy)
int64_t expert_action(const Track& t, const Settings& s, double x, double y, double heading, double speed,
                      int64_t index) {
    int64_t here = wrap(index, t.m);
    double look = s.lookahead_base + s.lookahead_time * speed;
    int64_t aim = wrap(here + (int64_t)std::nearbyint(look / t.spacing), t.m);
    double dx = t.centerline[2 * aim] - x, dy = t.centerline[2 * aim + 1] - y;
    double turn = std::atan2(dy, dx) - heading;
    double alpha = std::atan2(std::sin(turn), std::cos(turn));       // np.angle(np.exp(1j * turn))
    double path_curvature = 2.0 * std::sin(alpha) / std::max(std::hypot(dx, dy), 1e-6);
    double steer = std::clamp(speed * speed * path_curvature / s.expert_grip, -1.0, 1.0);

    double kappa = 0.0;
    for (int64_t k = 0; k <= t.sight; ++k) kappa = std::max(kappa, std::fabs(t.curvature[wrap(here + k, t.m)]));
    double corner_speed = std::sqrt(s.grip_margin * s.expert_grip / std::max(kappa, 1e-9));
    double sight_speed = std::sqrt(2.0 * s.grip_margin * s.expert_grip * s.expert_sight);
    double target = std::min(corner_speed, sight_speed);
    double throttle = speed < target - s.speed_band ? 1.0 : (speed > target + s.speed_band ? -1.0 : 0.0);

    return nearest(s.steer_levels, s.n_steer, steer) * s.n_throttle +
           nearest(s.throttle_levels, s.n_throttle, throttle);
}

// sim.simulate for one car: sense, act, step and track progress until the car is done
void drive(const Track& t, const Settings& s, const Programs* p, const double* setup, int64_t car, int64_t job,
           const Results& out) {
    double ray_length = setup[2 * car], steer_gain = setup[2 * car + 1];
    int64_t cap = (int64_t)std::ceil(s.ray_max_iterations * ray_length / s.ray_range);   // sim.ray_caps

    double x = t.start_x, y = t.start_y, heading = t.start_heading, speed = 0.0, time = 0.0;
    int64_t index = 0, laps_done = 0, steps = 0;
    double progress = 0.0, last_progress_time = 0.0, finish_time = std::nan("");
    bool crashed = false, finished = false, done = false;

    std::vector<double> obs(s.n_inputs, 0.0), values;
    while (!done) {
        int64_t action;
        if (p == nullptr) {
            action = expert_action(t, s, x, y, heading, speed, index);
        } else {
            for (int64_t k = 0; k < s.n_rays; ++k)                                     // sim.sense
                obs[k] = cast_ray(t, s, x, y, heading + s.ray_angles[k], ray_length, cap) / s.ray_range;
            obs[s.n_rays] = speed / s.speed_scale;
            int64_t i = wrap(index, t.m);                                               // sim.road_position
            double dx = x - t.centerline[2 * i], dy = y - t.centerline[2 * i + 1];
            double left = -dx * t.tangent[2 * i + 1] + dy * t.tangent[2 * i];
            obs[s.n_rays + 1] = std::clamp(-left / (t.width / 2), -1.0, 1.0);
            obs[s.n_rays + 2] = t.width / s.width_scale;
            action = network_action(*p, car, s.n_inputs, obs.data(), values);
        }
        double steer = s.steer_levels[action / s.n_throttle];                          // actions.class_to_controls
        double throttle = s.throttle_levels[action % s.n_throttle];

        double a_lat = std::clamp(steer * steer_gain, -1.0, 1.0) * s.grip;              // sim.step
        double a_long = std::clamp(throttle, -1.0, 1.0) * s.grip;
        double total = std::hypot(a_lat, a_long);
        double scale = total > s.grip ? s.grip / std::max(total, 1e-12) : 1.0;
        a_lat = a_lat * scale;
        a_long = a_long * scale;
        double yaw_rate = a_lat / std::max(speed, s.yaw_min_speed);
        speed = std::max(speed + a_long * s.dt, 0.0);
        heading = heading + yaw_rate * s.dt;
        x = x + speed * std::cos(heading) * s.dt;
        y = y + speed * std::sin(heading) * s.dt;
        time += s.dt;
        ++steps;

        crashed = wall_distance(t, s, x, y) <= 0.0;
        bool stalled = false;
        if (crashed) {
            speed = 0.0;
        } else {                                                                        // sim._update_progress
            int64_t best = index - t.back;
            double best_d2 = INFINITY;
            for (int64_t candidate = index - t.back; candidate <= index + t.ahead; ++candidate) {
                int64_t q = wrap(candidate, t.m);
                double ex = t.centerline[2 * q] - x, ey = t.centerline[2 * q + 1] - y;
                double d2 = ex * ex + ey * ey;
                if (d2 < best_d2) best_d2 = d2, best = candidate;
            }
            index = best;
            double reached = index * t.spacing;
            if (reached > progress) last_progress_time = time;
            progress = std::max(progress, reached);
            int64_t laps_now = (int64_t)std::nearbyint(progress / t.spacing) / t.m;
            if (laps_now > laps_done) {
                laps_done = laps_now;
                if (laps_done >= s.laps) finished = true, finish_time = time;
            }
            stalled = time - last_progress_time >= s.stall_time - s.time_eps;
        }
        bool out_of_time = time >= t.time_limit - s.time_eps;
        done = crashed || stalled || out_of_time || finished;
    }
    out.finished[job] = finished;
    out.crashed[job] = crashed;
    out.progress[job] = progress;
    out.finish_time[job] = finish_time;
    out.laps_done[job] = laps_done;
    out.steps[job] = steps;
}

}  // namespace

extern "C" void simulate(const Track* tracks, int64_t n_tracks, const Settings* settings, const Programs* programs,
                         const double* setup, int64_t n_cars, int64_t threads, const Results* out) {
    int64_t jobs = n_tracks * n_cars;
    if (jobs == 0) return;
    std::atomic<int64_t> next{0};
    auto work = [&] {
        for (int64_t job = next++; job < jobs; job = next++)
            drive(tracks[job / n_cars], *settings, programs, setup, job % n_cars, job, *out);
    };
    threads = std::max<int64_t>(1, std::min(threads, jobs));
    if (threads == 1) {
        work();
        return;
    }
    std::vector<std::thread> pool;
    for (int64_t k = 0; k < threads; ++k) pool.emplace_back(work);
    for (auto& thread : pool) thread.join();
}

extern "C" void forward(const Programs* programs, const double* obs, int64_t n_cars, int64_t n_inputs, double* out) {
    std::vector<double> values;
    for (int64_t car = 0; car < n_cars; ++car) {
        evaluate(*programs, car, n_inputs, obs + car * n_inputs, values);
        for (int64_t j = 0; j < programs->n_outputs; ++j)
            out[car * programs->n_outputs + j] = values[programs->outputs[car * programs->n_outputs + j]];
    }
}

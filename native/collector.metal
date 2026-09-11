#include <metal_stdlib>

using namespace metal;

constant ulong kSentinel = 0x7ffffffffffffffful;

struct CameraParameters {
    packed_float3 center;
    packed_float3 center_error;
    float rotation[9];
    float coefficients[6];
    float a11;
    float a12;
    float a22;
    float u0;
    float v0;
    float max_angle;
    uint width;
    uint height;
    uint point_count;
    uint selected_count;
    uint depth_width;
    uint depth_height;
};

struct Projection {
    float u;
    float v;
    float angle;
    float distance;
};

enum DecisionFlag : uint {
    kValid = 1u,
    kReliablePatch = 2u,
    kSurfaceRejected = 4u,
    kVisible = 8u,
    kAmbiguous = 16u,
    kProjectionAmbiguous = 32u,
};

inline float3 camera_point(float3 point, constant CameraParameters &camera) {
    float3 delta = point - float3(camera.center);
    return float3(
        fma(delta.z, camera.rotation[6],
            fma(delta.y, camera.rotation[3], delta.x * camera.rotation[0])),
        fma(delta.z, camera.rotation[7],
            fma(delta.y, camera.rotation[4], delta.x * camera.rotation[1])),
        fma(delta.z, camera.rotation[8],
            fma(delta.y, camera.rotation[5], delta.x * camera.rotation[2]))
    );
}

kernel void project_points(
    device const packed_float3 *points [[buffer(0)]],
    device const uint *selected [[buffer(1)]],
    device Projection *projection [[buffer(2)]],
    device ulong *depth_keys [[buffer(3)]],
    device uint *flags [[buffer(4)]],
    constant CameraParameters &parameters [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= parameters.selected_count) return;
    uint point_id = selected[index];
    float3 camera = camera_point(float3(points[point_id]), parameters);
    float radial = precise::sqrt(camera.x * camera.x + camera.y * camera.y);
    float angle = atan2(radial, camera.z);
    float distorted = angle;
    float power = angle * angle;
    for (uint coefficient = 0; coefficient < 6; ++coefficient) {
        distorted += parameters.coefficients[coefficient] * power;
        power *= angle;
    }
    float scale = radial > 1.0e-10f ? distorted / radial : 1.0f;
    float normalized_x = camera.x * scale;
    float normalized_y = camera.y * scale;
    float u = parameters.a11 * normalized_x + parameters.a12 * normalized_y + parameters.u0;
    float v = parameters.a22 * normalized_y + parameters.v0;
    float distance = precise::sqrt(
        (camera.x * camera.x + camera.y * camera.y) + camera.z * camera.z
    );
    projection[index] = {u, v, angle, distance};
    bool valid = distance > 0.1f && isfinite(u) && isfinite(v)
        && u >= 0.0f && v >= 0.0f
        && u < float(parameters.width - 1u) && v < float(parameters.height - 1u)
        && angle < parameters.max_angle;
    bool projection_ambiguous = !isfinite(u) || !isfinite(v) || !isfinite(angle)
        || !isfinite(distance)
        || (valid && (abs(u - rint(u)) <= 0.02f || abs(v - rint(v)) <= 0.02f))
        || abs(u) <= 0.02f || abs(v) <= 0.02f
        || abs(u - float(parameters.width - 1u)) <= 0.02f
        || abs(v - float(parameters.height - 1u)) <= 0.02f
        || abs(angle - parameters.max_angle) <= 0.000002f
        || abs(distance - 0.1f) <= 0.000002f
        || (angle >= 0.0f && angle <= 0.001f);
    flags[index] = (valid ? kValid : 0u)
        | (projection_ambiguous ? kProjectionAmbiguous : 0u);
    if (!valid) {
        depth_keys[index] = kSentinel;
        return;
    }
    long quantized = long(rint(distance * 1000000.0f));
    depth_keys[index] = ulong(quantized) * ulong(parameters.point_count) + ulong(point_id);
}

kernel void pack_depth_keys(
    device const uint *selected [[buffer(0)]],
    device const Projection *projection [[buffer(1)]],
    device const uint *flags [[buffer(2)]],
    device ulong *depth_keys [[buffer(3)]],
    constant CameraParameters &parameters [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= parameters.selected_count) return;
    if (!(flags[index] & kValid)) {
        depth_keys[index] = kSentinel;
        return;
    }
    long quantized = long(rint(projection[index].distance * 1000000.0f));
    depth_keys[index] = ulong(quantized) * ulong(parameters.point_count) + ulong(selected[index]);
}

kernel void build_depth_high(
    device const Projection *projection [[buffer(0)]],
    device const ulong *depth_keys [[buffer(1)]],
    device const uint *flags [[buffer(2)]],
    device atomic_uint *depth_high [[buffer(3)]],
    constant CameraParameters &parameters [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= parameters.selected_count || !(flags[index] & kValid)) return;
    Projection p = projection[index];
    uint pixel = (uint(p.v) / 4u) * parameters.depth_width + uint(p.u) / 4u;
    atomic_fetch_min_explicit(
        &depth_high[pixel], uint(depth_keys[index] >> 32), memory_order_relaxed
    );
}

kernel void build_depth_low(
    device const Projection *projection [[buffer(0)]],
    device const ulong *depth_keys [[buffer(1)]],
    device const uint *flags [[buffer(2)]],
    device const atomic_uint *depth_high [[buffer(3)]],
    device atomic_uint *depth_low [[buffer(4)]],
    constant CameraParameters &parameters [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= parameters.selected_count || !(flags[index] & kValid)) return;
    Projection p = projection[index];
    uint pixel = (uint(p.v) / 4u) * parameters.depth_width + uint(p.u) / 4u;
    ulong key = depth_keys[index];
    if (uint(key >> 32) == atomic_load_explicit(&depth_high[pixel], memory_order_relaxed)) {
        atomic_fetch_min_explicit(&depth_low[pixel], uint(key), memory_order_relaxed);
    }
}

kernel void horizontal_depth_minimum(
    device const atomic_uint *depth_high [[buffer(0)]],
    device const atomic_uint *depth_low [[buffer(1)]],
    device ulong *horizontal [[buffer(2)]],
    constant CameraParameters &parameters [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    uint pixel_count = parameters.depth_width * parameters.depth_height;
    if (index >= pixel_count) return;
    int x = int(index % parameters.depth_width);
    uint y = index / parameters.depth_width;
    ulong minimum = kSentinel;
    for (int dx = -3; dx <= 3; ++dx) {
        int xx = x + dx;
        if (xx < 0 || xx >= int(parameters.depth_width)) continue;
        uint pixel = y * parameters.depth_width + uint(xx);
        ulong candidate =
            (ulong(atomic_load_explicit(&depth_high[pixel], memory_order_relaxed)) << 32)
            | ulong(atomic_load_explicit(&depth_low[pixel], memory_order_relaxed));
        minimum = min(minimum, candidate);
    }
    horizontal[index] = minimum;
}

kernel void vertical_depth_minimum(
    device const ulong *horizontal [[buffer(0)]],
    device ulong *neighborhood [[buffer(1)]],
    constant CameraParameters &parameters [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    uint pixel_count = parameters.depth_width * parameters.depth_height;
    if (index >= pixel_count) return;
    uint x = index % parameters.depth_width;
    int y = int(index / parameters.depth_width);
    ulong minimum = kSentinel;
    for (int dy = -3; dy <= 3; ++dy) {
        int yy = y + dy;
        if (yy < 0 || yy >= int(parameters.depth_height)) continue;
        minimum = min(minimum, horizontal[uint(yy) * parameters.depth_width + x]);
    }
    neighborhood[index] = minimum;
}

kernel void find_blockers(
    device const Projection *projection [[buffer(0)]],
    device const uint *flags [[buffer(1)]],
    device const atomic_uint *depth_high [[buffer(2)]],
    device const atomic_uint *depth_low [[buffer(3)]],
    device const ulong *neighborhood [[buffer(4)]],
    device ulong *exact_keys [[buffer(5)]],
    device ulong *blocker_keys [[buffer(6)]],
    constant CameraParameters &parameters [[buffer(7)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= parameters.selected_count) return;
    if (!(flags[index] & kValid)) {
        exact_keys[index] = kSentinel;
        blocker_keys[index] = kSentinel;
        return;
    }
    Projection p = projection[index];
    uint pixel = (uint(p.v) / 4u) * parameters.depth_width + uint(p.u) / 4u;
    exact_keys[index] =
        (ulong(atomic_load_explicit(&depth_high[pixel], memory_order_relaxed)) << 32)
        | ulong(atomic_load_explicit(&depth_low[pixel], memory_order_relaxed));
    blocker_keys[index] = neighborhood[pixel];
}

kernel void test_visibility(
    device const packed_float3 *points [[buffer(0)]],
    device const packed_float3 *normals [[buffer(1)]],
    device const uint *selected [[buffer(2)]],
    device const Projection *projection [[buffer(3)]],
    device const ulong *exact_keys [[buffer(4)]],
    device const ulong *blocker_keys [[buffer(5)]],
    device uint *flags [[buffer(6)]],
    constant CameraParameters &parameters [[buffer(7)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= parameters.selected_count || !(flags[index] & kValid)) return;
    uint point_id = selected[index];
    Projection p = projection[index];
    uint blocker_id = uint(blocker_keys[index] % ulong(parameters.point_count));
    float3 center = float3(parameters.center);
    float3 ray = (float3(points[point_id]) - center) / p.distance;
    float3 blocker_normal = float3(normals[blocker_id]);
    float denominator = dot(blocker_normal, ray);
    float divisor = abs(denominator) > 0.05f ? denominator : 1.0f;
    float3 blocker_offset = float3(points[blocker_id]) - center;
    float numerator = dot(blocker_normal, blocker_offset);
    float plane_depth = numerator / divisor;
    float3 hit = center + ray * plane_depth;
    float patch_distance = distance(hit, float3(points[blocker_id]));
    bool reliable = abs(denominator) > 0.15f && plane_depth > 0.1f
        && patch_distance < 0.04f;
    float exact_depth = float(exact_keys[index] / ulong(parameters.point_count)) / 1000000.0f;
    bool surface_rejected = reliable && p.distance > plane_depth + 0.02f;
    bool visible = p.distance <= exact_depth + 0.025f + 0.005f * p.distance
        && (!reliable || p.distance <= plane_depth + 0.02f);
    float incidence = dot(float3(normals[point_id]), -ray);
    visible = visible && incidence > 0.05f;
    uint value = kValid;
    if (reliable) value |= kReliablePatch;
    if (surface_rejected) value |= kSurfaceRejected;
    if (visible) value |= kVisible;
    // NumPy's mixed-precision reference promotes the blocker-plane numerator
    // and hit point to float64.  Bound float32 dot/divide/position error so
    // cancellation at long ranges is sent back to that exact CPU contract.
    constexpr float base_tolerance = 0.0002f;
    constexpr float float_epsilon = 1.1920928955078125e-7f;
    float numerator_sum = dot(abs(blocker_normal), abs(blocker_offset));
    float denominator_sum = dot(abs(blocker_normal), abs(ray));
    float numerator_error = 16.0f * float_epsilon * numerator_sum
        + dot(abs(blocker_normal), float3(parameters.center_error));
    float denominator_error = 16.0f * float_epsilon * denominator_sum;
    float denominator_floor = max(abs(denominator) - denominator_error, 1.0e-6f);
    float plane_error = (numerator_error + abs(plane_depth) * denominator_error)
        / denominator_floor + 8.0f * float_epsilon * abs(plane_depth);
    float ray_length = length(ray);
    float hit_rounding = 16.0f * float_epsilon
        * (length(abs(center)) + abs(plane_depth) * ray_length)
        + length(float3(parameters.center_error));
    float patch_error = plane_error * ray_length + hit_rounding;
    float depth_error = 8.0f * float_epsilon * max(abs(exact_depth), abs(p.distance));
    float incidence_error = 16.0f * float_epsilon
        * dot(abs(float3(normals[point_id])), abs(ray));
    bool ambiguous = abs(abs(denominator) - 0.05f)
            <= base_tolerance + denominator_error
        || abs(abs(denominator) - 0.15f) <= base_tolerance + denominator_error
        || abs(plane_depth - 0.1f) <= base_tolerance + plane_error
        || abs(patch_distance - 0.04f) <= base_tolerance + patch_error
        || abs(exact_depth + 0.025f + 0.005f * p.distance - p.distance)
            <= base_tolerance + depth_error
        || abs(plane_depth + 0.02f - p.distance) <= base_tolerance + plane_error
        || abs(incidence - 0.05f) <= base_tolerance + incidence_error;
    if (ambiguous) value |= kAmbiguous;
    flags[index] = value;
}

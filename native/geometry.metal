#include <metal_stdlib>

using namespace metal;

constant float kEpsilon = 0.000001f;
constant float kRayVoxelSize = 0.015f;
constant float kPenetrationMargin = 0.06f;
constant float kRayMinDistance = 0.1f;
constant float kRayMaxDistance = 30.0f;
// Studio copies these host-side globals into CUDA constant memory before
// launch: RAY_DIST_REF=30.0 and RAY_LOG_ODDS_HIT=0.1.
constant float kDistanceReference = 30.0f;
constant float kLogOddsHit = 0.1f;
constant float kLogOddsVisit = -1.0f;
constant float kNoiseThreshold = 0.2f;
constant float kSmoothingGridSpan = 1.1f;
constant uint kMaxProbeCount = 4096;

inline bool point_key(float3 point, float voxel_size, thread ulong &key) {
    int3 cell = int3(floor(point / voxel_size));
    constexpr int coordinate_limit = (1 << 20) - 1;
    if (any(cell < -coordinate_limit) || any(cell > coordinate_limit)) {
        key = 0;
        return false;
    }

    constexpr ulong coordinate_mask = (1ul << 21) - 1ul;
    ulong packed = (ulong(uint(cell.x)) & coordinate_mask)
        | ((ulong(uint(cell.y)) & coordinate_mask) << 21)
        | ((ulong(uint(cell.z)) & coordinate_mask) << 42);
    key = packed + 1ul;
    return key != 0ul;
}

inline bool point_key_from_origin(
    float3 point,
    float voxel_size,
    float3 grid_origin,
    thread ulong &key
) {
    int3 cell = int3(floor((point - grid_origin) / voxel_size));
    constexpr int coordinate_limit = (1 << 20) - 1;
    if (any(cell < -coordinate_limit) || any(cell > coordinate_limit)) {
        key = 0;
        return false;
    }
    constexpr ulong coordinate_mask = (1ul << 21) - 1ul;
    ulong packed = (ulong(uint(cell.x)) & coordinate_mask)
        | ((ulong(uint(cell.y)) & coordinate_mask) << 21)
        | ((ulong(uint(cell.z)) & coordinate_mask) << 42);
    key = packed + 1ul;
    return key != 0ul;
}

inline bool segment_intersects_box(
    float3 origin,
    float3 endpoint,
    float3 box_min,
    float3 box_max,
    thread float &entry,
    thread float &exit
) {
    float3 delta = endpoint - origin;
    entry = 0.0f;
    exit = 1.0f;
    for (uint axis = 0; axis < 3; ++axis) {
        if (abs(delta[axis]) <= kEpsilon) {
            if (origin[axis] < box_min[axis] || origin[axis] > box_max[axis]) {
                return false;
            }
            continue;
        }
        float inverse = 1.0f / delta[axis];
        float first = (box_min[axis] - origin[axis]) * inverse;
        float second = (box_max[axis] - origin[axis]) * inverse;
        if (first > second) {
            float temporary = first;
            first = second;
            second = temporary;
        }
        entry = max(entry, first);
        exit = min(exit, second);
        if (exit < entry) return false;
    }
    return true;
}

inline bool cell_key(int3 cell, thread ulong &key) {
    constexpr int coordinate_limit = (1 << 20) - 1;
    if (any(cell < -coordinate_limit) || any(cell > coordinate_limit)) {
        key = 0;
        return false;
    }
    constexpr ulong coordinate_mask = (1ul << 21) - 1ul;
    ulong packed = (ulong(uint(cell.x)) & coordinate_mask)
        | ((ulong(uint(cell.y)) & coordinate_mask) << 21)
        | ((ulong(uint(cell.z)) & coordinate_mask) << 42);
    key = packed + 1ul;
    return key != 0ul;
}

inline ulong mix_key(ulong key) {
    key ^= key >> 32;
    key *= 0x94d049bb133111ebul;
    key ^= key >> 29;
    key *= 0x9fb21c651e98df25ul;
    key ^= key >> 32;
    return key;
}

inline ulong contextual_key(ulong spatial_key, uint context_id) {
    ulong salt = mix_key(ulong(context_id) + 0x9e3779b97f4a7c15ul);
    ulong key = mix_key(spatial_key ^ salt);
    return key == 0ul ? 1ul : key;
}

inline int3 smoothing_cell(float3 point, float voxel_size, float3 context_center) {
    int shift = int((kSmoothingGridSpan / voxel_size) * 0.5f);
    // CUDA uses cvt.rzi after applying the positive per-grid shift.
    return int3((point - context_center) / voxel_size + float(shift));
}

inline uint find_key(
    device const ulong *keys,
    uint hash_mask,
    ulong key
) {
    ulong hash = mix_key(key);
    for (uint probe = 0; probe < kMaxProbeCount; ++probe) {
        uint slot = uint(hash) & hash_mask;
        ulong resident = keys[slot];
        if (resident == key) {
            return slot;
        }
        if (resident == 0ul) {
            return UINT_MAX;
        }
        hash = ulong(slot) + 1ul;
    }
    return UINT_MAX;
}

inline void atomic_add_float(device atomic_uint *address, float value) {
    uint old_bits = atomic_load_explicit(address, memory_order_relaxed);
    while (true) {
        float old_value = as_type<float>(old_bits);
        uint new_bits = as_type<uint>(old_value + value);
        uint expected = old_bits;
        if (atomic_compare_exchange_weak_explicit(
                address, &expected, new_bits,
                memory_order_relaxed, memory_order_relaxed)) {
            return;
        }
        old_bits = expected;
    }
}

struct RayRecord {
    packed_float3 origin;
    packed_float3 endpoint;
    float distance;
    float start_distance;
    uint endpoint_inside;
    uint reserved;
};

kernel void shoot_and_update(
    device const RayRecord *rays [[buffer(0)]],
    device const ulong *keys [[buffer(1)]],
    device const float4 *representatives [[buffer(2)]],
    device atomic_uint *score_bits [[buffer(3)]],
    constant uint &point_count [[buffer(4)]],
    constant uint &hash_mask [[buffer(5)]],
    constant float4 &core_minimum [[buffer(6)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= point_count) {
        return;
    }

    RayRecord ray = rays[index];
    float3 origin = float3(ray.origin);
    float3 endpoint = float3(ray.endpoint);
    float3 delta = endpoint - origin;
    float direction_distance = length(delta);
    if (!(ray.distance > 0.0f) || !(direction_distance > kEpsilon)) {
        return;
    }

    float3 direction = delta / direction_distance;
    if (all(abs(direction) <= kEpsilon)) {
        return;
    }

    float end_distance = min(ray.distance - kPenetrationMargin, kRayMaxDistance);
    float start_distance = max(kRayMinDistance, ray.start_distance);
    if (!(end_distance > start_distance)) {
        return;
    }

    float3 start = origin + direction * start_distance;
    int3 cell = int3(floor((start - core_minimum.xyz) / kRayVoxelSize));
    bool3 positive = direction >= 0.0f;
    int3 step = select(int3(-1), int3(1), positive);
    float3 boundary = core_minimum.xyz
        + (float3(cell) + select(float3(0.0f), float3(1.0f), positive))
            * kRayVoxelSize;

    float3 t_delta = float3(1.0e30f);
    float3 t_max = float3(1.0e30f);
    if (abs(direction.x) > kEpsilon) {
        t_delta.x = abs(kRayVoxelSize / direction.x);
        t_max.x = start_distance + abs((boundary.x - start.x) / direction.x);
    }
    if (abs(direction.y) > kEpsilon) {
        t_delta.y = abs(kRayVoxelSize / direction.y);
        t_max.y = start_distance + abs((boundary.y - start.y) / direction.y);
    }
    if (abs(direction.z) > kEpsilon) {
        t_delta.z = abs(kRayVoxelSize / direction.z);
        t_max.z = start_distance + abs((boundary.z - start.z) / direction.z);
    }

    // The CUDA kernel advances at most about 667 cells because it caps rays at
    // ten metres on a 15 mm grid.  The larger guard is only corruption safety.
    for (uint iteration = 0; iteration < 2048; ++iteration) {
        ulong key;
            if (cell_key(cell, key)) {
            uint slot = find_key(keys, hash_mask, key);
            if (slot != UINT_MAX) {
                float representative_distance = length(
                    representatives[slot].xyz - origin
                );
                float reference = max(kDistanceReference, kEpsilon);
                float evidence = (reference / (representative_distance + reference))
                    * kLogOddsHit;
                atomic_add_float(&score_bits[slot], evidence);
            }
        }

        float next_crossing = min(t_max.x, min(t_max.y, t_max.z));
        if (!(next_crossing < end_distance)) {
            break;
        }

        if (abs(t_max.x - next_crossing) <= kEpsilon) {
            cell.x += step.x;
            t_max.x += t_delta.x;
        }
        if (abs(t_max.y - next_crossing) <= kEpsilon) {
            cell.y += step.y;
            t_max.y += t_delta.y;
        }
        if (abs(t_max.z - next_crossing) <= kEpsilon) {
            cell.z += step.z;
            t_max.z += t_delta.z;
        }
    }

    if (ray.endpoint_inside != 0u && ray.distance <= kRayMaxDistance) {
        ulong endpoint_key;
        if (point_key_from_origin(
                endpoint, kRayVoxelSize, core_minimum.xyz, endpoint_key)) {
            uint endpoint_slot = find_key(keys, hash_mask, endpoint_key);
            if (endpoint_slot != UINT_MAX) {
                atomic_add_float(&score_bits[endpoint_slot], kLogOddsVisit);
            }
        }
    }
}

kernel void classify_points(
    device const float4 *points [[buffer(0)]],
    device const ulong *keys [[buffer(1)]],
    device atomic_uint *score_bits [[buffer(2)]],
    device uchar *noise_mask [[buffer(3)]],
    device float *point_scores [[buffer(4)]],
    constant uint &point_count [[buffer(5)]],
    constant uint &hash_mask [[buffer(6)]],
    constant float4 &grid_origin [[buffer(7)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= point_count) {
        return;
    }

    ulong key;
    if (!point_key_from_origin(
            points[index].xyz, kRayVoxelSize, grid_origin.xyz, key)) {
        noise_mask[index] = uchar(0);
        point_scores[index] = 0.0f;
        return;
    }
    uint slot = find_key(keys, hash_mask, key);
    if (slot == UINT_MAX) {
        noise_mask[index] = uchar(0);
        point_scores[index] = 0.0f;
        return;
    }

    float score = as_type<float>(atomic_load_explicit(
        &score_bits[slot], memory_order_relaxed
    ));
    point_scores[index] = score;
    // This is a direct translation of Studio's noiseThresholdKernel: values
    // indistinguishable from zero are retained, otherwise score > 0.2 is noise.
    noise_mask[index] = abs(score) < kEpsilon
        ? uchar(0)
        : uchar(score > kNoiseThreshold);
}

inline uint representative_at_cell(
    device const ulong *keys,
    device const uint *point_indices,
    uint hash_mask,
    int3 cell,
    uint context_id
) {
    ulong key;
    if (!cell_key(cell, key)) {
        return UINT_MAX;
    }
    key = contextual_key(key, context_id);
    uint slot = find_key(keys, hash_mask, key);
    return slot == UINT_MAX ? UINT_MAX : point_indices[slot];
}

kernel void temporal_consistency(
    device const float4 *points [[buffer(0)]],
    device const float *times [[buffer(1)]],
    device const uchar *levels [[buffer(2)]],
    device const uchar *required_counts [[buffer(3)]],
    device const ulong *keys [[buffer(4)]],
    device const uint *point_indices [[buffer(5)]],
    device const uint *context_ids [[buffer(6)]],
    device uchar *remove_mask [[buffer(7)]],
    device const float4 *context_centers [[buffer(8)]],
    constant uint &point_count [[buffer(9)]],
    constant uint &hash_mask [[buffer(10)]],
    constant float &voxel_size [[buffer(11)]],
    constant uint &target_level [[buffer(12)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= point_count || uint(levels[index]) != target_level) {
        return;
    }
    uint required = uint(required_counts[index]);
    if (required == 0u) {
        return;
    }
    uint context_id = context_ids[index];
    int3 center = smoothing_cell(
        points[index].xyz, voxel_size, context_centers[context_id].xyz
    );
    uint diverse = 0u;
    for (int dx = -4; dx <= 4 && diverse < required; ++dx) {
        for (int dy = -4; dy <= 4 && diverse < required; ++dy) {
            for (int dz = -4; dz <= 4; ++dz) {
                uint other = representative_at_cell(
                    keys, point_indices, hash_mask, center + int3(dx, dy, dz),
                    context_id
                );
                if (other != UINT_MAX && abs(times[index] - times[other]) > 0.09f) {
                    ++diverse;
                    if (diverse >= required) break;
                }
            }
        }
    }
    if (diverse < required) {
        remove_mask[index] = uchar(1);
    }
}

inline float3 smallest_eigenvector(
    float cxx, float cxy, float cxz,
    float cyy, float cyz, float czz
) {
    // ShareFilter's CUDA is an unrolled Eigen-style self-adjoint Jacobi
    // solver: pick the largest off-diagonal entry, rotate, and repeat at most
    // eight times.  Keep row-major matrices here so the updates mirror the
    // recovered PTX directly.  V stores eigenvectors in its columns.
    thread float a[9] = {
        cxx, cxy, cxz,
        cxy, cyy, cyz,
        cxz, cyz, czz,
    };
    thread float v[9] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };
    constexpr float precision = 1.0e-7f;
    constexpr float half_pi = 1.57079632679489662f;

    for (uint iteration = 0; iteration < 8u; ++iteration) {
        uint p = 0u;
        uint q = 1u;
        float largest = abs(a[1]);
        float xz = abs(a[2]);
        if (xz > largest) {
            p = 0u;
            q = 2u;
            largest = xz;
        }
        float yz = abs(a[5]);
        if (yz > largest) {
            p = 1u;
            q = 2u;
            largest = yz;
        }
        if (largest < precision) break;

        float app = a[p * 3u + p];
        float aqq = a[q * 3u + q];
        float apq = a[q * 3u + p];
        float diagonal_difference = aqq - app;
        float angle;
        // These two branches, including the half-pi near-equal-diagonal case,
        // are present verbatim in the generated CUDA PTX.
        if (abs(diagonal_difference) < precision) {
            angle = half_pi;
        } else if (abs(apq) < precision) {
            angle = 0.0f;
        } else {
            angle = 0.5f * atan2(-2.0f * apq, diagonal_difference);
        }
        float sine = sin(angle);
        float cosine = cos(angle);

        for (uint k = 0u; k < 3u; ++k) {
            if (k == p || k == q) continue;
            float apk = a[p * 3u + k];
            float aqk = a[q * 3u + k];
            float new_apk = cosine * apk + sine * aqk;
            float new_aqk = cosine * aqk - sine * apk;
            a[p * 3u + k] = new_apk;
            a[k * 3u + p] = new_apk;
            a[q * 3u + k] = new_aqk;
            a[k * 3u + q] = new_aqk;
        }

        float cosine_squared = cosine * cosine;
        float sine_squared = sine * sine;
        float twice_product = 2.0f * cosine * sine * apq;
        a[p * 3u + p] = aqq * sine_squared + app * cosine_squared
            + twice_product;
        a[q * 3u + q] = aqq * cosine_squared + app * sine_squared
            - twice_product;
        float new_apq = (aqq - app) * cosine * sine
            + apq * (cosine_squared - sine_squared);
        a[p * 3u + q] = new_apq;
        a[q * 3u + p] = new_apq;

        for (uint row = 0u; row < 3u; ++row) {
            float vip = v[row * 3u + p];
            float viq = v[row * 3u + q];
            v[row * 3u + p] = cosine * vip + sine * viq;
            v[row * 3u + q] = cosine * viq - sine * vip;
        }
    }

    uint minimum = a[0] > a[4] ? 1u : 0u;
    if (a[minimum * 3u + minimum] > a[8]) minimum = 2u;
    return float3(v[minimum], v[3u + minimum], v[6u + minimum]);
}

kernel void calculate_normals(
    device const float4 *points [[buffer(0)]],
    device const uchar *levels [[buffer(1)]],
    device const ulong *keys [[buffer(2)]],
    device const uint *point_indices [[buffer(3)]],
    device const uint *context_ids [[buffer(4)]],
    device const float4 *context_centers [[buffer(5)]],
    device float4 *normals [[buffer(6)]],
    constant uint &point_count [[buffer(7)]],
    constant uint &hash_mask [[buffer(8)]],
    constant float &voxel_size [[buffer(9)]],
    constant uint &target_level [[buffer(10)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= point_count || uint(levels[index]) != target_level) return;
    uint context_id = context_ids[index];
    int3 center = smoothing_cell(
        points[index].xyz, voxel_size, context_centers[context_id].xyz
    );
    float3 mean = float3(0.0f);
    float cxx = 0.0f, cxy = 0.0f, cxz = 0.0f;
    float cyy = 0.0f, cyz = 0.0f, czz = 0.0f;
    uint count = 0u;
    for (int dx = -3; dx <= 3; ++dx) {
        for (int dy = -3; dy <= 3; ++dy) {
            for (int dz = -3; dz <= 3; ++dz) {
                uint other = representative_at_cell(
                    keys, point_indices, hash_mask, center + int3(dx, dy, dz),
                    context_id
                );
                if (other == UINT_MAX) continue;
                float3 sample = points[other].xyz;
                ++count;
                float3 delta = sample - mean;
                mean += delta / float(count);
                float3 centered = sample - mean;
                float inverse_count = 1.0f / float(count);
                // ShareFilter updates each covariance entry as a running mean
                // of products measured from the newly updated centroid.
                cxx += (centered.x * centered.x - cxx) * inverse_count;
                cxy += (centered.y * centered.x - cxy) * inverse_count;
                cxz += (centered.z * centered.x - cxz) * inverse_count;
                cyy += (centered.y * centered.y - cyy) * inverse_count;
                cyz += (centered.z * centered.y - cyz) * inverse_count;
                czz += (centered.z * centered.z - czz) * inverse_count;
            }
        }
    }
    // calculateNormalKernel uses count > valid_neighbor; the MID360 profile's
    // valid_neighbor is two, so exactly two samples are insufficient.
    if (count <= 2u) {
        normals[index] = float4(0.0f);
        return;
    }
    normals[index] = float4(smallest_eigenvector(cxx, cxy, cxz, cyy, cyz, czz), 1.0f);
}

kernel void smooth_normals(
    device const float4 *points [[buffer(0)]],
    device const float4 *input_normals [[buffer(1)]],
    device const uchar *levels [[buffer(2)]],
    device const ulong *keys [[buffer(3)]],
    device const uint *point_indices [[buffer(4)]],
    device const uint *context_ids [[buffer(5)]],
    device const float4 *context_centers [[buffer(6)]],
    device float4 *output_normals [[buffer(7)]],
    constant uint &point_count [[buffer(8)]],
    constant uint &hash_mask [[buffer(9)]],
    constant float &voxel_size [[buffer(10)]],
    constant uint &target_level [[buffer(11)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= point_count || uint(levels[index]) != target_level) return;
    float4 source = input_normals[index];
    if (source.w < 1.0f) {
        output_normals[index] = source;
        return;
    }
    constexpr float cosine_limit = 0.8660254037844386f;
    float3 base = normalize(source.xyz);
    float3 sum = float3(0.0f);
    uint count = 0u;
    uint context_id = context_ids[index];
    int3 center = smoothing_cell(
        points[index].xyz, voxel_size, context_centers[context_id].xyz
    );
    for (int dx = -5; dx <= 5; ++dx) {
        for (int dy = -5; dy <= 5; ++dy) {
            for (int dz = -5; dz <= 5; ++dz) {
                uint other = representative_at_cell(
                    keys, point_indices, hash_mask, center + int3(dx, dy, dz),
                    context_id
                );
                if (other == UINT_MAX || input_normals[other].w < 1.0f) continue;
                float3 candidate = normalize(input_normals[other].xyz);
                float alignment = dot(base, candidate);
                float weight = abs(alignment);
                if (weight < cosine_limit) continue;
                float3 oriented = alignment < 0.0f ? -candidate : candidate;
                sum += oriented * weight;
                ++count;
            }
        }
    }
    output_normals[index] = count >= 2u && dot(sum, sum) > 1.0e-12f
        ? float4(normalize(sum), 1.0f)
        : source;
}

kernel void normal_mls(
    device const float4 *points [[buffer(0)]],
    device const float4 *normals [[buffer(1)]],
    device const uchar *levels [[buffer(2)]],
    device const ulong *keys [[buffer(3)]],
    device const uint *point_indices [[buffer(4)]],
    device const uint *context_ids [[buffer(5)]],
    device const float4 *context_centers [[buffer(6)]],
    device float4 *output_points [[buffer(7)]],
    device float4 *output_normals [[buffer(8)]],
    constant uint &point_count [[buffer(9)]],
    constant uint &hash_mask [[buffer(10)]],
    constant float &voxel_size [[buffer(11)]],
    constant uint &target_level [[buffer(12)]],
    constant uint &invalidate_on_failure [[buffer(13)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= point_count || uint(levels[index]) != target_level) return;
    float4 original = points[index];
    float4 source_normal = normals[index];
    constexpr float cosine_limit = 0.8660254037844386f;
    float normal_length_squared = dot(source_normal.xyz, source_normal.xyz);
    float3 base = normal_length_squared > 1.0e-12f
        ? normalize(source_normal.xyz)
        : float3(0.0f);
    float3 centroid = float3(0.0f);
    float cxx = 0.0f, cxy = 0.0f, cxz = 0.0f;
    float cyy = 0.0f, cyz = 0.0f, czz = 0.0f;
    uint count = 0u;
    uint context_id = context_ids[index];
    int3 center = smoothing_cell(
        original.xyz, voxel_size, context_centers[context_id].xyz
    );
    for (int dx = -4; dx <= 4; ++dx) {
        for (int dy = -4; dy <= 4; ++dy) {
            for (int dz = -4; dz <= 4; ++dz) {
                uint other = representative_at_cell(
                    keys, point_indices, hash_mask, center + int3(dx, dy, dz),
                    context_id
                );
                if (other == UINT_MAX || normals[other].w < 1.0f) continue;
                float3 candidate_normal = normalize(normals[other].xyz);
                float alignment = dot(base, candidate_normal);
                if (abs(alignment) < cosine_limit) continue;
                ++count;
                float3 sample = points[other].xyz;
                centroid += (sample - centroid) / float(count);
                float3 centered = sample - centroid;
                float inverse_count = 1.0f / float(count);
                cxx += (centered.x * centered.x - cxx) * inverse_count;
                cxy += (centered.y * centered.x - cxy) * inverse_count;
                cxz += (centered.z * centered.x - cxz) * inverse_count;
                cyy += (centered.y * centered.y - cyy) * inverse_count;
                cyz += (centered.z * centered.y - cyz) * inverse_count;
                czz += (centered.z * centered.z - czz) * inverse_count;
            }
        }
    }
    // normalMLSKernel checks neighbor_count > valid_neighbor, with
    // valid_neighbor=1 in the MID360 profile.
    if (count > 1u) {
        float3 plane_normal = smallest_eigenvector(cxx, cxy, cxz, cyy, cyz, czz);
        float distance = dot(centroid - original.xyz, plane_normal);
        output_points[index] = float4(original.xyz + plane_normal * distance, original.w);
        output_normals[index] = float4(plane_normal, 1.0f);
        return;
    }

    // Sparse-neighborhood fallback: Studio uses the unqualified radius-one
    // centroid when it contains more than two representatives.  If that also
    // fails during the first MLS iteration, its kernel clears both validity
    // flags.  Preserve point.w (intensity) and use a negative normal.w as the
    // equivalent out-of-band removal marker for the host.
    centroid = float3(0.0f);
    count = 0u;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                uint other = representative_at_cell(
                    keys, point_indices, hash_mask, center + int3(dx, dy, dz),
                    context_id
                );
                if (other == UINT_MAX) continue;
                ++count;
                centroid += (points[other].xyz - centroid) / float(count);
            }
        }
    }
    if (count > 2u) {
        output_points[index] = float4(centroid, original.w);
        return;
    }
    if (invalidate_on_failure != 0u) {
        output_normals[index].w = -1.0f;
    }
}

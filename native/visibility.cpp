#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr uint8_t RELIABLE = 1u;
constexpr uint8_t SURFACE_REJECTED = 2u;
constexpr uint8_t VISIBLE = 4u;
constexpr uint8_t USABLE = 8u;

inline float dot3(const float *a, float x, float y, float z) {
    float value = 0.0f;
    value += a[0] * x;
    value += a[1] * y;
    value += a[2] * z;
    return value;
}

}  // namespace

extern "C" uint32_t s20_visibility_abi_version() { return 5u; }

extern "C" int s20_sort_count(
    float *observations,
    uint64_t point_count,
    uint32_t photo_count,
    uint64_t *per_photo
) {
    if (!observations || !per_photo || photo_count == 0u) return 1;
    for (uint64_t point = 0; point < point_count; ++point) {
        float *records = observations + size_t(point) * 4u * 8u;
        uint32_t order[4] = {0u, 1u, 2u, 3u};
        for (uint32_t index = 1; index < 4u; ++index) {
            const uint32_t current = order[index];
            const float current_score = records[size_t(current) * 8u + 7u];
            uint32_t destination = index;
            while (destination > 0u) {
                const float previous_score = records[size_t(order[destination - 1u]) * 8u + 7u];
                const bool move_left = !std::isnan(current_score) &&
                    (std::isnan(previous_score) || current_score > previous_score);
                if (!move_left) break;
                order[destination] = order[destination - 1u];
                --destination;
            }
            order[destination] = current;
        }
        if (order[0] != 0u || order[1] != 1u || order[2] != 2u || order[3] != 3u) {
            float copy[4u * 8u];
            std::copy_n(records, 4u * 8u, copy);
            for (uint32_t destination = 0; destination < 4u; ++destination) {
                std::copy_n(copy + size_t(order[destination]) * 8u, 8u,
                            records + size_t(destination) * 8u);
            }
        }
        for (uint32_t candidate = 0; candidate < 4u; ++candidate) {
            const float *record = records + size_t(candidate) * 8u;
            if (!(record[7] > 0.0f)) continue;
            const float photo_value = record[6];
            if (!std::isfinite(photo_value) || photo_value < 0.0f ||
                double(photo_value) >= double(photo_count)) {
                return 2;
            }
            ++per_photo[static_cast<uint32_t>(photo_value)];
        }
    }
    return 0;
}

extern "C" int s20_pack_slots(
    float *observations,
    uint64_t point_count,
    const void *slot_ids,
    uint64_t slot_count,
    uint32_t slot_bits,
    const uint8_t *image,
    uint32_t image_width,
    uint32_t image_height,
    uint32_t grid_width,
    uint32_t grid_height
) {
    if (!observations || !image || image_width < 2u || image_height < 2u || grid_width == 0u ||
        grid_height == 0u || (slot_bits != 32u && slot_bits != 64u) ||
        (slot_count > 0 && !slot_ids)) {
        return 1;
    }
    const auto *slots32 = static_cast<const uint32_t *>(slot_ids);
    const auto *slots64 = static_cast<const uint64_t *>(slot_ids);
    const uint64_t total_slots = point_count * 4u;
    for (uint64_t index = 0; index < slot_count; ++index) {
        const uint64_t flat_slot = slot_bits == 32u ? uint64_t(slots32[index]) : slots64[index];
        if (flat_slot >= total_slots) return 2;
        float *record = observations + size_t(flat_slot) * 8u;
        const float u = record[0];
        const float v = record[1];
        if (!std::isfinite(u) || !std::isfinite(v) || u < 0.0f || v < 0.0f ||
            u >= float(image_width - 1u) || v >= float(image_height - 1u)) {
            return 3;
        }
        const int32_t x = static_cast<int32_t>(u);
        const int32_t y = static_cast<int32_t>(v);
        const size_t pixel = (size_t(y) * image_width + uint32_t(x)) * 3u;
        const uint8_t *p00 = image + pixel;
        const uint8_t *p10 = p00 + 3u;
        const uint8_t *p01 = p00 + size_t(image_width) * 3u;
        const uint8_t *p11 = p01 + 3u;
        const double a = double(u) - double(x);
        const double b = double(v) - double(y);
        const double one_minus_a = 1.0 - a;
        const double one_minus_b = 1.0 - b;
        for (uint32_t channel = 0; channel < 3u; ++channel) {
            const double t00 = (one_minus_a * one_minus_b) * double(p00[channel]);
            const double t10 = (a * one_minus_b) * double(p10[channel]);
            const double t01 = (one_minus_a * b) * double(p01[channel]);
            const double t11 = (a * b) * double(p11[channel]);
            record[channel] = float(((t00 + t10) + t01) + t11);
        }
        int gradient = 0;
        for (uint32_t channel = 0; channel < 3u; ++channel) {
            const int horizontal = std::abs(int(p10[channel]) - int(p00[channel]));
            const int vertical = std::abs(int(p01[channel]) - int(p00[channel]));
            gradient = std::max(gradient, std::max(horizontal, vertical));
        }
        record[3] = float(gradient);
        float grid_u = u / float(image_width);
        grid_u = grid_u * float(grid_width);
        grid_u = grid_u - 0.5f;
        record[4] = std::clamp(grid_u, 0.0f, float(grid_width - 1u));
        float grid_v = v / float(image_height);
        grid_v = grid_v * float(grid_height);
        grid_v = grid_v - 0.5f;
        record[5] = std::clamp(grid_v, 0.0f, float(grid_height - 1u));
    }
    return 0;
}

extern "C" int s20_bucket_slots(
    const float *observations,
    uint64_t point_count,
    uint32_t photo_count,
    const uint64_t *offsets,
    uint64_t *cursors,
    void *slot_ids,
    uint32_t slot_bits
) {
    if (!observations || !offsets || !cursors || !slot_ids || photo_count == 0 ||
        (slot_bits != 32u && slot_bits != 64u)) {
        return 1;
    }
    auto *slots32 = static_cast<uint32_t *>(slot_ids);
    auto *slots64 = static_cast<uint64_t *>(slot_ids);
    for (uint64_t point = 0; point < point_count; ++point) {
        const float *records = observations + size_t(point) * 4u * 8u;
        for (uint32_t candidate = 0; candidate < 4; ++candidate) {
            const float *record = records + size_t(candidate) * 8u;
            if (!(record[7] > 0.0f)) continue;
            const float photo_value = record[6];
            if (!std::isfinite(photo_value) || photo_value < 0.0f ||
                double(photo_value) >= double(photo_count)) {
                return 2;
            }
            const uint32_t photo = static_cast<uint32_t>(photo_value);
            uint64_t destination = cursors[photo]++;
            if (destination >= offsets[photo + 1u]) return 3;
            const uint64_t flat_slot = point * 4u + candidate;
            if (slot_bits == 32u) {
                if (flat_slot > UINT32_MAX) return 4;
                slots32[destination] = static_cast<uint32_t>(flat_slot);
            } else {
                slots64[destination] = flat_slot;
            }
        }
    }
    return 0;
}

extern "C" int s20_rank_insert(
    float *observations,
    uint64_t point_count,
    const uint32_t *point_ids,
    const float *u,
    const float *v,
    const float *scores,
    uint64_t count,
    uint32_t photo,
    uint64_t *inserted
) {
    if (!observations || !inserted || (count > 0 && (!point_ids || !u || !v || !scores))) {
        return 1;
    }
    uint64_t accepted = 0;
    const float photo_value = static_cast<float>(photo);
    for (uint64_t index = 0; index < count; ++index) {
        const uint32_t point_id = point_ids[index];
        if (point_id >= point_count) return 2;
        float *point = observations + size_t(point_id) * 4u * 8u;
        uint32_t slot = 0;
        float minimum = point[7];
        if (!std::isnan(minimum)) {
            for (uint32_t candidate = 1; candidate < 4; ++candidate) {
                const float candidate_score = point[size_t(candidate) * 8u + 7u];
                if (std::isnan(candidate_score)) {
                    minimum = candidate_score;
                    slot = candidate;
                    break;
                }
                if (candidate_score < minimum) {
                    minimum = candidate_score;
                    slot = candidate;
                }
            }
        }
        if (scores[index] > minimum) {
            float *destination = point + size_t(slot) * 8u;
            destination[0] = u[index];
            destination[1] = v[index];
            destination[6] = photo_value;
            destination[7] = scores[index];
            ++accepted;
        }
    }
    *inserted = accepted;
    return 0;
}

extern "C" int s20_visibility_decide(
    const float *points,
    const float *normals,
    uint64_t point_count,
    const uint32_t *point_ids,
    const float *u,
    const float *v,
    const float *distance,
    const uint64_t *blocker_keys,
    const uint64_t *exact_keys,
    uint64_t count,
    const float *center32,
    const double *center64,
    const uint8_t *mask,
    uint32_t mask_width,
    uint32_t mask_height,
    uint8_t *flags,
    float *incidence,
    uint64_t *denominator_shortcuts,
    uint64_t *plane_shortcuts
) {
    if (!points || !normals || !center32 || !center64 || !mask || !flags || !incidence ||
        !denominator_shortcuts || !plane_shortcuts || point_count == 0 || mask_width < 2 ||
        mask_height < 2) {
        return 1;
    }
    if (count > 0 &&
        (!point_ids || !u || !v || !distance || !blocker_keys || !exact_keys)) {
        return 1;
    }

    uint64_t denominator_skips = 0;
    uint64_t plane_skips = 0;
    for (uint64_t index = 0; index < count; ++index) {
        const uint32_t point_id = point_ids[index];
        if (point_id >= point_count) return 2;
        const uint64_t blocker_id64 = blocker_keys[index] % point_count;
        if (blocker_id64 > UINT32_MAX) return 2;
        const uint32_t blocker_id = static_cast<uint32_t>(blocker_id64);
        const float *point = points + size_t(point_id) * 3u;
        const float *normal = normals + size_t(point_id) * 3u;
        const float *blocker = points + size_t(blocker_id) * 3u;
        const float *blocker_normal = normals + size_t(blocker_id) * 3u;
        const float d = distance[index];
        if (!std::isfinite(u[index]) || !std::isfinite(v[index]) || u[index] < 0.0f ||
            v[index] < 0.0f || u[index] >= float(mask_width - 1u) ||
            v[index] >= float(mask_height - 1u)) {
            return 3;
        }

        const float ray_x = (point[0] - center32[0]) / d;
        const float ray_y = (point[1] - center32[1]) / d;
        const float ray_z = (point[2] - center32[2]) / d;
        const float denominator = dot3(blocker_normal, ray_x, ray_y, ray_z);
        const float point_incidence = dot3(normal, -ray_x, -ray_y, -ray_z);
        incidence[index] = point_incidence;

        bool reliable = false;
        bool surface_rejected = false;
        if (std::fabs(denominator) > 0.15f) {
            double numerator = double(blocker_normal[0]) * (double(blocker[0]) - center64[0]);
            numerator += double(blocker_normal[1]) * (double(blocker[1]) - center64[1]);
            numerator += double(blocker_normal[2]) * (double(blocker[2]) - center64[2]);
            const double plane_depth = numerator / double(denominator);
            if (plane_depth > 0.1) {
                const double dx = center64[0] + double(ray_x) * plane_depth - double(blocker[0]);
                const double dy = center64[1] + double(ray_y) * plane_depth - double(blocker[1]);
                const double dz = center64[2] + double(ray_z) * plane_depth - double(blocker[2]);
                double squared_distance = dx * dx;
                squared_distance += dy * dy;
                squared_distance += dz * dz;
                reliable = std::sqrt(squared_distance) < 0.04;
                surface_rejected = reliable && double(d) > plane_depth + 0.02;
            } else {
                ++plane_skips;
            }
        } else {
            ++denominator_skips;
        }

        const double exact_depth = double(exact_keys[index] / point_count) / 1e6;
        const float relative_tolerance = 0.005f * d;
        bool visible = double(d) <= exact_depth + 0.025 + double(relative_tolerance);
        visible = visible && (!reliable || !surface_rejected) && point_incidence > 0.05f;
        bool usable = false;
        if (visible) {
            const int32_t x = static_cast<int32_t>(u[index]);
            const int32_t y = static_cast<int32_t>(v[index]);
            const size_t pixel = size_t(y) * mask_width + uint32_t(x);
            usable = mask[pixel] == 0 && mask[pixel + 1] == 0 &&
                     mask[pixel + mask_width] == 0 && mask[pixel + mask_width + 1] == 0;
        }
        flags[index] = (reliable ? RELIABLE : 0u) |
                       (surface_rejected ? SURFACE_REJECTED : 0u) |
                       (visible ? VISIBLE : 0u) | (usable ? USABLE : 0u);
    }
    *denominator_shortcuts = denominator_skips;
    *plane_shortcuts = plane_skips;
    return 0;
}

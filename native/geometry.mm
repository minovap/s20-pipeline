#include "stage_profile.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr float kRayVoxelSize = 0.015f;
constexpr float kOutputVoxelSize = 0.005f;
constexpr float kBlockSize = 1.0f;
constexpr float kSubfileSize = 50.0f;
constexpr float kSubfileHalo = 0.05f;
constexpr size_t kMinimumRaySubfilePoints = 64;
constexpr std::array<float, 4> kSmoothVoxelSizes = {0.005f, 0.010f, 0.020f, 0.040f};
constexpr std::array<float, 3> kDensityThresholds = {0.005f, 0.010f, 0.020f};
constexpr int kStudioTrimFrames = 100;
constexpr int kStudioTrimBypassFrames = 300;

struct alignas(16) Float4 {
    float x;
    float y;
    float z;
    float w;
};

struct RayRecord {
    float origin_x;
    float origin_y;
    float origin_z;
    float endpoint_x;
    float endpoint_y;
    float endpoint_z;
    float distance;
    float start_distance;
    uint32_t endpoint_inside;
    uint32_t reserved;
};

static_assert(sizeof(RayRecord) == 40, "Studio Ray records are forty bytes");

struct Pose {
    bool present = false;
    double timestamp = 0.0;
    double tx = 0.0;
    double ty = 0.0;
    double tz = 0.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    double qw = 1.0;
};

struct Cloud {
    std::vector<Float4> points;
    std::vector<uint32_t> frame_ids;
    std::vector<float> relative_times;
    std::vector<Float4> origins;
    std::vector<double> frame_timestamps;
    size_t frames_loaded = 0;
    int first_frame = -1;
    int last_frame = -1;
    bool studio_auto_trim = false;
};

bool segment_intersects_box_cpu(
    const Float4 &origin,
    const Float4 &endpoint,
    const Float4 &box_minimum,
    const Float4 &box_maximum,
    float &entry,
    float &exit
) {
    const float origins[3] = {origin.x, origin.y, origin.z};
    const float endpoints[3] = {endpoint.x, endpoint.y, endpoint.z};
    const float minima[3] = {box_minimum.x, box_minimum.y, box_minimum.z};
    const float maxima[3] = {box_maximum.x, box_maximum.y, box_maximum.z};
    entry = 0.0f;
    exit = 1.0f;
    for (int axis = 0; axis < 3; ++axis) {
        float delta = endpoints[axis] - origins[axis];
        // The host slabTest uses the 1e-12 constant at DAT_1800e8a04.
        if (std::abs(delta) < 1.0e-12f) {
            if (origins[axis] < minima[axis] || origins[axis] > maxima[axis]) {
                return false;
            }
            continue;
        }
        float inverse = 1.0f / delta;
        float first = (minima[axis] - origins[axis]) * inverse;
        float second = (maxima[axis] - origins[axis]) * inverse;
        float near_value = std::min(first, second);
        float far_value = std::max(first, second);
        entry = std::max(entry, near_value);
        exit = std::min(exit, far_value);
        if (entry > exit) {
            return false;
        }
    }
    return true;
}

bool point_inside_box_cpu(
    const Float4 &point,
    const Float4 &box_minimum,
    const Float4 &box_maximum
) {
    return box_minimum.x <= point.x && point.x <= box_maximum.x
        && box_minimum.y <= point.y && point.y <= box_maximum.y
        && box_minimum.z <= point.z && point.z <= box_maximum.z;
}

// Ray records for source points [begin, end). Called per batch so the
// forty-byte records never exist for the whole cloud at once.
void prepare_ray_records(
    const Cloud &source,
    size_t begin,
    size_t end,
    const Float4 &core_minimum,
    const Float4 &core_maximum,
    bool clip_to_subfile,
    std::vector<RayRecord> &records
) {
    records.clear();
    Float4 expanded_minimum = {
        core_minimum.x - 0.06f,
        core_minimum.y - 0.06f,
        core_minimum.z - 0.06f,
        0.0f,
    };
    Float4 expanded_maximum = {
        core_maximum.x + 0.06f,
        core_maximum.y + 0.06f,
        core_maximum.z + 0.06f,
        0.0f,
    };
    for (size_t index = begin; index < end; ++index) {
        const Float4 &endpoint = source.points[index];
        const Float4 &origin = source.origins[source.frame_ids[index]];
        if (!std::isfinite(endpoint.x) || !std::isfinite(endpoint.y)
            || !std::isfinite(endpoint.z) || !std::isfinite(origin.x)
            || !std::isfinite(origin.y) || !std::isfinite(origin.z)) {
            continue;
        }
        float dx = endpoint.x - origin.x;
        float dy = endpoint.y - origin.y;
        float dz = endpoint.z - origin.z;
        float squared_distance = dy * dy + dz * dz + dx * dx;
        // prepareRays accepts sensor ranges from 0.1 m through 200 m.
        if (!(squared_distance >= 0.01f && squared_distance <= 40000.0f)) {
            continue;
        }
        float distance = std::sqrt(squared_distance);
        float start_distance = 0.0f;
        uint32_t endpoint_inside = 1u;
        if (clip_to_subfile) {
            float expanded_entry = 0.0f;
            float expanded_exit = 0.0f;
            if (!segment_intersects_box_cpu(
                    origin, endpoint, expanded_minimum, expanded_maximum,
                    expanded_entry, expanded_exit)) {
                continue;
            }
            float core_entry = 0.0f;
            float core_exit = 0.0f;
            if (segment_intersects_box_cpu(
                    origin, endpoint, core_minimum, core_maximum,
                    core_entry, core_exit)) {
                start_distance = std::max(core_entry, 0.0f) * distance;
            }
            endpoint_inside = point_inside_box_cpu(
                endpoint, core_minimum, core_maximum
            ) ? 1u : 0u;
        }
        records.push_back({
            origin.x, origin.y, origin.z,
            endpoint.x, endpoint.y, endpoint.z,
            distance, start_distance, endpoint_inside, 0u,
        });
    }
}

// Source points per ray batch: 16 M points is at most 640 MB of records.
constexpr size_t kRayBatchPoints = 16u << 20;

struct FilterResult {
    std::vector<uint8_t> noise_mask;
    std::vector<uint8_t> keep_mask;
    std::vector<float> point_scores;
    std::vector<Float4> normals;
    uint64_t input_points = 0;
    uint64_t pre_smooth_points = 0;
    uint64_t expanded_context_points = 0;
    uint64_t sparse_block_removed_points = 0;
    uint64_t temporal_removed_points = 0;
    uint64_t mls_context_removed_points = 0;
    uint64_t mls_removed_points = 0;
    uint64_t occupied_voxels = 0;
    uint64_t noise_points = 0;
    uint64_t kept_points = 0;
    uint64_t ray_subfile_input_points = 0;
    uint64_t sparse_ray_subfile_points = 0;
    uint32_t ray_subfile_count = 0;
    double insert_seconds = 0.0;
    double ray_seconds = 0.0;
    double classify_seconds = 0.0;
    double dedup_seconds = 0.0;
    double density_seconds = 0.0;
    double temporal_seconds = 0.0;
    double smoothing_seconds = 0.0;
    uint64_t hash_capacity = 0;
    std::array<uint64_t, 4> density_level_counts = {0, 0, 0, 0};
    std::string device_name;
};

struct Options {
    fs::path input;
    fs::path output;
    fs::path kernels;
    int frame_start = -1;
    int frame_end = -1;
    int max_frames = -1;
    bool deduplicate = true;
    bool all_frames = false;
    bool ray_only = false;
    bool self_test = false;
};

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

std::string trim(std::string value) {
    while (!value.empty() && (value.back() == '\r' || value.back() == '\n' || value.back() == ' ')) {
        value.pop_back();
    }
    size_t first = 0;
    while (first < value.size() && value[first] == ' ') {
        ++first;
    }
    return value.substr(first);
}

std::vector<std::string> split_words(const std::string &line) {
    std::istringstream stream(line);
    std::vector<std::string> words;
    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }
    return words;
}

std::vector<Pose> read_poses(const fs::path &path) {
    std::ifstream input(path);
    if (!input) {
        fail("cannot open pose file: " + path.string());
    }

    std::unordered_map<int, Pose> sparse;
    std::string line;
    int maximum_index = -1;
    while (std::getline(input, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream row(line);
        int index = -1;
        Pose pose;
        if (!(row >> index >> pose.timestamp >> pose.tx >> pose.ty >> pose.tz
                  >> pose.qx >> pose.qy >> pose.qz >> pose.qw)) {
            fail("invalid pose row in " + path.string() + ": " + line);
        }
        pose.present = true;
        sparse[index] = pose;
        maximum_index = std::max(maximum_index, index);
    }
    if (maximum_index < 0) {
        fail("pose file contains no poses: " + path.string());
    }

    std::vector<Pose> poses(static_cast<size_t>(maximum_index) + 1);
    for (const auto &[index, pose] : sparse) {
        poses[static_cast<size_t>(index)] = pose;
    }
    return poses;
}

Float4 transform_point(float x, float y, float z, float intensity, const Pose &pose) {
    double qnorm = std::sqrt(
        pose.qx * pose.qx + pose.qy * pose.qy + pose.qz * pose.qz + pose.qw * pose.qw
    );
    if (!(qnorm > 0.0)) {
        fail("encountered a zero-length pose quaternion");
    }
    double qx = pose.qx / qnorm;
    double qy = pose.qy / qnorm;
    double qz = pose.qz / qnorm;
    double qw = pose.qw / qnorm;

    double xx = qx * qx;
    double yy = qy * qy;
    double zz = qz * qz;
    double xy = qx * qy;
    double xz = qx * qz;
    double yz = qy * qz;
    double wx = qw * qx;
    double wy = qw * qy;
    double wz = qw * qz;

    double world_x = (1.0 - 2.0 * (yy + zz)) * x
        + 2.0 * (xy - wz) * y
        + 2.0 * (xz + wy) * z
        + pose.tx;
    double world_y = 2.0 * (xy + wz) * x
        + (1.0 - 2.0 * (xx + zz)) * y
        + 2.0 * (yz - wx) * z
        + pose.ty;
    double world_z = 2.0 * (xz - wy) * x
        + 2.0 * (yz + wx) * y
        + (1.0 - 2.0 * (xx + yy)) * z
        + pose.tz;
    return {
        static_cast<float>(world_x),
        static_cast<float>(world_y),
        static_cast<float>(world_z),
        intensity,
    };
}

template <typename T>
T read_scalar(const char *data) {
    T value;
    std::memcpy(&value, data, sizeof(T));
    return value;
}

struct PcdLayout {
    size_t record_size = 0;
    size_t point_count = 0;
    std::unordered_map<std::string, size_t> offsets;
    std::unordered_map<std::string, size_t> sizes;
};

PcdLayout read_pcd_header(std::ifstream &input, const fs::path &path) {
    std::vector<std::string> fields;
    std::vector<size_t> sizes;
    std::vector<size_t> counts;
    size_t point_count = 0;
    bool binary = false;
    std::string line;
    while (std::getline(input, line)) {
        line = trim(line);
        auto words = split_words(line);
        if (words.empty() || words[0][0] == '#') {
            continue;
        }
        if (words[0] == "FIELDS") {
            fields.assign(words.begin() + 1, words.end());
        } else if (words[0] == "SIZE") {
            for (size_t i = 1; i < words.size(); ++i) {
                sizes.push_back(static_cast<size_t>(std::stoul(words[i])));
            }
        } else if (words[0] == "COUNT") {
            for (size_t i = 1; i < words.size(); ++i) {
                counts.push_back(static_cast<size_t>(std::stoul(words[i])));
            }
        } else if (words[0] == "POINTS" && words.size() == 2) {
            point_count = static_cast<size_t>(std::stoull(words[1]));
        } else if (words[0] == "DATA") {
            binary = words.size() == 2 && words[1] == "binary";
            break;
        }
    }

    if (!binary) {
        fail("only DATA binary PCD is supported: " + path.string());
    }
    if (fields.empty() || fields.size() != sizes.size()) {
        fail("invalid FIELDS/SIZE metadata: " + path.string());
    }
    if (counts.empty()) {
        counts.assign(fields.size(), 1);
    }
    if (counts.size() != fields.size()) {
        fail("invalid COUNT metadata: " + path.string());
    }

    PcdLayout layout;
    layout.point_count = point_count;
    for (size_t i = 0; i < fields.size(); ++i) {
        layout.offsets[fields[i]] = layout.record_size;
        layout.sizes[fields[i]] = sizes[i];
        layout.record_size += sizes[i] * counts[i];
    }
    for (const char *required : {"x", "y", "z", "intensity"}) {
        if (!layout.offsets.count(required) || layout.sizes[required] != sizeof(float)) {
            fail("PCD lacks required float field " + std::string(required) + ": " + path.string());
        }
    }
    return layout;
}

void append_pcd(Cloud &cloud, const fs::path &path, uint32_t frame, const Pose &pose) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail("cannot open scan: " + path.string());
    }
    PcdLayout layout = read_pcd_header(input, path);
    if (layout.record_size == 0) {
        fail("PCD has a zero record size: " + path.string());
    }

    std::vector<char> bytes(layout.record_size * layout.point_count);
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (static_cast<size_t>(input.gcount()) != bytes.size()) {
        fail("truncated PCD payload: " + path.string());
    }

    size_t curvature_offset = 0;
    bool has_curvature = layout.offsets.count("curvature")
        && layout.sizes["curvature"] == sizeof(float);
    if (has_curvature) {
        curvature_offset = layout.offsets["curvature"];
    }

    // Capacity is reserved once by load_cloud from the PCD headers; an exact
    // reserve here would reallocate and copy the whole cloud for every frame.
    for (size_t index = 0; index < layout.point_count; ++index) {
        const char *record = bytes.data() + index * layout.record_size;
        float x = read_scalar<float>(record + layout.offsets["x"]);
        float y = read_scalar<float>(record + layout.offsets["y"]);
        float z = read_scalar<float>(record + layout.offsets["z"]);
        float intensity = read_scalar<float>(record + layout.offsets["intensity"]);
        float relative_time = has_curvature
            ? read_scalar<float>(record + curvature_offset)
            : 0.0f;
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
            continue;
        }
        cloud.points.push_back(transform_point(x, y, z, intensity, pose));
        cloud.frame_ids.push_back(frame);
        cloud.relative_times.push_back(relative_time);
    }
}

Cloud load_cloud(const Options &options) {
    fs::path scan_directory = options.input / "SCANS";
    fs::path pose_path = options.input / "FrameOptPose.txt";
    if (!fs::is_directory(scan_directory)) {
        fail("input has no SCANS directory: " + options.input.string());
    }
    if (!fs::is_regular_file(pose_path)) {
        fail("input has no FrameOptPose.txt: " + options.input.string());
    }

    std::vector<Pose> poses = read_poses(pose_path);
    Cloud cloud;
    cloud.origins.resize(poses.size());
    cloud.frame_timestamps.resize(poses.size());
    for (size_t index = 0; index < poses.size(); ++index) {
        if (poses[index].present) {
            cloud.origins[index] = {
                static_cast<float>(poses[index].tx),
                static_cast<float>(poses[index].ty),
                static_cast<float>(poses[index].tz),
                0.0f,
            };
            cloud.frame_timestamps[index] = poses[index].timestamp;
        }
    }

    std::vector<std::pair<int, fs::path>> scans;
    for (const auto &entry : fs::directory_iterator(scan_directory)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".pcd") {
            continue;
        }
        try {
            int index = std::stoi(entry.path().stem().string());
            scans.emplace_back(index, entry.path());
        } catch (const std::exception &) {
            continue;
        }
    }
    std::sort(scans.begin(), scans.end(), [](const auto &left, const auto &right) {
        return left.first < right.first;
    });

    int selected_start = options.frame_start;
    int selected_end = options.frame_end;
    if (options.all_frames) {
        selected_start = scans.empty() ? 0 : scans.front().first;
        selected_end = scans.empty() ? -1 : scans.back().first;
    } else if (selected_start < 0 && selected_end < 0) {
        selected_start = scans.empty() ? 0 : scans.front().first;
        selected_end = scans.empty() ? -1 : scans.back().first;
        if (static_cast<int>(scans.size()) >= kStudioTrimBypassFrames) {
            selected_start += kStudioTrimFrames;
            selected_end -= kStudioTrimFrames;
            cloud.studio_auto_trim = true;
        }
    } else {
        selected_start = std::max(selected_start, 0);
    }

    // Size the cloud once from the headers so loading never reallocates.
    size_t expected_points = 0;
    size_t expected_frames = 0;
    for (const auto &[frame, path] : scans) {
        if (frame < selected_start || (selected_end >= 0 && frame > selected_end)) continue;
        if (options.max_frames >= 0 && expected_frames >= static_cast<size_t>(options.max_frames)) break;
        std::ifstream input(path, std::ios::binary);
        if (!input) fail("cannot open scan: " + path.string());
        expected_points += read_pcd_header(input, path).point_count;
        ++expected_frames;
    }
    cloud.points.reserve(expected_points);
    cloud.frame_ids.reserve(expected_points);
    cloud.relative_times.reserve(expected_points);

    for (const auto &[frame, path] : scans) {
        if (frame < selected_start) {
            continue;
        }
        if (selected_end >= 0 && frame > selected_end) {
            continue;
        }
        if (options.max_frames >= 0
            && cloud.frames_loaded >= static_cast<size_t>(options.max_frames)) {
            break;
        }
        if (frame < 0 || static_cast<size_t>(frame) >= poses.size() || !poses[frame].present) {
            fail("scan has no matching pose: " + path.string());
        }
        append_pcd(cloud, path, static_cast<uint32_t>(frame), poses[frame]);
        if (cloud.first_frame < 0) {
            cloud.first_frame = frame;
        }
        cloud.last_frame = frame;
        ++cloud.frames_loaded;
        if (cloud.frames_loaded % 25 == 0) {
            std::cerr << "Loaded " << cloud.frames_loaded << " frames, "
                      << cloud.points.size() << " points\n";
        }
    }
    if (cloud.points.empty()) {
        fail("no points selected from input");
    }
    return cloud;
}

uint64_t next_power_of_two(uint64_t value) {
    if (value <= 1) {
        return 1;
    }
    --value;
    for (unsigned shift = 1; shift < 64; shift <<= 1) {
        value |= value >> shift;
    }
    return value + 1;
}

double elapsed_seconds(std::chrono::steady_clock::time_point start);

bool point_key_cpu(const Float4 &point, float voxel_size, uint64_t &key) {
    int64_t x = static_cast<int64_t>(std::floor(point.x / voxel_size));
    int64_t y = static_cast<int64_t>(std::floor(point.y / voxel_size));
    int64_t z = static_cast<int64_t>(std::floor(point.z / voxel_size));
    constexpr int64_t coordinate_limit = (int64_t(1) << 20) - 1;
    if (x < -coordinate_limit || x > coordinate_limit
        || y < -coordinate_limit || y > coordinate_limit
        || z < -coordinate_limit || z > coordinate_limit) {
        key = 0;
        return false;
    }
    constexpr uint64_t coordinate_mask = (uint64_t(1) << 21) - 1;
    uint64_t packed = (static_cast<uint64_t>(x) & coordinate_mask)
        | ((static_cast<uint64_t>(y) & coordinate_mask) << 21)
        | ((static_cast<uint64_t>(z) & coordinate_mask) << 42);
    key = packed + 1;
    return key != 0;
}

bool smoothing_point_key_cpu(
    const Float4 &point,
    float voxel_size,
    const Float4 &context_center,
    uint64_t &key
) {
    // CUDAGridParameters uses int((1.0 + 0.1) / voxel * 0.5) as its
    // positive grid shift.  CUDA's cvt.rzi then truncates the shifted local
    // coordinate toward zero.
    const float shift = static_cast<float>(static_cast<int>(
        ((kBlockSize + 2.0f * kSubfileHalo) / voxel_size) * 0.5f
    ));
    int64_t x = static_cast<int64_t>((point.x - context_center.x) / voxel_size + shift);
    int64_t y = static_cast<int64_t>((point.y - context_center.y) / voxel_size + shift);
    int64_t z = static_cast<int64_t>((point.z - context_center.z) / voxel_size + shift);
    constexpr int64_t coordinate_limit = (int64_t(1) << 20) - 1;
    if (x < -coordinate_limit || x > coordinate_limit
        || y < -coordinate_limit || y > coordinate_limit
        || z < -coordinate_limit || z > coordinate_limit) {
        key = 0;
        return false;
    }
    constexpr uint64_t coordinate_mask = (uint64_t(1) << 21) - 1;
    uint64_t packed = (static_cast<uint64_t>(x) & coordinate_mask)
        | ((static_cast<uint64_t>(y) & coordinate_mask) << 21)
        | ((static_cast<uint64_t>(z) & coordinate_mask) << 42);
    key = packed + 1;
    return key != 0;
}

bool point_key_cpu_from_origin(
    const Float4 &point,
    float voxel_size,
    const Float4 &origin,
    uint64_t &key
) {
    Float4 local = {
        point.x - origin.x,
        point.y - origin.y,
        point.z - origin.z,
        point.w,
    };
    return point_key_cpu(local, voxel_size, key);
}

uint64_t mix_key_cpu(uint64_t key) {
    key ^= key >> 32;
    key *= UINT64_C(0x94d049bb133111eb);
    key ^= key >> 29;
    key *= UINT64_C(0x9fb21c651e98df25);
    key ^= key >> 32;
    return key;
}

uint64_t contextual_key_cpu(uint64_t spatial_key, uint32_t context_id) {
    uint64_t salt = mix_key_cpu(
        static_cast<uint64_t>(context_id) + UINT64_C(0x9e3779b97f4a7c15)
    );
    uint64_t key = mix_key_cpu(spatial_key ^ salt);
    return key == 0 ? UINT64_C(1) : key;
}

struct KeyPoint {
    uint64_t key;
    uint32_t point_index;
};

struct CpuHashTable {
    std::vector<uint64_t> keys;
    std::vector<Float4> representatives;
    uint64_t occupied = 0;
};

CpuHashTable build_ray_hash(const std::vector<Float4> &points, const Float4 &grid_origin) {
    if (points.size() > std::numeric_limits<uint32_t>::max()) {
        fail("too many points for ray hash indexing");
    }
    std::vector<KeyPoint> entries;
    entries.reserve(points.size());
    for (uint32_t index = 0; index < points.size(); ++index) {
        uint64_t key;
        if (point_key_cpu_from_origin(points[index], kRayVoxelSize, grid_origin, key)) {
            entries.push_back({key, index});
        }
    }
    std::sort(entries.begin(), entries.end(), [](const KeyPoint &left, const KeyPoint &right) {
        if (left.key != right.key) {
            return left.key < right.key;
        }
        return left.point_index < right.point_index;
    });

    uint64_t unique_count = 0;
    uint64_t previous = 0;
    for (const KeyPoint &entry : entries) {
        if (unique_count == 0 || entry.key != previous) {
            ++unique_count;
            previous = entry.key;
        }
    }
    uint64_t requested_capacity = std::max<uint64_t>(1024, (unique_count * 10 + 6) / 7);
    uint64_t capacity = next_power_of_two(requested_capacity);
    if (capacity > std::numeric_limits<uint32_t>::max()) {
        fail("required ray hash table is too large");
    }

    CpuHashTable table;
    table.keys.assign(static_cast<size_t>(capacity), 0);
    table.representatives.resize(static_cast<size_t>(capacity));
    uint64_t mask = capacity - 1;
    previous = 0;
    bool have_previous = false;
    for (const KeyPoint &entry : entries) {
        if (have_previous && entry.key == previous) {
            continue;
        }
        have_previous = true;
        previous = entry.key;
        uint64_t hash = mix_key_cpu(entry.key);
        bool inserted = false;
        for (uint32_t probe = 0; probe < 4096; ++probe) {
            uint64_t slot = hash & mask;
            if (table.keys[slot] == 0) {
                table.keys[slot] = entry.key;
                const Float4 &point = points[entry.point_index];
                int64_t cell_x = static_cast<int64_t>(std::floor(
                    (point.x - grid_origin.x) / kRayVoxelSize
                ));
                int64_t cell_y = static_cast<int64_t>(std::floor(
                    (point.y - grid_origin.y) / kRayVoxelSize
                ));
                int64_t cell_z = static_cast<int64_t>(std::floor(
                    (point.z - grid_origin.z) / kRayVoxelSize
                ));
                table.representatives[slot] = {
                    grid_origin.x + (static_cast<float>(cell_x) + 0.5f) * kRayVoxelSize,
                    grid_origin.y + (static_cast<float>(cell_y) + 0.5f) * kRayVoxelSize,
                    grid_origin.z + (static_cast<float>(cell_z) + 0.5f) * kRayVoxelSize,
                    1.0f,
                };
                inserted = true;
                break;
            }
            hash = slot + 1;
        }
        if (!inserted) {
            fail("ray hash exceeded Studio's 4096-probe safety limit");
        }
    }
    table.occupied = unique_count;
    return table;
}

struct CpuIndexHash {
    std::vector<uint64_t> keys;
    std::vector<uint32_t> point_indices;
    uint64_t occupied = 0;
};

CpuIndexHash build_index_hash(
    const std::vector<Float4> &points,
    float voxel_size,
    const std::vector<uint8_t> *levels = nullptr,
    uint8_t target_level = 0,
    const std::vector<uint32_t> *context_ids = nullptr,
    const std::vector<Float4> *context_centers = nullptr
) {
    if (context_ids != nullptr && context_ids->size() != points.size()) {
        fail("internal context-index size mismatch");
    }
    if ((context_ids == nullptr) != (context_centers == nullptr)) {
        fail("smoothing context IDs and centers must be supplied together");
    }
    std::vector<KeyPoint> entries;
    entries.reserve(points.size());
    for (uint32_t index = 0; index < points.size(); ++index) {
        if (levels != nullptr && (*levels)[index] != target_level) {
            continue;
        }
        uint64_t key;
        bool valid = context_ids == nullptr
            ? point_key_cpu(points[index], voxel_size, key)
            : smoothing_point_key_cpu(
                points[index], voxel_size,
                (*context_centers)[(*context_ids)[index]], key
            );
        if (valid) {
            if (context_ids != nullptr) {
                key = contextual_key_cpu(key, (*context_ids)[index]);
            }
            entries.push_back({key, index});
        }
    }
    std::sort(entries.begin(), entries.end(), [](const KeyPoint &left, const KeyPoint &right) {
        if (left.key != right.key) {
            return left.key < right.key;
        }
        return left.point_index < right.point_index;
    });

    uint64_t unique_count = 0;
    uint64_t previous = 0;
    for (const KeyPoint &entry : entries) {
        if (unique_count == 0 || entry.key != previous) {
            ++unique_count;
            previous = entry.key;
        }
    }
    uint64_t capacity = next_power_of_two(std::max<uint64_t>(
        1024, (unique_count * 10 + 6) / 7
    ));
    if (capacity > std::numeric_limits<uint32_t>::max()) {
        fail("required smoothing hash table is too large");
    }

    CpuIndexHash table;
    table.keys.assign(static_cast<size_t>(capacity), 0);
    table.point_indices.assign(static_cast<size_t>(capacity), UINT32_MAX);
    const uint64_t mask = capacity - 1;
    previous = 0;
    bool have_previous = false;
    for (const KeyPoint &entry : entries) {
        if (have_previous && entry.key == previous) {
            continue;
        }
        have_previous = true;
        previous = entry.key;
        uint64_t hash = mix_key_cpu(entry.key);
        bool inserted = false;
        for (uint32_t probe = 0; probe < 4096; ++probe) {
            uint64_t slot = hash & mask;
            if (table.keys[slot] == 0) {
                table.keys[slot] = entry.key;
                table.point_indices[slot] = entry.point_index;
                inserted = true;
                break;
            }
            hash = slot + 1;
        }
        if (!inserted) {
            fail("smoothing hash exceeded Studio's 4096-probe safety limit");
        }
    }
    table.occupied = unique_count;
    return table;
}

void compact_cloud(
    Cloud &cloud,
    const std::vector<uint8_t> &keep,
    std::vector<float> &scores
) {
    if (keep.size() != cloud.points.size() || scores.size() != cloud.points.size()) {
        fail("internal compact-cloud size mismatch");
    }
    size_t retained = static_cast<size_t>(std::count(keep.begin(), keep.end(), uint8_t(1)));
    std::vector<Float4> points;
    std::vector<uint32_t> frame_ids;
    std::vector<float> relative_times;
    std::vector<float> retained_scores;
    points.reserve(retained);
    frame_ids.reserve(retained);
    relative_times.reserve(retained);
    retained_scores.reserve(retained);
    for (size_t index = 0; index < keep.size(); ++index) {
        if (keep[index] == 0) {
            continue;
        }
        points.push_back(cloud.points[index]);
        frame_ids.push_back(cloud.frame_ids[index]);
        relative_times.push_back(cloud.relative_times[index]);
        retained_scores.push_back(scores[index]);
    }
    cloud.points = std::move(points);
    cloud.frame_ids = std::move(frame_ids);
    cloud.relative_times = std::move(relative_times);
    scores = std::move(retained_scores);
}

int32_t studio_downsample_coordinate(float value, float voxel_size) {
    float quotient = value / voxel_size;
    // CloudProcessCPU::downSample subtracts a double-precision one before its
    // truncating conversion whenever the quotient is negative.  This differs
    // from floor only for an exactly integral negative quotient.
    if (quotient < 0.0f) {
        quotient = static_cast<float>(static_cast<double>(quotient) - 1.0);
    }
    return static_cast<int32_t>(quotient);
}

uint32_t studio_voxel_hash(int32_t x, int32_t y, int32_t z) {
    // std::hash<VoxelLoc> in the shipped MSVC binary performs these operations
    // in 32 bits, with the usual modulo-2^32 wraparound.
    uint32_t hash = static_cast<uint32_t>(z) * UINT32_C(115077);
    hash += static_cast<uint32_t>(y);
    hash *= UINT32_C(115077);
    hash += static_cast<uint32_t>(x);
    return hash;
}

struct StudioVoxelSelection {
    uint64_t key;
    uint64_t bucket;
    uint32_t first_index;
    uint32_t best_index;
    uint32_t bucket_first_index;
};

void compact_cloud_in_studio_downsample_order(
    Cloud &cloud,
    const std::vector<uint8_t> &noise_mask,
    std::vector<float> &scores,
    float voxel_size
) {
    if (noise_mask.size() != cloud.points.size() || scores.size() != cloud.points.size()) {
        fail("internal Studio-downsample compact size mismatch");
    }

    std::vector<KeyPoint> entries;
    entries.reserve(cloud.points.size());
    for (uint32_t index = 0; index < cloud.points.size(); ++index) {
        if (noise_mask[index] != 0) continue;
        const Float4 &point = cloud.points[index];
        int32_t x = studio_downsample_coordinate(point.x, voxel_size);
        int32_t y = studio_downsample_coordinate(point.y, voxel_size);
        int32_t z = studio_downsample_coordinate(point.z, voxel_size);
        constexpr uint64_t coordinate_mask = (UINT64_C(1) << 21) - 1;
        uint64_t key = (static_cast<uint64_t>(x) & coordinate_mask)
            | ((static_cast<uint64_t>(y) & coordinate_mask) << 21)
            | ((static_cast<uint64_t>(z) & coordinate_mask) << 42);
        entries.push_back({key, index});
    }
    std::sort(entries.begin(), entries.end(), [](const KeyPoint &left, const KeyPoint &right) {
        if (left.key != right.key) return left.key < right.key;
        return left.point_index < right.point_index;
    });

    // downSample reserves ceil(input_size / 0.7) buckets before insertion;
    // MSVC's _Forced_rehash rounds that request to a power of two.
    float requested_float = std::ceil(
        static_cast<float>(entries.size()) / 0.7f
    );
    uint64_t requested = std::max<uint64_t>(8, static_cast<uint64_t>(requested_float));
    uint64_t bucket_mask = next_power_of_two(requested) - 1;

    std::vector<StudioVoxelSelection> selections;
    selections.reserve(entries.size());
    for (size_t begin = 0; begin < entries.size();) {
        size_t end = begin + 1;
        while (end < entries.size() && entries[end].key == entries[begin].key) ++end;
        uint32_t first_index = entries[begin].point_index;
        uint32_t best_index = first_index;
        const Float4 &first = cloud.points[first_index];
        int32_t x = studio_downsample_coordinate(first.x, voxel_size);
        int32_t y = studio_downsample_coordinate(first.y, voxel_size);
        int32_t z = studio_downsample_coordinate(first.z, voxel_size);
        float center_x = (static_cast<float>(x) + 0.5f) * voxel_size;
        float center_y = (static_cast<float>(y) + 0.5f) * voxel_size;
        float center_z = (static_cast<float>(z) + 0.5f) * voxel_size;
        auto distance_to_center = [&](uint32_t index) {
            float dx = cloud.points[index].x - center_x;
            float dy = cloud.points[index].y - center_y;
            float dz = cloud.points[index].z - center_z;
            return std::sqrt(dx * dx + dy * dy + dz * dz);
        };
        float best_distance = distance_to_center(best_index);
        for (size_t cursor = begin + 1; cursor < end; ++cursor) {
            uint32_t candidate = entries[cursor].point_index;
            float candidate_distance = distance_to_center(candidate);
            // The binary uses scalar sqrt followed by a strict comparison.
            if (candidate_distance < best_distance) {
                best_distance = candidate_distance;
                best_index = candidate;
            }
        }
        uint64_t signed_hash = static_cast<uint64_t>(
            static_cast<int64_t>(static_cast<int32_t>(studio_voxel_hash(x, y, z)))
        );
        selections.push_back({
            entries[begin].key,
            signed_hash & bucket_mask,
            first_index,
            best_index,
            first_index,
        });
        begin = end;
    }

    // With the map pre-reserved, MSVC inserts the first node of every newly
    // encountered bucket at the list front, then appends later collisions to
    // that bucket.  downSample starts at sentinel.prev and follows prev, so its
    // observed order is bucket discovery order and reverse insertion order
    // within each bucket.
    std::sort(selections.begin(), selections.end(), [](const StudioVoxelSelection &left,
                                                        const StudioVoxelSelection &right) {
        if (left.bucket != right.bucket) return left.bucket < right.bucket;
        return left.first_index < right.first_index;
    });
    for (size_t begin = 0; begin < selections.size();) {
        size_t end = begin + 1;
        while (end < selections.size() && selections[end].bucket == selections[begin].bucket) {
            ++end;
        }
        uint32_t bucket_first = selections[begin].first_index;
        for (size_t cursor = begin; cursor < end; ++cursor) {
            selections[cursor].bucket_first_index = bucket_first;
        }
        begin = end;
    }
    std::sort(selections.begin(), selections.end(), [](const StudioVoxelSelection &left,
                                                        const StudioVoxelSelection &right) {
        if (left.bucket_first_index != right.bucket_first_index) {
            return left.bucket_first_index < right.bucket_first_index;
        }
        return left.first_index > right.first_index;
    });

    Cloud retained;
    retained.origins = cloud.origins;
    retained.frame_timestamps = cloud.frame_timestamps;
    retained.frames_loaded = cloud.frames_loaded;
    retained.first_frame = cloud.first_frame;
    retained.last_frame = cloud.last_frame;
    retained.studio_auto_trim = cloud.studio_auto_trim;
    retained.points.reserve(selections.size());
    retained.frame_ids.reserve(selections.size());
    retained.relative_times.reserve(selections.size());
    std::vector<float> retained_scores;
    retained_scores.reserve(selections.size());
    for (const StudioVoxelSelection &selection : selections) {
        uint32_t index = selection.best_index;
        retained.points.push_back(cloud.points[index]);
        retained.frame_ids.push_back(cloud.frame_ids[index]);
        retained.relative_times.push_back(cloud.relative_times[index]);
        retained_scores.push_back(scores[index]);
    }
    cloud = std::move(retained);
    scores = std::move(retained_scores);
}

std::vector<float> point_times(const Cloud &cloud) {
    std::vector<float> times(cloud.points.size());
    double reference = cloud.frame_timestamps[cloud.frame_ids.front()];
    for (size_t index = 0; index < cloud.points.size(); ++index) {
        uint32_t frame = cloud.frame_ids[index];
        times[index] = static_cast<float>(
            (cloud.frame_timestamps[frame] - reference)
            + static_cast<double>(cloud.relative_times[index]) * 0.001
        );
    }
    return times;
}

struct DensityAssignment {
    std::vector<uint8_t> levels;
    std::vector<uint8_t> temporal_required;
    std::vector<uint8_t> active;
    std::array<uint64_t, 4> level_counts = {0, 0, 0, 0};
};

struct KdNode {
    uint32_t point_index = 0;
    int left = -1;
    int right = -1;
    uint8_t axis = 0;
};

float coordinate(const Float4 &point, int axis) {
    if (axis == 0) return point.x;
    if (axis == 1) return point.y;
    return point.z;
}

int build_kd_tree(
    std::vector<uint32_t> &indices,
    size_t begin,
    size_t end,
    int depth,
    const std::vector<Float4> &points,
    std::vector<KdNode> &nodes
) {
    if (begin >= end) {
        return -1;
    }
    size_t middle = begin + (end - begin) / 2;
    int axis = depth % 3;
    std::nth_element(
        indices.begin() + static_cast<ptrdiff_t>(begin),
        indices.begin() + static_cast<ptrdiff_t>(middle),
        indices.begin() + static_cast<ptrdiff_t>(end),
        [&](uint32_t left, uint32_t right) {
            float a = coordinate(points[left], axis);
            float b = coordinate(points[right], axis);
            return a == b ? left < right : a < b;
        }
    );
    int node_index = static_cast<int>(nodes.size());
    nodes.push_back({indices[middle], -1, -1, static_cast<uint8_t>(axis)});
    int left = build_kd_tree(indices, begin, middle, depth + 1, points, nodes);
    int right = build_kd_tree(indices, middle + 1, end, depth + 1, points, nodes);
    nodes[static_cast<size_t>(node_index)].left = left;
    nodes[static_cast<size_t>(node_index)].right = right;
    return node_index;
}

void nearest_other(
    int node_index,
    uint32_t query_index,
    const std::vector<KdNode> &nodes,
    const std::vector<Float4> &points,
    float &best_squared
) {
    if (node_index < 0) {
        return;
    }
    const KdNode &node = nodes[static_cast<size_t>(node_index)];
    const Float4 &query = points[query_index];
    const Float4 &candidate = points[node.point_index];
    float dx = query.x - candidate.x;
    float dy = query.y - candidate.y;
    float dz = query.z - candidate.z;
    float squared = dx * dx + dy * dy + dz * dz;
    if (node.point_index != query_index && squared < best_squared) {
        best_squared = squared;
    }
    float delta = coordinate(query, node.axis) - coordinate(candidate, node.axis);
    int near_node = delta < 0.0f ? node.left : node.right;
    int far_node = delta < 0.0f ? node.right : node.left;
    nearest_other(near_node, query_index, nodes, points, best_squared);
    if (delta * delta < best_squared) {
        nearest_other(far_node, query_index, nodes, points, best_squared);
    }
}

DensityAssignment assign_density_levels(
    const std::vector<Float4> &points,
    const std::vector<Float4> &trajectory
) {
    DensityAssignment assignment;
    assignment.levels.resize(points.size());
    assignment.temporal_required.resize(points.size());
    assignment.active.resize(points.size());
    std::vector<KeyPoint> ordered;
    ordered.reserve(points.size());
    for (uint32_t index = 0; index < points.size(); ++index) {
        uint64_t key;
        if (!point_key_cpu(points[index], kBlockSize, key)) {
            fail("point is outside the supported one-metre block range");
        }
        ordered.push_back({key, index});
    }
    std::sort(ordered.begin(), ordered.end(), [](const KeyPoint &left, const KeyPoint &right) {
        if (left.key != right.key) return left.key < right.key;
        return left.point_index < right.point_index;
    });

    std::vector<std::pair<size_t, size_t>> ranges;
    for (size_t begin = 0; begin < ordered.size();) {
        size_t end = begin + 1;
        while (end < ordered.size() && ordered[end].key == ordered[begin].key) {
            ++end;
        }
        ranges.emplace_back(begin, end);
        begin = end;
    }
    std::unordered_map<uint64_t, size_t> range_by_key;
    range_by_key.reserve(ranges.size() * 2);
    for (size_t index = 0; index < ranges.size(); ++index) {
        range_by_key.emplace(ordered[ranges[index].first].key, index);
    }

    std::atomic<size_t> next_range = 0;
    unsigned worker_count = std::max(1u, std::thread::hardware_concurrency());
    if (const char *limit = std::getenv("S20_CPU_THREADS")) {
        unsigned requested = std::stoul(limit);
        if (requested < 1 || requested > 128) fail("Invalid S20_CPU_THREADS");
        worker_count = std::min(worker_count, requested);
    }
    worker_count = std::min<unsigned>(worker_count, static_cast<unsigned>(ranges.size()));
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (unsigned worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&] {
            while (true) {
                size_t range_index = next_range.fetch_add(1);
                if (range_index >= ranges.size()) break;
                auto [begin, end] = ranges[range_index];
                const Float4 &sample = points[ordered[begin].point_index];
                int block_x = static_cast<int>(std::floor(sample.x / kBlockSize));
                int block_y = static_cast<int>(std::floor(sample.y / kBlockSize));
                int block_z = static_cast<int>(std::floor(sample.z / kBlockSize));
                float center_x = (static_cast<float>(block_x) + 0.5f) * kBlockSize;
                float center_y = (static_cast<float>(block_y) + 0.5f) * kBlockSize;
                float center_z = (static_cast<float>(block_z) + 0.5f) * kBlockSize;
                std::vector<uint32_t> local_indices;
                local_indices.reserve((end - begin) * 2);
                for (int dx = -1; dx <= 1; ++dx) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dz = -1; dz <= 1; ++dz) {
                            Float4 neighbor_center = {
                                (static_cast<float>(block_x + dx) + 0.5f) * kBlockSize,
                                (static_cast<float>(block_y + dy) + 0.5f) * kBlockSize,
                                (static_cast<float>(block_z + dz) + 0.5f) * kBlockSize,
                                0.0f,
                            };
                            uint64_t neighbor_key;
                            if (!point_key_cpu(neighbor_center, kBlockSize, neighbor_key)) continue;
                            auto found = range_by_key.find(neighbor_key);
                            if (found == range_by_key.end()) continue;
                            auto [neighbor_begin, neighbor_end] = ranges[found->second];
                            for (size_t cursor = neighbor_begin; cursor < neighbor_end; ++cursor) {
                                uint32_t point_index = ordered[cursor].point_index;
                                const Float4 &point = points[point_index];
                                if ((dx == 0 && dy == 0 && dz == 0)
                                    || (std::abs(point.x - center_x) < 0.55f * kBlockSize
                                        && std::abs(point.y - center_y) < 0.55f * kBlockSize
                                        && std::abs(point.z - center_z) < 0.55f * kBlockSize)) {
                                    local_indices.push_back(point_index);
                                }
                            }
                        }
                    }
                }

                float mean_distance = 0.0f;
                if (end - begin > 7 && local_indices.size() > 1) {
                    std::vector<KdNode> nodes;
                    nodes.reserve(local_indices.size());
                    int root = build_kd_tree(
                        local_indices, 0, local_indices.size(), 0, points, nodes
                    );
                    double sum = 0.0;
                    // Studio appends the five-centimetre halo to the core cloud
                    // before calculateDenseLevel(), then averages the 2-NN
                    // distance over that entire combined cloud.  The core count
                    // is used only for the >= 8-point eligibility check above.
                    for (uint32_t index : local_indices) {
                        float best = std::numeric_limits<float>::infinity();
                        nearest_other(root, index, nodes, points, best);
                        if (std::isfinite(best)) sum += std::sqrt(best);
                    }
                    mean_distance = static_cast<float>(
                        sum / static_cast<double>(local_indices.size())
                    );
                }
                uint8_t level = 0;
                if (mean_distance >= kDensityThresholds[2]) level = 3;
                else if (mean_distance >= kDensityThresholds[1]) level = 2;
                else if (mean_distance >= kDensityThresholds[0]) level = 1;

                float nearest_squared = std::numeric_limits<float>::infinity();
                for (const Float4 &pose : trajectory) {
                    float dx = center_x - pose.x;
                    float dy = center_y - pose.y;
                    float dz = center_z - pose.z;
                    nearest_squared = std::min(nearest_squared, dx * dx + dy * dy + dz * dz);
                }
                float trajectory_distance = std::sqrt(nearest_squared);
                uint8_t required = 0;
                const bool active = end - begin > 7;
                if (active && trajectory_distance <= 2.0f) {
                    int interpolated = 1;
                    if (trajectory_distance >= 1.0f) {
                        interpolated = static_cast<int>(2.0f - trajectory_distance);
                    }
                    required = static_cast<uint8_t>(std::max(0, interpolated));
                }
                for (size_t cursor = begin; cursor < end; ++cursor) {
                    uint32_t index = ordered[cursor].point_index;
                    assignment.levels[index] = level;
                    assignment.temporal_required[index] = required;
                    assignment.active[index] = active ? 1 : 0;
                }
            }
        });
    }
    for (std::thread &worker : workers) worker.join();
    for (size_t index = 0; index < assignment.levels.size(); ++index) {
        if (assignment.active[index] != 0) {
            ++assignment.level_counts[assignment.levels[index]];
        }
    }
    return assignment;
}

struct ExpandedBlockContexts {
    Cloud cloud;
    std::vector<float> scores;
    std::vector<uint32_t> context_ids;
    std::vector<Float4> context_centers;
    std::vector<uint8_t> core_flags;
    std::vector<uint8_t> levels;
    std::vector<uint8_t> temporal_required;
    std::array<uint64_t, 4> level_counts = {0, 0, 0, 0};
};

ExpandedBlockContexts expand_block_contexts(
    const Cloud &source,
    const std::vector<float> &source_scores,
    const DensityAssignment &density
) {
    if (source.points.size() != source_scores.size()
        || source.points.size() != density.levels.size()
        || source.points.size() != density.temporal_required.size()
        || source.points.size() != density.active.size()) {
        fail("internal block-context size mismatch");
    }

    std::vector<KeyPoint> ordered;
    ordered.reserve(source.points.size());
    for (uint32_t index = 0; index < source.points.size(); ++index) {
        uint64_t key;
        if (!point_key_cpu(source.points[index], kBlockSize, key)) {
            fail("point is outside the supported one-metre block range");
        }
        ordered.push_back({key, index});
    }
    std::sort(ordered.begin(), ordered.end(), [](const KeyPoint &left, const KeyPoint &right) {
        if (left.key != right.key) return left.key < right.key;
        return left.point_index < right.point_index;
    });

    std::vector<std::pair<size_t, size_t>> ranges;
    for (size_t begin = 0; begin < ordered.size();) {
        size_t end = begin + 1;
        while (end < ordered.size() && ordered[end].key == ordered[begin].key) ++end;
        ranges.emplace_back(begin, end);
        begin = end;
    }
    std::unordered_map<uint64_t, size_t> range_by_key;
    range_by_key.reserve(ranges.size() * 2);
    for (size_t index = 0; index < ranges.size(); ++index) {
        range_by_key.emplace(ordered[ranges[index].first].key, index);
    }

    ExpandedBlockContexts expanded;
    expanded.cloud.origins = source.origins;
    expanded.cloud.frame_timestamps = source.frame_timestamps;
    expanded.cloud.frames_loaded = source.frames_loaded;
    expanded.cloud.first_frame = source.first_frame;
    expanded.cloud.last_frame = source.last_frame;
    expanded.cloud.studio_auto_trim = source.studio_auto_trim;
    size_t reserve_count = source.points.size() + source.points.size() / 5;
    expanded.cloud.points.reserve(reserve_count);
    expanded.cloud.frame_ids.reserve(reserve_count);
    expanded.cloud.relative_times.reserve(reserve_count);
    expanded.scores.reserve(reserve_count);
    expanded.context_ids.reserve(reserve_count);
    expanded.core_flags.reserve(reserve_count);
    expanded.levels.reserve(reserve_count);
    expanded.temporal_required.reserve(reserve_count);

    uint32_t context_id = 0;
    for (const auto &[begin, end] : ranges) {
        uint32_t first_index = ordered[begin].point_index;
        if (density.active[first_index] == 0) continue;
        if (context_id == std::numeric_limits<uint32_t>::max()) {
            fail("too many one-metre block contexts");
        }
        uint8_t level = density.levels[first_index];
        uint8_t required = density.temporal_required[first_index];
        const Float4 &sample = source.points[first_index];
        int block_x = static_cast<int>(std::floor(sample.x / kBlockSize));
        int block_y = static_cast<int>(std::floor(sample.y / kBlockSize));
        int block_z = static_cast<int>(std::floor(sample.z / kBlockSize));
        float center_x = (static_cast<float>(block_x) + 0.5f) * kBlockSize;
        float center_y = (static_cast<float>(block_y) + 0.5f) * kBlockSize;
        float center_z = (static_cast<float>(block_z) + 0.5f) * kBlockSize;
        expanded.context_centers.push_back({center_x, center_y, center_z, 0.0f});

        auto append = [&](uint32_t index, bool core) {
            expanded.cloud.points.push_back(source.points[index]);
            expanded.cloud.frame_ids.push_back(source.frame_ids[index]);
            expanded.cloud.relative_times.push_back(source.relative_times[index]);
            expanded.scores.push_back(source_scores[index]);
            expanded.context_ids.push_back(context_id);
            expanded.core_flags.push_back(core ? 1 : 0);
            expanded.levels.push_back(level);
            expanded.temporal_required.push_back(required);
            ++expanded.level_counts[level];
        };

        // divideCloudToBlock walks the downsampled cloud once and immediately
        // appends each point to its core block and every eligible halo block.
        // Reconstruct that per-context source order: putting the whole core
        // before its halo would change allocatePointKernel's first-writer
        // representative whenever both occupy one smoothing voxel.
        std::vector<std::pair<uint32_t, bool>> context_points;
        context_points.reserve((end - begin) * 2);
        for (size_t cursor = begin; cursor < end; ++cursor) {
            context_points.emplace_back(ordered[cursor].point_index, true);
        }
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    Float4 neighbor_center = {
                        (static_cast<float>(block_x + dx) + 0.5f) * kBlockSize,
                        (static_cast<float>(block_y + dy) + 0.5f) * kBlockSize,
                        (static_cast<float>(block_z + dz) + 0.5f) * kBlockSize,
                        0.0f,
                    };
                    uint64_t neighbor_key;
                    if (!point_key_cpu(neighbor_center, kBlockSize, neighbor_key)) continue;
                    auto found = range_by_key.find(neighbor_key);
                    if (found == range_by_key.end()) continue;
                    auto [neighbor_begin, neighbor_end] = ranges[found->second];
                    for (size_t cursor = neighbor_begin; cursor < neighbor_end; ++cursor) {
                        uint32_t index = ordered[cursor].point_index;
                        const Float4 &point = source.points[index];
                        if (std::abs(point.x - center_x) < 0.55f * kBlockSize
                            && std::abs(point.y - center_y) < 0.55f * kBlockSize
                            && std::abs(point.z - center_z) < 0.55f * kBlockSize) {
                            context_points.emplace_back(index, false);
                        }
                    }
                }
            }
        }
        std::sort(context_points.begin(), context_points.end(), [](const auto &left,
                                                                   const auto &right) {
            return left.first < right.first;
        });
        for (const auto &[index, core] : context_points) append(index, core);
        ++context_id;
    }
    return expanded;
}

void compact_block_contexts(
    ExpandedBlockContexts &expanded,
    const std::vector<uint8_t> &keep
) {
    if (keep.size() != expanded.cloud.points.size()
        || keep.size() != expanded.context_ids.size()
        || keep.size() != expanded.core_flags.size()
        || keep.size() != expanded.levels.size()
        || keep.size() != expanded.temporal_required.size()) {
        fail("internal expanded-context compact size mismatch");
    }
    compact_cloud(expanded.cloud, keep, expanded.scores);
    auto compact_bytes = [&](std::vector<uint8_t> &values) {
        std::vector<uint8_t> retained;
        retained.reserve(expanded.cloud.points.size());
        for (size_t index = 0; index < keep.size(); ++index) {
            if (keep[index] != 0) retained.push_back(values[index]);
        }
        values = std::move(retained);
    };
    std::vector<uint32_t> contexts;
    contexts.reserve(expanded.cloud.points.size());
    for (size_t index = 0; index < keep.size(); ++index) {
        if (keep[index] != 0) contexts.push_back(expanded.context_ids[index]);
    }
    expanded.context_ids = std::move(contexts);
    compact_bytes(expanded.core_flags);
    compact_bytes(expanded.levels);
    compact_bytes(expanded.temporal_required);
    expanded.level_counts = {0, 0, 0, 0};
    for (uint8_t level : expanded.levels) ++expanded.level_counts[level];
}

void orient_normals_to_trajectory(const Cloud &cloud, std::vector<Float4> &normals) {
    if (normals.size() != cloud.points.size()) {
        fail("internal normal-orientation size mismatch");
    }
    for (size_t index = 0; index < normals.size(); ++index) {
        Float4 &normal = normals[index];
        float squared = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
        if (!(squared > 0.0f)) continue;
        float inverse = 1.0f / std::sqrt(squared);
        normal.x *= inverse;
        normal.y *= inverse;
        normal.z *= inverse;
        const Float4 &point = cloud.points[index];
        const Float4 &origin = cloud.origins[cloud.frame_ids[index]];
        float facing = normal.x * (origin.x - point.x)
            + normal.y * (origin.y - point.y)
            + normal.z * (origin.z - point.z);
        if (facing < 0.0f) {
            normal.x = -normal.x;
            normal.y = -normal.y;
            normal.z = -normal.z;
        }
    }
}

void build_keep_mask(
    const Cloud &cloud,
    FilterResult &result,
    bool deduplicate
) {
    auto started = std::chrono::steady_clock::now();
    result.keep_mask.assign(cloud.points.size(), 0);
    if (!deduplicate) {
        for (size_t index = 0; index < cloud.points.size(); ++index) {
            result.keep_mask[index] = result.noise_mask[index] == 0 ? 1 : 0;
        }
    } else {
        std::vector<KeyPoint> survivors;
        survivors.reserve(cloud.points.size() - static_cast<size_t>(result.noise_points));
        for (uint32_t index = 0; index < cloud.points.size(); ++index) {
            if (result.noise_mask[index] != 0) {
                continue;
            }
            uint64_t key;
            if (point_key_cpu(cloud.points[index], kOutputVoxelSize, key)) {
                survivors.push_back({key, index});
            }
        }
        std::sort(survivors.begin(), survivors.end(), [](const KeyPoint &left, const KeyPoint &right) {
            if (left.key != right.key) {
                return left.key < right.key;
            }
            return left.point_index < right.point_index;
        });
        for (size_t begin = 0; begin < survivors.size();) {
            size_t end = begin + 1;
            while (end < survivors.size() && survivors[end].key == survivors[begin].key) {
                ++end;
            }
            uint32_t best_index = survivors[begin].point_index;
            float cell_x = std::floor(cloud.points[best_index].x / kOutputVoxelSize);
            float cell_y = std::floor(cloud.points[best_index].y / kOutputVoxelSize);
            float cell_z = std::floor(cloud.points[best_index].z / kOutputVoxelSize);
            float center_x = (cell_x + 0.5f) * kOutputVoxelSize;
            float center_y = (cell_y + 0.5f) * kOutputVoxelSize;
            float center_z = (cell_z + 0.5f) * kOutputVoxelSize;
            auto center_distance_squared = [&](uint32_t index) {
                float dx = cloud.points[index].x - center_x;
                float dy = cloud.points[index].y - center_y;
                float dz = cloud.points[index].z - center_z;
                return dx * dx + dy * dy + dz * dz;
            };
            float best_distance = center_distance_squared(best_index);
            for (size_t cursor = begin + 1; cursor < end; ++cursor) {
                uint32_t candidate = survivors[cursor].point_index;
                float candidate_distance = center_distance_squared(candidate);
                if (candidate_distance < best_distance) {
                    best_distance = candidate_distance;
                    best_index = candidate;
                }
            }
            result.keep_mask[best_index] = 1;
            begin = end;
        }
    }
    result.kept_points = static_cast<uint64_t>(std::count(
        result.keep_mask.begin(), result.keep_mask.end(), uint8_t(1)
    ));
    result.dedup_seconds = elapsed_seconds(started);
}

double elapsed_seconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void copy_cloud_metadata(const Cloud &source, Cloud &destination) {
    destination.origins = source.origins;
    destination.frame_timestamps = source.frame_timestamps;
    destination.frames_loaded = source.frames_loaded;
    destination.first_frame = source.first_frame;
    destination.last_frame = source.last_frame;
    destination.studio_auto_trim = source.studio_auto_trim;
}

struct SpatialSubfile {
    std::array<int, 3> coordinate;
    Cloud cloud;
};

// Subfile coordinates a point contributes to: its core cell plus any
// neighbour it lies within the halo of.
void subfile_cells(const Float4 &point, std::array<std::vector<int>, 3> &axes) {
    const float values[3] = {point.x, point.y, point.z};
    for (size_t axis = 0; axis < 3; ++axis) {
        axes[axis].clear();
        int core = static_cast<int>(std::floor(values[axis] / kSubfileSize));
        float lower = static_cast<float>(core) * kSubfileSize;
        axes[axis].push_back(core);
        if (values[axis] - lower < kSubfileHalo) {
            axes[axis].push_back(core - 1);
        }
        if (lower + kSubfileSize - values[axis] < kSubfileHalo) {
            axes[axis].push_back(core + 1);
        }
    }
}

// Every subfile coordinate with its point count, in the same order the
// former all-at-once division produced.
constexpr size_t kScanChunks = 32;

std::map<std::array<int, 3>, size_t> list_ray_subfiles(const Cloud &source) {
    std::vector<std::map<std::array<int, 3>, size_t>> partial(kScanChunks);
    auto *partial_data = partial.data();
    const size_t total = source.points.size();
    const size_t step = (total + kScanChunks - 1) / kScanChunks;
    dispatch_apply(kScanChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t chunk) {
        std::array<std::vector<int>, 3> axes;
        auto &counts = partial_data[chunk];
        for (size_t index = chunk * step; index < std::min(total, (chunk + 1) * step); ++index) {
            subfile_cells(source.points[index], axes);
            for (int x : axes[0]) for (int y : axes[1]) for (int z : axes[2]) ++counts[{x, y, z}];
        }
    });
    std::map<std::array<int, 3>, size_t> counts;
    for (const auto &part : partial) for (const auto &[key, count] : part) counts[key] += count;
    return counts;
}

// One subfile (core plus halo), extracted on demand so only one exists at a time.
Cloud extract_ray_subfile(const Cloud &source, const std::array<int, 3> &coordinate, size_t expected) {
    const size_t total = source.points.size();
    const size_t step = (total + kScanChunks - 1) / kScanChunks;
    std::vector<std::vector<uint32_t>> members(kScanChunks);
    auto *members_data = members.data();
    dispatch_apply(kScanChunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t chunk) {
        std::array<std::vector<int>, 3> axes;
        auto &found = members_data[chunk];
        for (size_t index = chunk * step; index < std::min(total, (chunk + 1) * step); ++index) {
            subfile_cells(source.points[index], axes);
            bool member = false;
            for (int x : axes[0]) for (int y : axes[1]) for (int z : axes[2]) {
                if (x == coordinate[0] && y == coordinate[1] && z == coordinate[2]) member = true;
            }
            if (member) found.push_back(static_cast<uint32_t>(index));
        }
    });
    Cloud part;
    part.points.reserve(expected);
    part.frame_ids.reserve(expected);
    part.relative_times.reserve(expected);
    for (const auto &found : members) {
        for (uint32_t index : found) {
            part.points.push_back(source.points[index]);
            part.frame_ids.push_back(source.frame_ids[index]);
            part.relative_times.push_back(source.relative_times[index]);
        }
    }
    copy_cloud_metadata(source, part);
    return part;
}

bool belongs_to_subfile_core(
    const Float4 &point,
    const std::array<int, 3> &coordinate
) {
    return static_cast<int>(std::floor(point.x / kSubfileSize)) == coordinate[0]
        && static_cast<int>(std::floor(point.y / kSubfileSize)) == coordinate[1]
        && static_cast<int>(std::floor(point.z / kSubfileSize)) == coordinate[2];
}

class MetalRayFilter {
public:
    explicit MetalRayFilter(const fs::path &kernel_path) {
        device_ = MTLCreateSystemDefaultDevice();
        if (device_ == nil) {
            fail("Metal is unavailable on this Mac");
        }
        queue_ = [device_ newCommandQueue];
        if (queue_ == nil) {
            fail("failed to create Metal command queue");
        }

        NSError *read_error = nil;
        NSString *path = [NSString stringWithUTF8String:kernel_path.c_str()];
        NSString *source = [NSString stringWithContentsOfFile:path
                                                     encoding:NSUTF8StringEncoding
                                                        error:&read_error];
        if (source == nil) {
            fail("cannot read Metal kernels: " + std::string(read_error.localizedDescription.UTF8String));
        }
        MTLCompileOptions *compile_options = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
            // The recovered CUDA kernels consistently use .approx.ftz for
            // division, square root, reciprocal, and trigonometry.  Metal's
            // fast mode is the closest native Apple-GPU execution model.
            compile_options.mathMode = MTLMathModeFast;
            compile_options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsFast;
        } else {
            [compile_options setValue:@YES forKey:@"fastMathEnabled"];
        }
        if (@available(macOS 26.0, *)) {
            compile_options.languageVersion = MTLLanguageVersion4_0;
        }
        NSError *compile_error = nil;
        library_ = [device_ newLibraryWithSource:source
                                        options:compile_options
                                          error:&compile_error];
        if (library_ == nil) {
            fail("Metal kernel compilation failed: "
                 + std::string(compile_error.localizedDescription.UTF8String));
        }
        ray_pipeline_ = make_pipeline(@"shoot_and_update");
        classify_pipeline_ = make_pipeline(@"classify_points");
        temporal_pipeline_ = make_pipeline(@"temporal_consistency");
        normal_pipeline_ = make_pipeline(@"calculate_normals");
        smooth_normal_pipeline_ = make_pipeline(@"smooth_normals");
        mls_pipeline_ = make_pipeline(@"normal_mls");
    }

    FilterResult run(
        const Cloud &cloud,
        bool deduplicate,
        const Cloud *ray_source = nullptr,
        const std::array<int, 3> *ray_subfile_coordinate = nullptr
    ) {
        if (cloud.points.size() > std::numeric_limits<uint32_t>::max()) {
            fail("point count exceeds the Metal kernel's uint32 index range");
        }
        const Cloud &rays = ray_source == nullptr ? cloud : *ray_source;
        if (rays.points.size() > std::numeric_limits<uint32_t>::max()) {
            fail("ray count exceeds the Metal kernel's uint32 index range");
        }
        uint32_t point_count = static_cast<uint32_t>(cloud.points.size());
        auto insert_started = std::chrono::steady_clock::now();
        Float4 core_minimum = {0.0f, 0.0f, 0.0f, 0.0f};
        Float4 core_maximum = {0.0f, 0.0f, 0.0f, 0.0f};
        uint32_t clip_to_subfile = ray_subfile_coordinate == nullptr ? 0u : 1u;
        if (ray_subfile_coordinate != nullptr) {
            // Studio's threadReader constructs SubMapPacket::bbox from the
            // actual finite points loaded from raw_subfile_*.las.  prepareRays
            // and voxelizeCloud then reuse that measured box, rather than the
            // nominal fifty-metre core, for clipping and the local grid origin.
            core_minimum = {
                std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                0.0f,
            };
            core_maximum = {
                -std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                0.0f,
            };
            for (const Float4 &point : cloud.points) {
                if (!std::isfinite(point.x) || !std::isfinite(point.y)
                    || !std::isfinite(point.z)) {
                    continue;
                }
                core_minimum.x = std::min(core_minimum.x, point.x);
                core_minimum.y = std::min(core_minimum.y, point.y);
                core_minimum.z = std::min(core_minimum.z, point.z);
                core_maximum.x = std::max(core_maximum.x, point.x);
                core_maximum.y = std::max(core_maximum.y, point.y);
                core_maximum.z = std::max(core_maximum.z, point.z);
            }
            if (!std::isfinite(core_minimum.x) || !std::isfinite(core_maximum.x)) {
                fail("ray subfile contains no finite points");
            }
        }
        CpuHashTable hash = build_ray_hash(cloud.points, core_minimum);
        uint64_t capacity64 = hash.keys.size();
        uint32_t capacity = static_cast<uint32_t>(capacity64);
        uint32_t hash_mask = capacity - 1;

        id<MTLBuffer> points = make_buffer(cloud.points.data(), cloud.points.size() * sizeof(Float4), @"target points");
        id<MTLBuffer> keys = make_buffer(hash.keys.data(), hash.keys.size() * sizeof(uint64_t), @"ray voxel keys");
        id<MTLBuffer> representatives = make_buffer(
            hash.representatives.data(), hash.representatives.size() * sizeof(Float4),
            @"ray voxel representatives"
        );
        id<MTLBuffer> scores = make_zero_buffer(capacity64 * sizeof(uint32_t), @"ray scores");
        id<MTLBuffer> noise_mask = make_zero_buffer(point_count * sizeof(uint8_t), @"noise mask");
        id<MTLBuffer> point_scores = make_zero_buffer(point_count * sizeof(float), @"point scores");

        FilterResult result;
        result.input_points = cloud.points.size();
        result.hash_capacity = capacity64;
        result.device_name = device_.name.UTF8String;
        result.occupied_voxels = hash.occupied;
        result.insert_seconds = elapsed_seconds(insert_started);
        hash.keys.clear();
        hash.representatives.clear();
        hash.keys.shrink_to_fit();
        hash.representatives.shrink_to_fit();

        auto started = std::chrono::steady_clock::now();
        // Rays are streamed in batches; the score buffer accumulates across
        // dispatches, so memory no longer scales with the whole cloud.
        std::vector<RayRecord> prepared_rays;
        prepared_rays.reserve(std::min(rays.points.size(), kRayBatchPoints));
        for (size_t begin = 0; begin < rays.points.size(); begin += kRayBatchPoints) {
            @autoreleasepool {
                size_t end = std::min(rays.points.size(), begin + kRayBatchPoints);
                prepare_ray_records(
                    rays, begin, end, core_minimum, core_maximum, clip_to_subfile != 0u, prepared_rays
                );
                if (prepared_rays.empty()) continue;
                uint32_t ray_count = static_cast<uint32_t>(prepared_rays.size());
                id<MTLBuffer> ray_records = make_buffer(
                    prepared_rays.data(), prepared_rays.size() * sizeof(RayRecord),
                    @"prepared Studio ray records"
                );
                dispatch(ray_pipeline_, ray_count, [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:ray_records offset:0 atIndex:0];
                    [encoder setBuffer:keys offset:0 atIndex:1];
                    [encoder setBuffer:representatives offset:0 atIndex:2];
                    [encoder setBuffer:scores offset:0 atIndex:3];
                    [encoder setBytes:&ray_count length:sizeof(ray_count) atIndex:4];
                    [encoder setBytes:&hash_mask length:sizeof(hash_mask) atIndex:5];
                    [encoder setBytes:&core_minimum length:sizeof(core_minimum) atIndex:6];
                });
            }
        }
        prepared_rays.clear();
        prepared_rays.shrink_to_fit();
        result.ray_seconds = elapsed_seconds(started);

        started = std::chrono::steady_clock::now();
        dispatch(classify_pipeline_, point_count, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:points offset:0 atIndex:0];
            [encoder setBuffer:keys offset:0 atIndex:1];
            [encoder setBuffer:scores offset:0 atIndex:2];
            [encoder setBuffer:noise_mask offset:0 atIndex:3];
            [encoder setBuffer:point_scores offset:0 atIndex:4];
            [encoder setBytes:&point_count length:sizeof(point_count) atIndex:5];
            [encoder setBytes:&hash_mask length:sizeof(hash_mask) atIndex:6];
            [encoder setBytes:&core_minimum length:sizeof(core_minimum) atIndex:7];
        });
        result.classify_seconds = elapsed_seconds(started);

        auto *noise_values = static_cast<uint8_t *>(noise_mask.contents);
        auto *score_values = static_cast<float *>(point_scores.contents);
        result.noise_mask.assign(noise_values, noise_values + point_count);
        result.point_scores.assign(score_values, score_values + point_count);
        for (uint32_t index = 0; index < point_count; ++index) {
            result.noise_points += result.noise_mask[index] != 0;
        }
        build_keep_mask(cloud, result, deduplicate);
        return result;
    }

    void smooth(Cloud &cloud, FilterResult &result, bool deduplicate) {
        if (deduplicate) {
            compact_cloud_in_studio_downsample_order(
                cloud, result.noise_mask, result.point_scores, kOutputVoxelSize
            );
        } else {
            compact_cloud(cloud, result.keep_mask, result.point_scores);
        }
        result.pre_smooth_points = cloud.points.size();
        result.keep_mask.assign(cloud.points.size(), 1);
        result.noise_mask.assign(cloud.points.size(), 0);
        result.kept_points = cloud.points.size();
        if (cloud.points.empty()) {
            result.normals.clear();
            return;
        }
        if (cloud.points.size() > std::numeric_limits<uint32_t>::max()) {
            fail("point count exceeds smoothing kernel index range");
        }

        auto density_started = std::chrono::steady_clock::now();
        DensityAssignment density = assign_density_levels(cloud.points, cloud.origins);
        result.sparse_block_removed_points = static_cast<uint64_t>(std::count(
            density.active.begin(), density.active.end(), uint8_t(0)
        ));
        ExpandedBlockContexts expanded = expand_block_contexts(
            cloud, result.point_scores, density
        );
        result.density_seconds += elapsed_seconds(density_started);
        result.density_level_counts = density.level_counts;
        result.expanded_context_points = expanded.cloud.points.size();
        if (expanded.cloud.points.empty()) {
            cloud.points.clear();
            cloud.frame_ids.clear();
            cloud.relative_times.clear();
            result.keep_mask.clear();
            result.noise_mask.clear();
            result.point_scores.clear();
            result.normals.clear();
            result.kept_points = 0;
            return;
        }
        if (expanded.cloud.points.size() > std::numeric_limits<uint32_t>::max()) {
            fail("expanded block contexts exceed the smoothing kernel index range");
        }

        uint32_t point_count = static_cast<uint32_t>(expanded.cloud.points.size());
        std::vector<float> times = point_times(expanded.cloud);

        id<MTLBuffer> points = make_buffer(
            expanded.cloud.points.data(), expanded.cloud.points.size() * sizeof(Float4),
            @"block-context points"
        );
        id<MTLBuffer> time_buffer = make_buffer(
            times.data(), times.size() * sizeof(float), @"point times"
        );
        id<MTLBuffer> levels = make_buffer(
            expanded.levels.data(), expanded.levels.size(), @"density levels"
        );
        id<MTLBuffer> required = make_buffer(
            expanded.temporal_required.data(), expanded.temporal_required.size(),
            @"time-diversity requirements"
        );
        id<MTLBuffer> contexts = make_buffer(
            expanded.context_ids.data(),
            expanded.context_ids.size() * sizeof(uint32_t),
            @"block context IDs"
        );
        id<MTLBuffer> context_centers = make_buffer(
            expanded.context_centers.data(),
            expanded.context_centers.size() * sizeof(Float4),
            @"block context centers"
        );
        id<MTLBuffer> temporal_mask = make_zero_buffer(
            expanded.cloud.points.size(), @"temporal removal mask"
        );

        auto temporal_started = std::chrono::steady_clock::now();
        for (uint32_t level = 0; level < 4; ++level) {
            if (expanded.level_counts[level] == 0) continue;
            float voxel_size = kSmoothVoxelSizes[level];
            CpuIndexHash hash = build_index_hash(
                expanded.cloud.points,
                voxel_size,
                &expanded.levels,
                static_cast<uint8_t>(level),
                &expanded.context_ids,
                &expanded.context_centers
            );
            uint32_t hash_mask = static_cast<uint32_t>(hash.keys.size() - 1);
            id<MTLBuffer> keys = make_buffer(
                hash.keys.data(), hash.keys.size() * sizeof(uint64_t), @"temporal keys"
            );
            id<MTLBuffer> indices = make_buffer(
                hash.point_indices.data(), hash.point_indices.size() * sizeof(uint32_t),
                @"temporal representatives"
            );
            dispatch(temporal_pipeline_, point_count, [&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:points offset:0 atIndex:0];
                [encoder setBuffer:time_buffer offset:0 atIndex:1];
                [encoder setBuffer:levels offset:0 atIndex:2];
                [encoder setBuffer:required offset:0 atIndex:3];
                [encoder setBuffer:keys offset:0 atIndex:4];
                [encoder setBuffer:indices offset:0 atIndex:5];
                [encoder setBuffer:contexts offset:0 atIndex:6];
                [encoder setBuffer:temporal_mask offset:0 atIndex:7];
                [encoder setBuffer:context_centers offset:0 atIndex:8];
                [encoder setBytes:&point_count length:sizeof(point_count) atIndex:9];
                [encoder setBytes:&hash_mask length:sizeof(hash_mask) atIndex:10];
                [encoder setBytes:&voxel_size length:sizeof(voxel_size) atIndex:11];
                [encoder setBytes:&level length:sizeof(level) atIndex:12];
            });
        }
        result.temporal_seconds = elapsed_seconds(temporal_started);

        auto *removed = static_cast<uint8_t *>(temporal_mask.contents);
        std::vector<uint8_t> keep(point_count, 1);
        for (uint32_t index = 0; index < point_count; ++index) {
            if (removed[index] != 0) {
                keep[index] = 0;
                if (expanded.core_flags[index] != 0) {
                    ++result.temporal_removed_points;
                }
            }
        }
        compact_block_contexts(expanded, keep);
        if (expanded.cloud.points.empty()) {
            cloud.points.clear();
            cloud.frame_ids.clear();
            cloud.relative_times.clear();
            result.keep_mask.clear();
            result.noise_mask.clear();
            result.point_scores.clear();
            result.normals.clear();
            result.kept_points = 0;
            return;
        }

        point_count = static_cast<uint32_t>(expanded.cloud.points.size());

        auto smoothing_started = std::chrono::steady_clock::now();
        std::vector<Float4> last_normals(expanded.cloud.points.size());
        for (int iteration = 0; iteration < 2; ++iteration) {
            uint32_t invalidate_on_failure = iteration == 0 ? 1u : 0u;
            points = make_buffer(
                expanded.cloud.points.data(),
                expanded.cloud.points.size() * sizeof(Float4),
                @"block-context points"
            );
            levels = make_buffer(
                expanded.levels.data(), expanded.levels.size(), @"density levels"
            );
            contexts = make_buffer(
                expanded.context_ids.data(),
                expanded.context_ids.size() * sizeof(uint32_t),
                @"block context IDs"
            );
            id<MTLBuffer> normals = make_zero_buffer(
                expanded.cloud.points.size() * sizeof(Float4), @"calculated normals"
            );

            for (uint32_t level = 0; level < 4; ++level) {
                if (expanded.level_counts[level] == 0) continue;
                float voxel_size = kSmoothVoxelSizes[level];
                CpuIndexHash hash = build_index_hash(
                    expanded.cloud.points,
                    voxel_size,
                    &expanded.levels,
                    static_cast<uint8_t>(level),
                    &expanded.context_ids,
                    &expanded.context_centers
                );
                uint32_t hash_mask = static_cast<uint32_t>(hash.keys.size() - 1);
                id<MTLBuffer> keys = make_buffer(
                    hash.keys.data(), hash.keys.size() * sizeof(uint64_t), @"normal keys"
                );
                id<MTLBuffer> indices = make_buffer(
                    hash.point_indices.data(), hash.point_indices.size() * sizeof(uint32_t),
                    @"normal representatives"
                );
                dispatch(normal_pipeline_, point_count, [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:points offset:0 atIndex:0];
                    [encoder setBuffer:levels offset:0 atIndex:1];
                    [encoder setBuffer:keys offset:0 atIndex:2];
                    [encoder setBuffer:indices offset:0 atIndex:3];
                    [encoder setBuffer:contexts offset:0 atIndex:4];
                    [encoder setBuffer:context_centers offset:0 atIndex:5];
                    [encoder setBuffer:normals offset:0 atIndex:6];
                    [encoder setBytes:&point_count length:sizeof(point_count) atIndex:7];
                    [encoder setBytes:&hash_mask length:sizeof(hash_mask) atIndex:8];
                    [encoder setBytes:&voxel_size length:sizeof(voxel_size) atIndex:9];
                    [encoder setBytes:&level length:sizeof(level) atIndex:10];
                });
            }

            id<MTLBuffer> smoothed_normals = make_zero_buffer(
                expanded.cloud.points.size() * sizeof(Float4), @"smoothed normals"
            );
            for (uint32_t level = 0; level < 4; ++level) {
                if (expanded.level_counts[level] == 0) continue;
                float voxel_size = kSmoothVoxelSizes[level];
                CpuIndexHash hash = build_index_hash(
                    expanded.cloud.points,
                    voxel_size,
                    &expanded.levels,
                    static_cast<uint8_t>(level),
                    &expanded.context_ids,
                    &expanded.context_centers
                );
                uint32_t hash_mask = static_cast<uint32_t>(hash.keys.size() - 1);
                id<MTLBuffer> keys = make_buffer(
                    hash.keys.data(), hash.keys.size() * sizeof(uint64_t), @"normal-smooth keys"
                );
                id<MTLBuffer> indices = make_buffer(
                    hash.point_indices.data(), hash.point_indices.size() * sizeof(uint32_t),
                    @"normal-smooth representatives"
                );
                dispatch(smooth_normal_pipeline_, point_count, [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:points offset:0 atIndex:0];
                    [encoder setBuffer:normals offset:0 atIndex:1];
                    [encoder setBuffer:levels offset:0 atIndex:2];
                    [encoder setBuffer:keys offset:0 atIndex:3];
                    [encoder setBuffer:indices offset:0 atIndex:4];
                    [encoder setBuffer:contexts offset:0 atIndex:5];
                    [encoder setBuffer:context_centers offset:0 atIndex:6];
                    [encoder setBuffer:smoothed_normals offset:0 atIndex:7];
                    [encoder setBytes:&point_count length:sizeof(point_count) atIndex:8];
                    [encoder setBytes:&hash_mask length:sizeof(hash_mask) atIndex:9];
                    [encoder setBytes:&voxel_size length:sizeof(voxel_size) atIndex:10];
                    [encoder setBytes:&level length:sizeof(level) atIndex:11];
                });
            }

            id<MTLBuffer> moved_points = make_buffer(
                expanded.cloud.points.data(),
                expanded.cloud.points.size() * sizeof(Float4),
                @"MLS output points"
            );
            id<MTLBuffer> mls_normals = make_buffer(
                smoothed_normals.contents,
                expanded.cloud.points.size() * sizeof(Float4),
                @"MLS output normals"
            );
            for (uint32_t level = 0; level < 4; ++level) {
                if (std::getenv("S20_BENCH_NO_MLS")) break;
                if (expanded.level_counts[level] == 0) continue;
                float voxel_size = kSmoothVoxelSizes[level];
                CpuIndexHash hash = build_index_hash(
                    expanded.cloud.points,
                    voxel_size,
                    &expanded.levels,
                    static_cast<uint8_t>(level),
                    &expanded.context_ids,
                    &expanded.context_centers
                );
                uint32_t hash_mask = static_cast<uint32_t>(hash.keys.size() - 1);
                id<MTLBuffer> keys = make_buffer(
                    hash.keys.data(), hash.keys.size() * sizeof(uint64_t), @"MLS keys"
                );
                id<MTLBuffer> indices = make_buffer(
                    hash.point_indices.data(), hash.point_indices.size() * sizeof(uint32_t),
                    @"MLS representatives"
                );
                dispatch(mls_pipeline_, point_count, [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:points offset:0 atIndex:0];
                    [encoder setBuffer:smoothed_normals offset:0 atIndex:1];
                    [encoder setBuffer:levels offset:0 atIndex:2];
                    [encoder setBuffer:keys offset:0 atIndex:3];
                    [encoder setBuffer:indices offset:0 atIndex:4];
                    [encoder setBuffer:contexts offset:0 atIndex:5];
                    [encoder setBuffer:context_centers offset:0 atIndex:6];
                    [encoder setBuffer:moved_points offset:0 atIndex:7];
                    [encoder setBuffer:mls_normals offset:0 atIndex:8];
                    [encoder setBytes:&point_count length:sizeof(point_count) atIndex:9];
                    [encoder setBytes:&hash_mask length:sizeof(hash_mask) atIndex:10];
                    [encoder setBytes:&voxel_size length:sizeof(voxel_size) atIndex:11];
                    [encoder setBytes:&level length:sizeof(level) atIndex:12];
                    [encoder setBytes:&invalidate_on_failure length:sizeof(invalidate_on_failure) atIndex:13];
                });
            }
            auto *moved = static_cast<Float4 *>(moved_points.contents);
            expanded.cloud.points.assign(moved, moved + point_count);
            auto *normal_values = static_cast<Float4 *>(mls_normals.contents);
            last_normals.assign(normal_values, normal_values + point_count);
            if (invalidate_on_failure != 0u && !std::getenv("S20_BENCH_NO_MLS")) {
                std::vector<uint8_t> mls_keep(point_count, 1);
                for (uint32_t index = 0; index < point_count; ++index) {
                    if (last_normals[index].w >= 0.0f) continue;
                    mls_keep[index] = 0;
                    ++result.mls_context_removed_points;
                    if (expanded.core_flags[index] != 0) {
                        ++result.mls_removed_points;
                    }
                }
                if (result.mls_context_removed_points != 0) {
                    compact_block_contexts(expanded, mls_keep);
                    point_count = static_cast<uint32_t>(expanded.cloud.points.size());
                    last_normals.clear();
                    if (point_count == 0) break;
                }
            }
        }
        result.smoothing_seconds = elapsed_seconds(smoothing_started);

        Cloud collapsed;
        collapsed.origins = expanded.cloud.origins;
        collapsed.frame_timestamps = expanded.cloud.frame_timestamps;
        collapsed.frames_loaded = expanded.cloud.frames_loaded;
        collapsed.first_frame = expanded.cloud.first_frame;
        collapsed.last_frame = expanded.cloud.last_frame;
        collapsed.studio_auto_trim = expanded.cloud.studio_auto_trim;
        size_t core_count = static_cast<size_t>(std::count(
            expanded.core_flags.begin(), expanded.core_flags.end(), uint8_t(1)
        ));
        collapsed.points.reserve(core_count);
        collapsed.frame_ids.reserve(core_count);
        collapsed.relative_times.reserve(core_count);
        std::vector<float> collapsed_scores;
        std::vector<Float4> collapsed_normals;
        collapsed_scores.reserve(core_count);
        collapsed_normals.reserve(core_count);
        result.density_level_counts = {0, 0, 0, 0};
        for (size_t index = 0; index < expanded.cloud.points.size(); ++index) {
            if (expanded.core_flags[index] == 0) continue;
            collapsed.points.push_back(expanded.cloud.points[index]);
            collapsed.frame_ids.push_back(expanded.cloud.frame_ids[index]);
            collapsed.relative_times.push_back(expanded.cloud.relative_times[index]);
            collapsed_scores.push_back(expanded.scores[index]);
            collapsed_normals.push_back(last_normals[index]);
            ++result.density_level_counts[expanded.levels[index]];
        }
        cloud = std::move(collapsed);
        result.point_scores = std::move(collapsed_scores);
        result.normals = std::move(collapsed_normals);
        orient_normals_to_trajectory(cloud, result.normals);
        result.noise_mask.assign(cloud.points.size(), 0);
        build_keep_mask(cloud, result, deduplicate);
    }

private:
    id<MTLComputePipelineState> make_pipeline(NSString *name) {
        id<MTLFunction> function = [library_ newFunctionWithName:name];
        if (function == nil) {
            fail("Metal library has no function " + std::string(name.UTF8String));
        }
        NSError *error = nil;
        id<MTLComputePipelineState> pipeline = [device_ newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            fail("failed to create Metal pipeline " + std::string(name.UTF8String)
                 + ": " + std::string(error.localizedDescription.UTF8String));
        }
        return pipeline;
    }

    id<MTLBuffer> make_buffer(const void *bytes, size_t length, NSString *label) {
        if (length == 0) {
            length = 1;
        }
        if (length > device_.maxBufferLength) {
            fail("Metal buffer exceeds the device limit: " + std::string(label.UTF8String));
        }
        id<MTLBuffer> buffer = [device_ newBufferWithBytes:bytes
                                                   length:length
                                                  options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            fail("failed to allocate Metal buffer: " + std::string(label.UTF8String));
        }
        buffer.label = label;
        return buffer;
    }

    id<MTLBuffer> make_zero_buffer(size_t length, NSString *label) {
        if (length == 0) {
            length = 1;
        }
        if (length > device_.maxBufferLength) {
            fail("Metal buffer exceeds the device limit: " + std::string(label.UTF8String));
        }
        id<MTLBuffer> buffer = [device_ newBufferWithLength:length
                                                    options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            fail("failed to allocate Metal buffer: " + std::string(label.UTF8String));
        }
        std::memset(buffer.contents, 0, length);
        buffer.label = label;
        return buffer;
    }

    template <typename Binder>
    void dispatch(id<MTLComputePipelineState> pipeline, uint32_t count, Binder bind) {
        std::string profName=pipeline==ray_pipeline_?"ray_cast":pipeline==classify_pipeline_?"classify":pipeline==temporal_pipeline_?"temporal":pipeline==normal_pipeline_?"normals":pipeline==smooth_normal_pipeline_?"smooth_normals":"mls";
        prof::Scope phase(profName);
        // Command buffers are autoreleased and retain every buffer they
        // encode. Drain a pool per launch so finished subfiles free their
        // GPU memory instead of holding it until the program exits.
        @autoreleasepool {
            id<MTLCommandBuffer> command = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            bind(encoder);
            NSUInteger width = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
            [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            prof::gpu(profName,command.GPUEndTime-command.GPUStartTime,device_.currentAllocatedSize);
            if (command.status == MTLCommandBufferStatusError) {
                fail("Metal command failed: " + std::string(command.error.localizedDescription.UTF8String));
            }
        }
    }

    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLLibrary> library_;
    id<MTLComputePipelineState> ray_pipeline_;
    id<MTLComputePipelineState> classify_pipeline_;
    id<MTLComputePipelineState> temporal_pipeline_;
    id<MTLComputePipelineState> normal_pipeline_;
    id<MTLComputePipelineState> smooth_normal_pipeline_;
    id<MTLComputePipelineState> mls_pipeline_;
};

FilterResult run_partitioned_filter(
    Cloud &cloud,
    MetalRayFilter &filter,
    bool deduplicate,
    bool ray_only
) {
    std::map<std::array<int, 3>, size_t> subfile_counts = list_ray_subfiles(cloud);
    {
        size_t total_points = 0;
        for (const auto &entry : subfile_counts) total_points += entry.second;
        std::cerr << "Ray subfiles total=" << subfile_counts.size() << " points=" << total_points << "\n";
    }
    FilterResult aggregate;
    aggregate.input_points = cloud.points.size();
    Cloud combined;
    copy_cloud_metadata(cloud, combined);
    std::vector<float> combined_scores;
    std::vector<Float4> combined_normals;

    for (const auto &[coordinate, raw_points] : subfile_counts) {
        aggregate.ray_subfile_input_points += raw_points;
        std::cerr << "Ray subfile [" << coordinate[0] << "," << coordinate[1]
                  << "," << coordinate[2] << "] points=" << raw_points;
        if (raw_points < kMinimumRaySubfilePoints) {
            aggregate.sparse_ray_subfile_points += raw_points;
            std::cerr << " skipped (sparse)\n";
            continue;
        }
        std::cerr << "\n";
        ++aggregate.ray_subfile_count;
        @autoreleasepool {
        SpatialSubfile subfile{coordinate, extract_ray_subfile(cloud, coordinate, raw_points)};
        FilterResult part = filter.run(
            subfile.cloud,
            ray_only ? false : deduplicate,
            &cloud,
            &coordinate
        );
        if (!ray_only) {
            filter.smooth(subfile.cloud, part, deduplicate);
        }

        aggregate.device_name = part.device_name;
        aggregate.occupied_voxels += part.occupied_voxels;
        aggregate.noise_points += part.noise_points;
        aggregate.pre_smooth_points += part.pre_smooth_points;
        aggregate.expanded_context_points += part.expanded_context_points;
        aggregate.sparse_block_removed_points += part.sparse_block_removed_points;
        aggregate.temporal_removed_points += part.temporal_removed_points;
        aggregate.mls_context_removed_points += part.mls_context_removed_points;
        aggregate.mls_removed_points += part.mls_removed_points;
        aggregate.hash_capacity = std::max(aggregate.hash_capacity, part.hash_capacity);
        aggregate.insert_seconds += part.insert_seconds;
        aggregate.ray_seconds += part.ray_seconds;
        aggregate.classify_seconds += part.classify_seconds;
        aggregate.dedup_seconds += part.dedup_seconds;
        aggregate.density_seconds += part.density_seconds;
        aggregate.temporal_seconds += part.temporal_seconds;
        aggregate.smoothing_seconds += part.smoothing_seconds;
        for (size_t level = 0; level < aggregate.density_level_counts.size(); ++level) {
            aggregate.density_level_counts[level] += part.density_level_counts[level];
        }

        uint64_t written_points = 0;
        for (size_t index = 0; index < subfile.cloud.points.size(); ++index) {
            if (part.keep_mask[index] == 0) continue;
            if (!belongs_to_subfile_core(subfile.cloud.points[index], coordinate)) continue;
            combined.points.push_back(subfile.cloud.points[index]);
            combined.frame_ids.push_back(subfile.cloud.frame_ids[index]);
            combined.relative_times.push_back(subfile.cloud.relative_times[index]);
            combined_scores.push_back(part.point_scores[index]);
            if (!ray_only) combined_normals.push_back(part.normals[index]);
            ++written_points;
        }
        std::cerr << "Ray subfile [" << coordinate[0] << "," << coordinate[1]
                  << "," << coordinate[2] << "] wrote=" << written_points << "\n";
        }
    }

    cloud = std::move(combined);
    aggregate.point_scores = std::move(combined_scores);
    aggregate.normals = std::move(combined_normals);
    aggregate.noise_mask.assign(cloud.points.size(), 0);
    aggregate.keep_mask.assign(cloud.points.size(), 1);
    aggregate.kept_points = cloud.points.size();
    return aggregate;
}

#pragma pack(push, 1)
struct PlyPoint {
    float x;
    float y;
    float z;
    float normal_x;
    float normal_y;
    float normal_z;
    uint16_t intensity;
    double gps_time;
    uint32_t frame;
    float ray_noise_score;
};
#pragma pack(pop)

void write_ply(const fs::path &path, const Cloud &cloud, const FilterResult &result) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        fail("cannot create output point cloud: " + path.string());
    }
    output << "ply\n"
           << "format binary_little_endian 1.0\n"
           << "comment PointClouds Studio native Metal filter port\n"
           << "comment coordinates transformed with FrameOptPose.txt\n"
           << "element vertex " << result.kept_points << "\n"
           << "property float x\n"
           << "property float y\n"
           << "property float z\n"
           << "property float normal_x\n"
           << "property float normal_y\n"
           << "property float normal_z\n"
           << "property ushort intensity\n"
           << "property double gps_time\n"
           << "property uint frame\n"
           << "property float ray_noise_score\n"
           << "end_header\n";

    std::vector<PlyPoint> records;
    records.reserve(static_cast<size_t>(result.kept_points));
    for (size_t index = 0; index < cloud.points.size(); ++index) {
        if (result.keep_mask[index] == 0) {
            continue;
        }
        const Float4 &point = cloud.points[index];
        Float4 normal = result.normals.size() == cloud.points.size()
            ? result.normals[index]
            : Float4{0.0f, 0.0f, 0.0f, 0.0f};
        uint32_t frame = cloud.frame_ids[index];
        float source_intensity = std::clamp(point.w, 0.0f, 255.0f);
        // Direct translation of CloudProcessCPU::intensityStretch.
        float stretched_intensity = source_intensity > 50.0f
            ? (source_intensity - 50.0f) * 55.0f / 205.0f + 200.0f
            : source_intensity * 200.0f / 50.0f;
        records.push_back({
            point.x,
            point.y,
            point.z,
            normal.x,
            normal.y,
            normal.z,
            static_cast<uint16_t>(stretched_intensity),
            // Studio writes the registered frame timestamp to LAS.  Curvature's
            // millisecond offset is used by temporal filtering, not export time.
            cloud.frame_timestamps[frame],
            frame,
            result.point_scores[index],
        });
    }
    output.write(
        reinterpret_cast<const char *>(records.data()),
        static_cast<std::streamsize>(records.size() * sizeof(PlyPoint))
    );
    if (!output) {
        fail("failed while writing point cloud: " + path.string());
    }
}

void write_stats(
    const fs::path &path,
    const Cloud &cloud,
    const FilterResult &result,
    const Options &options,
    double total_seconds
) {
    std::ofstream output(path);
    if (!output) {
        fail("cannot create stats file: " + path.string());
    }
    output << std::setprecision(12)
           << "{\n"
           << "  \"format\": \"studio-metal-filter-v4\",\n"
           << "  \"implementation_status\": \"native-metal-port-of-recovered-cuda-semantics\",\n"
           << "  \"compatibility_note\": \"Recovered Studio settings include independent fifty-metre ray subfiles with five-centimetre overlap, nearest-to-voxel-center output selection, trajectory-based temporal thresholds, one-metre density cores with five-centimetre halos, sparse-core skipping, and the CUDA eight-sweep Jacobi normal solver; residual CUDA-versus-Metal floating-point order can still differ\",\n"
           << "  \"metal_device\": \"" << result.device_name << "\",\n"
           << "  \"input_frames\": " << cloud.frames_loaded << ",\n"
           << "  \"first_frame\": " << cloud.first_frame << ",\n"
           << "  \"last_frame\": " << cloud.last_frame << ",\n"
           << "  \"studio_auto_trim\": " << (cloud.studio_auto_trim ? "true" : "false") << ",\n"
           << "  \"input_points\": " << result.input_points << ",\n"
           << "  \"ray_subfile_count\": " << result.ray_subfile_count << ",\n"
           << "  \"ray_subfile_input_points_with_overlap\": " << result.ray_subfile_input_points << ",\n"
           << "  \"sparse_ray_subfile_points\": " << result.sparse_ray_subfile_points << ",\n"
           << "  \"occupied_ray_voxels\": " << result.occupied_voxels << ",\n"
           << "  \"noise_points\": " << result.noise_points << ",\n"
           << "  \"pre_smooth_points\": " << result.pre_smooth_points << ",\n"
           << "  \"expanded_context_points\": " << result.expanded_context_points << ",\n"
           << "  \"sparse_block_removed_points\": " << result.sparse_block_removed_points << ",\n"
           << "  \"temporal_removed_points\": " << result.temporal_removed_points << ",\n"
           << "  \"mls_context_removed_points\": " << result.mls_context_removed_points << ",\n"
           << "  \"mls_removed_points\": " << result.mls_removed_points << ",\n"
           << "  \"kept_points\": " << result.kept_points << ",\n"
           << "  \"ray_voxel_size_m\": " << kRayVoxelSize << ",\n"
           << "  \"output_voxel_size_m\": " << kOutputVoxelSize << ",\n"
           << "  \"deduplicated\": " << (options.deduplicate ? "true" : "false") << ",\n"
           << "  \"ray_only\": " << (options.ray_only ? "true" : "false") << ",\n"
           << "  \"density_level_points\": ["
           << result.density_level_counts[0] << ", "
           << result.density_level_counts[1] << ", "
           << result.density_level_counts[2] << ", "
           << result.density_level_counts[3] << "],\n"
           << "  \"hash_capacity\": " << result.hash_capacity << ",\n"
           << "  \"timing_seconds\": {\n"
           << "    \"insert\": " << result.insert_seconds << ",\n"
           << "    \"ray_cast\": " << result.ray_seconds << ",\n"
           << "    \"classify\": " << result.classify_seconds << ",\n"
           << "    \"density_assignment\": " << result.density_seconds << ",\n"
           << "    \"temporal_consistency\": " << result.temporal_seconds << ",\n"
           << "    \"normal_and_mls\": " << result.smoothing_seconds << ",\n"
           << "    \"deduplicate\": " << result.dedup_seconds << ",\n"
           << "    \"total\": " << total_seconds << "\n"
           << "  }\n"
           << "}\n";
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string argument = argv[i];
        auto value = [&](const char *name) -> std::string {
            if (i + 1 >= argc) {
                fail(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (argument == "--input") {
            options.input = value("--input");
        } else if (argument == "--output") {
            options.output = value("--output");
        } else if (argument == "--kernels") {
            options.kernels = value("--kernels");
        } else if (argument == "--frame-start") {
            options.frame_start = std::stoi(value("--frame-start"));
        } else if (argument == "--frame-end") {
            options.frame_end = std::stoi(value("--frame-end"));
        } else if (argument == "--max-frames") {
            options.max_frames = std::stoi(value("--max-frames"));
        } else if (argument == "--no-deduplicate") {
            options.deduplicate = false;
        } else if (argument == "--all-frames") {
            options.all_frames = true;
        } else if (argument == "--ray-only") {
            options.ray_only = true;
        } else if (argument == "--self-test") {
            options.self_test = true;
        } else {
            fail("unknown argument: " + argument);
        }
    }
    if (options.kernels.empty()) {
        fail("--kernels is required");
    }
    if (!options.self_test && (options.input.empty() || options.output.empty())) {
        fail("--input and --output are required");
    }
    if (options.all_frames) {
        if (options.frame_start >= 0 || options.frame_end >= 0) {
            fail("--all-frames cannot be combined with an explicit frame range");
        }
        options.frame_start = 0;
    }
    return options;
}

void run_self_test(const fs::path &kernels) {
    Cloud cloud;
    cloud.origins.push_back({0.0f, 0.0f, 0.0f, 0.0f});
    cloud.frame_timestamps.push_back(1000.0);
    cloud.frames_loaded = 1;
    cloud.points.push_back({1.0f, 0.0f, 0.0f, 10.0f});
    // With Studio's 30 m distance reference and 0.1 hit odds, twenty
    // coincident two-metre rays make the penetrated voxel exceed 0.2.
    for (int i = 0; i < 20; ++i) {
        cloud.points.push_back({2.0f, 0.0f, 0.0f, 20.0f});
    }
    cloud.frame_ids.assign(cloud.points.size(), 0);
    cloud.relative_times.assign(cloud.points.size(), 0.0f);

    MetalRayFilter filter(kernels);
    FilterResult result = filter.run(cloud, true);
    if (result.noise_mask[0] != 1) {
        fail("self-test failed: penetrated one-metre endpoint was not classified as noise");
    }
    if (result.noise_mask[1] != 0) {
        fail("self-test failed: supported two-metre endpoint was classified as noise");
    }
    if (result.kept_points != 1) {
        fail("self-test failed: 5 mm deduplication did not retain exactly one survivor");
    }
    std::cout << "Metal self-test passed on " << result.device_name
              << "; penetrated score=" << result.point_scores[0]
              << ", supported score=" << result.point_scores[1] << "\n";
}

}  // namespace

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            Options options = parse_options(argc, argv);
            if (options.self_test) {
                run_self_test(options.kernels);
                return 0;
            }

            auto total_started = std::chrono::steady_clock::now();
            fs::create_directories(options.output);
            Cloud cloud = load_cloud(options);
            std::cerr << "Running Studio-compatible partitioned ray evidence on "
                      << cloud.points.size() << " points\n";
            MetalRayFilter filter(options.kernels);
            if (!options.ray_only) {
                std::cerr << "Running adaptive temporal cleanup, normals, and two MLS iterations\n";
            }
            FilterResult result = run_partitioned_filter(
                cloud, filter, options.deduplicate, options.ray_only
            );
            double total_seconds = elapsed_seconds(total_started);

            write_ply(options.output / "filtered.ply", cloud, result);
            write_stats(options.output / "filter_stats.json", cloud, result, options, total_seconds);
            std::cout << "Kept " << result.kept_points << " of " << cloud.points.size()
                      << " points; Metal compute took "
                      << (result.ray_seconds + result.temporal_seconds + result.smoothing_seconds)
                      << " s\n";
            return 0;
        } catch (const std::exception &error) {
            std::cerr << "error: " << error.what() << "\n";
            return 1;
        }
    }
}

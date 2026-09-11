#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

namespace {

struct CameraParameters {
    float center[3];
    float center_error[3];
    float rotation[9];
    float coefficients[6];
    float a11;
    float a12;
    float a22;
    float u0;
    float v0;
    float max_angle;
    uint32_t width;
    uint32_t height;
    uint32_t point_count;
    uint32_t selected_count;
    uint32_t depth_width;
    uint32_t depth_height;
};

struct Projection { float u, v, angle, distance; };

void copy_error(char *destination, size_t capacity, NSString *message) {
    if (!destination || capacity == 0) return;
    const char *text = message ? message.UTF8String : "unknown Metal collector error";
    std::strncpy(destination, text, capacity - 1);
    destination[capacity - 1] = '\0';
}

class Collector {
public:
    Collector(
        const float *points,
        const float *normals,
        uint32_t point_count,
        const char *kernel_path
    ) : point_count_(point_count) {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (!device_) throw std::runtime_error("No Metal GPU");
            NSError *error = nil;
            NSString *path = [NSString stringWithUTF8String:kernel_path];
            NSString *source = [NSString stringWithContentsOfFile:path
                encoding:NSUTF8StringEncoding error:&error];
            if (!source) throw std::runtime_error(error.description.UTF8String);
            MTLCompileOptions *options = [MTLCompileOptions new];
            if (@available(macOS 15.0, *)) {
                options.languageVersion = MTLLanguageVersion3_2;
                options.mathMode = MTLMathModeSafe;
                options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
            } else {
                throw std::runtime_error("Metal collector requires Metal language 3.2");
            }
            id<MTLLibrary> library = [device_ newLibraryWithSource:source options:options error:&error];
            if (!library) throw std::runtime_error(error.description.UTF8String);
            project_ = pipeline(library, @"project_points");
            pack_depth_keys_ = pipeline(library, @"pack_depth_keys");
            build_depth_high_ = pipeline(library, @"build_depth_high");
            build_depth_low_ = pipeline(library, @"build_depth_low");
            horizontal_minimum_ = pipeline(library, @"horizontal_depth_minimum");
            vertical_minimum_ = pipeline(library, @"vertical_depth_minimum");
            blockers_ = pipeline(library, @"find_blockers");
            visibility_ = pipeline(library, @"test_visibility");
            queue_ = [device_ newCommandQueue];
            if (!queue_) throw std::runtime_error("Metal command queue allocation failed");
            const NSUInteger geometry_bytes = NSUInteger(point_count_) * 3u * sizeof(float);
            points_ = [device_ newBufferWithBytes:points length:geometry_bytes
                options:MTLResourceStorageModeShared];
            normals_ = [device_ newBufferWithBytes:normals length:geometry_bytes
                options:MTLResourceStorageModeShared];
            if (!points_ || !normals_) throw std::runtime_error("Metal geometry allocation failed");
            allocated_bytes_ = uint64_t(points_.length) + uint64_t(normals_.length);
        }
    }

    int project(
        const uint32_t *selected,
        uint32_t selected_count,
        const float *center,
        const float *center_error,
        const float *rotation,
        const float *coefficients,
        const float *intrinsics,
        uint32_t width,
        uint32_t height,
        float max_angle
    ) {
        @autoreleasepool {
            ensure_point_capacity(std::max(1u, selected_count));
            if (selected_count > 0) {
                std::memcpy(selected_.contents, selected, size_t(selected_count) * sizeof(uint32_t));
            }
            CameraParameters parameters{};
            std::memcpy(parameters.center, center, sizeof(parameters.center));
            std::memcpy(parameters.center_error, center_error, sizeof(parameters.center_error));
            std::memcpy(parameters.rotation, rotation, sizeof(parameters.rotation));
            std::memcpy(parameters.coefficients, coefficients, sizeof(parameters.coefficients));
            parameters.a11 = intrinsics[0];
            parameters.a12 = intrinsics[1];
            parameters.a22 = intrinsics[2];
            parameters.u0 = intrinsics[3];
            parameters.v0 = intrinsics[4];
            parameters.max_angle = max_angle;
            parameters.width = width;
            parameters.height = height;
            parameters.point_count = point_count_;
            parameters.selected_count = selected_count;
            parameters.depth_width = (width + 3u) / 4u;
            parameters.depth_height = (height + 3u) / 4u;
            parameters_ = parameters;

            if (selected_count == 0) {
                selected_count_ = 0;
                return 0;
            }

            id<MTLCommandBuffer> command = [queue_ commandBuffer];
            encode(command, project_, selected_count,
                {points_, selected_, projection_, depth_keys_, flags_}, parameters);
            return complete(command, selected_count);
        }
    }

    int finish(uint32_t width, uint32_t height) {
        @autoreleasepool {
            ensure_depth_capacity((width + 3u) / 4u, (height + 3u) / 4u);
            parameters_.width = width;
            parameters_.height = height;
            parameters_.depth_width = (width + 3u) / 4u;
            parameters_.depth_height = (height + 3u) / 4u;
            size_t depth_count = size_t(parameters_.depth_width) * parameters_.depth_height;
            std::fill_n(static_cast<uint32_t *>(depth_high_.contents), depth_count,
                        uint32_t(std::numeric_limits<int32_t>::max()));
            std::fill_n(static_cast<uint32_t *>(depth_low_.contents), depth_count,
                        std::numeric_limits<uint32_t>::max());
            id<MTLCommandBuffer> command = [queue_ commandBuffer];
            encode(command, pack_depth_keys_, selected_count_,
                {selected_, projection_, flags_, depth_keys_}, parameters_);
            encode(command, build_depth_high_, selected_count_,
                {projection_, depth_keys_, flags_, depth_high_}, parameters_);
            encode(command, build_depth_low_, selected_count_,
                {projection_, depth_keys_, flags_, depth_high_, depth_low_}, parameters_);
            encode(command, horizontal_minimum_, uint32_t(depth_count),
                {depth_high_, depth_low_, horizontal_minimum_buffer_}, parameters_);
            encode(command, vertical_minimum_, uint32_t(depth_count),
                {horizontal_minimum_buffer_, neighborhood_minimum_buffer_}, parameters_);
            encode(command, blockers_, selected_count_,
                {projection_, flags_, depth_high_, depth_low_, neighborhood_minimum_buffer_,
                 exact_keys_, blocker_keys_},
                parameters_);
            encode(command, visibility_, selected_count_,
                {points_, normals_, selected_, projection_, exact_keys_, blocker_keys_, flags_},
                parameters_);
            return complete(command, selected_count_);
        }
    }

    int complete(id<MTLCommandBuffer> command, uint32_t selected_count) {
            [command commit];
            [command waitUntilCompleted];
            if (command.status == MTLCommandBufferStatusError) {
                last_error_ = command.error.description.UTF8String;
                return 1;
            }
            gpu_seconds_ += command.GPUEndTime - command.GPUStartTime;
            launches_ += 1;
            selected_count_ = selected_count;
            return 0;
    }

    const Projection *projection() const { return static_cast<const Projection *>(projection_.contents); }
    const uint64_t *depth_keys() const { return static_cast<const uint64_t *>(depth_keys_.contents); }
    const uint64_t *exact_keys() const { return static_cast<const uint64_t *>(exact_keys_.contents); }
    const uint64_t *blocker_keys() const { return static_cast<const uint64_t *>(blocker_keys_.contents); }
    const uint32_t *flags() const { return static_cast<const uint32_t *>(flags_.contents); }
    Projection *mutable_projection() { return static_cast<Projection *>(projection_.contents); }
    uint64_t *mutable_depth_keys() { return static_cast<uint64_t *>(depth_keys_.contents); }
    uint32_t *mutable_flags() { return static_cast<uint32_t *>(flags_.contents); }
    const std::string &last_error() const { return last_error_; }
    double gpu_seconds() const { return gpu_seconds_; }
    uint64_t allocated_bytes() const { return allocated_bytes_; }
    uint32_t launches() const { return launches_; }
    const char *device_name() {
        device_name_ = device_.name.UTF8String;
        return device_name_.c_str();
    }

private:
    id<MTLComputePipelineState> pipeline(id<MTLLibrary> library, NSString *name) {
        NSError *error = nil;
        id<MTLFunction> function = [library newFunctionWithName:name];
        id<MTLComputePipelineState> result = [device_ newComputePipelineStateWithFunction:function
            error:&error];
        if (!result) throw std::runtime_error(error.description.UTF8String);
        return result;
    }

    id<MTLBuffer> make_buffer(NSUInteger length) {
        id<MTLBuffer> result = [device_ newBufferWithLength:std::max<NSUInteger>(length, 1)
            options:MTLResourceStorageModeShared];
        if (!result) throw std::runtime_error("Metal buffer allocation failed");
        return result;
    }

    void ensure_point_capacity(uint32_t count) {
        if (count <= point_capacity_) return;
        point_capacity_ = count;
        selected_ = make_buffer(NSUInteger(count) * sizeof(uint32_t));
        projection_ = make_buffer(NSUInteger(count) * sizeof(Projection));
        depth_keys_ = make_buffer(NSUInteger(count) * sizeof(uint64_t));
        exact_keys_ = make_buffer(NSUInteger(count) * sizeof(uint64_t));
        blocker_keys_ = make_buffer(NSUInteger(count) * sizeof(uint64_t));
        flags_ = make_buffer(NSUInteger(count) * sizeof(uint32_t));
        update_allocated_bytes();
    }

    void ensure_depth_capacity(uint32_t width, uint32_t height) {
        uint64_t count = uint64_t(width) * height;
        if (count <= depth_capacity_) return;
        depth_capacity_ = count;
        depth_high_ = make_buffer(NSUInteger(count) * sizeof(uint32_t));
        depth_low_ = make_buffer(NSUInteger(count) * sizeof(uint32_t));
        horizontal_minimum_buffer_ = make_buffer(NSUInteger(count) * sizeof(uint64_t));
        neighborhood_minimum_buffer_ = make_buffer(NSUInteger(count) * sizeof(uint64_t));
        update_allocated_bytes();
    }

    void update_allocated_bytes() {
        allocated_bytes_ = uint64_t(points_.length) + uint64_t(normals_.length);
        if (selected_) allocated_bytes_ += uint64_t(selected_.length);
        if (projection_) allocated_bytes_ += uint64_t(projection_.length);
        if (depth_keys_) allocated_bytes_ += uint64_t(depth_keys_.length);
        if (exact_keys_) allocated_bytes_ += uint64_t(exact_keys_.length);
        if (blocker_keys_) allocated_bytes_ += uint64_t(blocker_keys_.length);
        if (flags_) allocated_bytes_ += uint64_t(flags_.length);
        if (depth_high_) allocated_bytes_ += uint64_t(depth_high_.length);
        if (depth_low_) allocated_bytes_ += uint64_t(depth_low_.length);
        if (horizontal_minimum_buffer_) {
            allocated_bytes_ += uint64_t(horizontal_minimum_buffer_.length);
        }
        if (neighborhood_minimum_buffer_) {
            allocated_bytes_ += uint64_t(neighborhood_minimum_buffer_.length);
        }
    }

    void encode(
        id<MTLCommandBuffer> command,
        id<MTLComputePipelineState> pipeline,
        uint32_t count,
        std::initializer_list<id<MTLBuffer>> buffers,
        const CameraParameters &parameters
    ) {
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        NSUInteger index = 0;
        for (id<MTLBuffer> buffer : buffers) [encoder setBuffer:buffer offset:0 atIndex:index++];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:index];
        NSUInteger width = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
    }

    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> project_, pack_depth_keys_, build_depth_high_, build_depth_low_;
    id<MTLComputePipelineState> horizontal_minimum_, vertical_minimum_, blockers_;
    id<MTLComputePipelineState> visibility_;
    id<MTLBuffer> points_, normals_, selected_, projection_, depth_keys_, exact_keys_;
    id<MTLBuffer> blocker_keys_, flags_, depth_high_, depth_low_;
    id<MTLBuffer> horizontal_minimum_buffer_, neighborhood_minimum_buffer_;
    uint32_t point_count_ = 0, selected_count_ = 0, point_capacity_ = 0;
    uint64_t depth_capacity_ = 0, allocated_bytes_ = 0;
    double gpu_seconds_ = 0.0;
    uint32_t launches_ = 0;
    CameraParameters parameters_{};
    std::string last_error_, device_name_;
};

}

extern "C" {

void *s20_collector_create(const float *points, const float *normals, uint32_t point_count,
                           const char *kernel_path, char *error, size_t error_capacity) {
    try {
        return new Collector(points, normals, point_count, kernel_path);
    } catch (const std::exception &exception) {
        copy_error(error, error_capacity, [NSString stringWithUTF8String:exception.what()]);
        return nullptr;
    }
}

void s20_collector_destroy(void *handle) { delete static_cast<Collector *>(handle); }

int s20_collector_project(void *handle, const uint32_t *selected, uint32_t selected_count,
                          const float *center, const float *center_error, const float *rotation,
                          const float *coefficients,
                          const float *intrinsics, uint32_t width, uint32_t height, float max_angle,
                          char *error, size_t error_capacity) {
    try {
        Collector *collector = static_cast<Collector *>(handle);
        int result = collector->project(selected, selected_count, center, center_error, rotation,
                                        coefficients, intrinsics, width, height, max_angle);
        if (result) copy_error(error, error_capacity,
            [NSString stringWithUTF8String:collector->last_error().c_str()]);
        return result;
    } catch (const std::exception &exception) {
        copy_error(error, error_capacity, [NSString stringWithUTF8String:exception.what()]);
        return 1;
    }
}

int s20_collector_finish(void *handle, uint32_t width, uint32_t height,
                         char *error, size_t error_capacity) {
    try {
        Collector *collector = static_cast<Collector *>(handle);
        int result = collector->finish(width, height);
        if (result) copy_error(error, error_capacity,
            [NSString stringWithUTF8String:collector->last_error().c_str()]);
        return result;
    } catch (const std::exception &exception) {
        copy_error(error, error_capacity, [NSString stringWithUTF8String:exception.what()]);
        return 1;
    }
}

const void *s20_collector_projection(void *handle) {
    return static_cast<Collector *>(handle)->projection();
}
const void *s20_collector_depth_keys(void *handle) {
    return static_cast<Collector *>(handle)->depth_keys();
}
const void *s20_collector_exact_keys(void *handle) {
    return static_cast<Collector *>(handle)->exact_keys();
}
const void *s20_collector_blocker_keys(void *handle) {
    return static_cast<Collector *>(handle)->blocker_keys();
}
const void *s20_collector_flags(void *handle) { return static_cast<Collector *>(handle)->flags(); }
double s20_collector_gpu_seconds(void *handle) {
    return static_cast<Collector *>(handle)->gpu_seconds();
}
uint64_t s20_collector_allocated_bytes(void *handle) {
    return static_cast<Collector *>(handle)->allocated_bytes();
}
uint32_t s20_collector_launches(void *handle) {
    return static_cast<Collector *>(handle)->launches();
}
const char *s20_collector_device_name(void *handle) {
    return static_cast<Collector *>(handle)->device_name();
}

}

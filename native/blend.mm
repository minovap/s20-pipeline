#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <cmath>
#include <climits>

int main(int argc,const char** argv){@autoreleasepool{
    if(argc!=6){std::cerr<<"blend kernel observations field output stats\n";return 2;}
    NSError* error=nil;id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(!device){std::cerr<<"No Metal GPU\n";return 3;}
    NSString* source=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
    MTLCompileOptions* options=[MTLCompileOptions new];options.fastMathEnabled=NO;
    id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
    if(!library){std::cerr<<[[error description] UTF8String];return 4;}
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"exposure_blend"] error:&error];
    NSData* input=[NSData dataWithContentsOfFile:@(argv[2]) options:NSDataReadingMappedIfSafe error:&error];
    NSData* field=[NSData dataWithContentsOfFile:@(argv[3]) options:NSDataReadingMappedIfSafe error:&error];
    if(!input||!field||input.length==0||input.length%128!=0||input.length/128>UINT32_MAX||field.length==0||field.length%(48*3*4)!=0){std::cerr<<"Invalid input\n";return 5;}
    uint32_t count=uint32_t(input.length/128);
    const float* records=(const float*)input.bytes;
    const size_t images=field.length/(48*3*4);
    for(size_t j=0;j<input.length/sizeof(float);++j)if(!std::isfinite(records[j])){std::cerr<<"Nonfinite candidate";return 5;}
    for(size_t j=0;j<input.length/sizeof(float);j+=8){
      if(records[j+4]<0||records[j+4]>7||records[j+5]<0||records[j+5]>5||records[j+6]<0||records[j+6]>=images||floor(records[j+6])!=records[j+6]||records[j+7]<0){std::cerr<<"Invalid candidate address";return 5;}
    }
    const float* offsets=(const float*)field.bytes;
    for(size_t j=0;j<field.length/sizeof(float);++j)if(!std::isfinite(offsets[j])||fabs(offsets[j])>32){std::cerr<<"Invalid correction";return 5;}
    id<MTLBuffer> observations=[device newBufferWithBytes:input.bytes length:input.length options:MTLResourceStorageModeShared];
    id<MTLBuffer> gains=[device newBufferWithBytes:field.bytes length:field.length options:MTLResourceStorageModeShared];
    id<MTLBuffer> output=[device newBufferWithLength:count*16ull options:MTLResourceStorageModeShared];
    if(!pipeline||!observations||!gains||!output){std::cerr<<"Metal allocation failed\n";return 6;}
    id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];[encoder setComputePipelineState:pipeline];
    [encoder setBuffer:observations offset:0 atIndex:0];[encoder setBuffer:gains offset:0 atIndex:1];[encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    NSUInteger width=MIN((NSUInteger)256,pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];[encoder endEncoding];
    [command commit];[command waitUntilCompleted];
    if(command.status==MTLCommandBufferStatusError){std::cerr<<[[command.error description] UTF8String];return 7;}
    NSData* data=[NSData dataWithBytesNoCopy:output.contents length:output.length freeWhenDone:NO];
    if(![data writeToFile:@(argv[4]) options:NSDataWritingAtomic error:&error])return 8;
    NSDictionary* stats=@{@"gpu_command_s":@(command.GPUEndTime-command.GPUStartTime),@"device":device.name,@"points":@(count),@"allocated_buffer_bytes":@(observations.length+gains.length+output.length),@"threads_per_group":@(width),@"fast_math":@NO};
    NSData* json=[NSJSONSerialization dataWithJSONObject:stats options:NSJSONWritingPrettyPrinted error:&error];
    if(![json writeToFile:@(argv[5]) atomically:YES])return 9;
    std::cout<<[[[NSString alloc]initWithData:json encoding:NSUTF8StringEncoding] UTF8String]<<"\n";
    return 0;
}}

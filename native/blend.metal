#include <metal_stdlib>
using namespace metal;

float3 correction(device const float* field, device const float* obs) {
    float x=obs[4], y=obs[5]; uint ix=uint(x), iy=uint(y), image=uint(obs[6]);
    uint jx=min(ix+1,7u), jy=min(iy+1,5u);float a=x-ix,b=y-iy;
    uint i00=(image*48+iy*8+ix)*3,i10=(image*48+iy*8+jx)*3;
    uint i01=(image*48+jy*8+ix)*3,i11=(image*48+jy*8+jx)*3;
    // Fixed-point bilinear offsets make threshold decisions reproducible across
    // CPU and GPU. Correction precision is 1/256 of an 8-bit channel level.
    int wx=int(floor(a*256+.5f)),wy=int(floor(b*256+.5f));float3 result;
    for(uint k=0;k<3;k++){
        int q00=int(floor(field[i00+k]*256+.5f)),q10=int(floor(field[i10+k]*256+.5f));
        int q01=int(floor(field[i01+k]*256+.5f)),q11=int(floor(field[i11+k]*256+.5f));
        int s=(256-wx)*(256-wy)*q00+wx*(256-wy)*q10+(256-wx)*wy*q01+wx*wy*q11+32768;
        int q=s>=0?s/65536:-((-s+65535)/65536);
        result[k]=float(q)/256;
    }
    return result;
}
float3 corrected(device const float* f, device const float* o) {
    return clamp(floor(float3(o[0],o[1],o[2])*256+.5f)/256+correction(f,o),0.0f,255.0f)/255.0f;
}
kernel void exposure_blend(device const float* observations [[buffer(0)]],
                           device const float* field [[buffer(1)]],
                           device float4* output [[buffer(2)]],
                           constant uint& count [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=count)return;
    device const float* anchor=observations+i*32;
    if(anchor[7]<=0){output[i]=float4(0);return;}
    float3 colors[4]; int3 quantized[4];
    for(uint k=0;k<4;k++){
        colors[k]=corrected(field,anchor+k*8);
        quantized[k]=int3(floor(colors[k]*65280+.5f));
    }
    // Choose a view supported by the other valid observations. An isolated
    // specular highlight should not become the reference for a floor patch.
    uint chosen=0;int bestCost=2147483647;
    if(anchor[3]<22){
        for(uint j=0;j<4;j++){
            device const float* candidate=anchor+j*8;
            if(candidate[7]<.1f*anchor[7]||candidate[7]<=0||candidate[3]>=22)continue;
            int3 cj=quantized[j];int cost=0;
            for(uint l=0;l<4;l++){
                device const float* other=anchor+l*8;
                if(other[7]<.1f*anchor[7]||other[7]<=0||other[3]>=22)continue;
                int3 diff=abs(cj-quantized[l]);
                cost+=min(max(diff.x,max(diff.y,diff.z)),12288);
            }
            if(cost<bestCost){bestCost=cost;chosen=j;}
        }
    }
    device const float* reference=anchor+chosen*8;
    float3 sum=0;float total=0;uint used=0;
    for(uint k=0;k<4;k++){
        device const float* o=anchor+k*8;float3 rgb=colors[k];
        // Integer differences preserve the established consensus decisions.
        int3 diff=abs(quantized[k]-quantized[chosen]);
        float delta=float(max(diff.x,max(diff.y,diff.z)))/256;
        bool good=o[7]>0 && o[7]>=.1f*anchor[7] && o[3]<22 && reference[3]<22 && delta<48;
        if(k==chosen)good=true;
        if(!good)continue;
        float taper=max(0.0f,1-delta*delta/(48*48));
        float weight=sqrt(o[7]/anchor[7])*taper*taper/(1+o[3]*o[3]/(22*22));
        float3 linear=select(pow((rgb+.055f)/1.055f,float3(2.4f)),rgb/12.92f,rgb<=.04045f);
        sum+=weight*linear;total+=weight;used++;
    }
    float3 mean=sum/total;
    float3 srgb=select(1.055f*pow(max(mean,0.0f),float3(1.0f/2.4f))-.055f,mean*12.92f,mean<=.0031308f)*255.0f;
    output[i]=float4(srgb,used>1?1.0f:0.0f);
}

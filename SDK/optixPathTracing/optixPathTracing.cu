//
// Copyright (c) 2021 Margarit Alexandru.
// You can use anything from here but I would like a mention (not obligatory)

#include <optix.h>

#include "optixPathTracing.h"
#include <cuda/helpers.h>
#include "random.h"

#include <sutil/vec_math.h>

extern "C" {
__constant__ Params params;
}

struct CustomPayload {
    float3 color;
    float3 emmision;
    int depth;
    unsigned int seed;
    bool inside;
};

static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1)
{
    const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}


static __forceinline__ __device__ void  packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( float_as_int( p.x ) );
    optixSetPayload_1( float_as_int( p.y ) );
    optixSetPayload_2( float_as_int( p.z ) );
}

static __forceinline__ __device__ CustomPayload* getPayload()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<CustomPayload*>(unpackPointer(u0, u1));
}

static __forceinline__ __device__ void computeRay( uint3 idx, uint3 dim, float2 subpixel_jitter, float3& origin, float3& direction)
{
    const float3 U = params.cam_u;
    const float3 V = params.cam_v;
    const float3 W = params.cam_w;
    const float2 d = 2.0f * make_float2(
            static_cast<float>( idx.x + subpixel_jitter.x) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y + subpixel_jitter.x) / static_cast<float>( dim.y )
            ) - 1.0f;

    origin = params.cam_eye
        + (static_cast<float>(subpixel_jitter.x - 0.5) / static_cast<float>(dim.x) * U
        + static_cast<float>(subpixel_jitter.y - 0.5) / static_cast<float>(dim.y) * V) * 10.0;
    direction = normalize(( d.x * U + d.y * V + W) / length(W) * 4 - origin + params.cam_eye);
}

extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    unsigned int secondary = 0;
    
    int number_of_rays = 0;
    float2 subpixel_jitter;

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    float3 ray_origin, ray_direction;
    
    // MAKE PAYLOAD
    CustomPayload retrivedPayload;
    retrivedPayload.depth = 1;
    float3 color = {0, 0, 0};
    unsigned int u1, u2;
    packPointer(&retrivedPayload, u1, u2);
    
    for (int i = 0; i < params.samples_per_launch; i++) {
        retrivedPayload.inside = false;
        unsigned int seed = tea<4>(idx.y * params.image_width * params.random * params.random + idx.x + i, i);
        retrivedPayload.seed = seed;
        subpixel_jitter = make_float2(rnd(seed), rnd(seed));
        computeRay(idx, dim, subpixel_jitter, ray_origin, ray_direction);
        optixTrace(
            params.handle,
            ray_origin,
            ray_direction,
            0.0f,                // Min intersection distance
            1e16f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask(255), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            u1, u2);
            color += retrivedPayload.color + retrivedPayload.emmision;
            retrivedPayload.color = { 0, 0, 0 };
            retrivedPayload.emmision = { 0, 0, 0 };
    }
    color /= params.samples_per_launch;
    color = { powf(color.x, 1.0 / 3), powf(color.y, 1.0 / 3), powf(color.z, 1.0 / 3) };
    //color = { log2(1 + color.x), log2(1 + color.y), log2(1 + color.z) };
    // Record results in our output 
    uchar4 color_2 = make_color(color);

    params.image[idx.y * params.image_width + idx.x].x = (((unsigned int)params.image[idx.y * params.image_width + idx.x].x * (params.count - 1)) + color_2.x) / params.count;
        
    params.image[idx.y * params.image_width + idx.x].y = (((unsigned int)params.image[idx.y * params.image_width + idx.x].y * (params.count - 1)) + color_2.y) / params.count;
        
    params.image[idx.y * params.image_width + idx.x].z = (((unsigned int)params.image[idx.y * params.image_width + idx.x].z * (params.count - 1)) + color_2.z) / params.count;
}


extern "C" __global__ void __miss__ms()
{
    // PREPARE STUFF
    CustomPayload* payload = getPayload();
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    payload->color = { 0.9 * (1.1f - (float)idx.y / dim.y), 0.9f * (float) idx.y / dim.y, 0.9 };
    payload->color = { 0.0, 0.0, 0.0 };
    payload->emmision = { 0, 0, 0 };
}


static __forceinline__ __device__ void ilum1Albedo(float3 *outDir, float3 *albedo, float3* emmision, material material, float3 jitter, float3 normal) {
    float3 diffuse = material.diffuse;
    *emmision = material.emission;

    float3 rayDirection = optixGetWorldRayDirection();
    *outDir = normal + jitter;
    //*outDir = jitter * dot(jitter, normal);
    *outDir = normalize(*outDir);
    float3 H = normalize(*outDir - rayDirection);
    float3 specular = material.specular * pow(fmaxf(dot(normal, H), 0), material.shininess);
    *albedo = diffuse + specular;
}

static __forceinline__ __device__ void ilum2Albedo(float3* outDir, float3* albedo, float3* emmision, material material, float3 jitter, float3 normal) {
    float3 diffuse = material.diffuse;
    float3 specular = material.specular;
    *emmision = material.emission;

    float3 rayDirection = optixGetWorldRayDirection();

    // SPECULAR REFLECTION POSIBILITY
    float r0 = (1 - material.ior) / (1 + material.ior);
    r0 = r0 * r0;
    r0 = r0 + (1 - r0) * pow(1.0f - dot(normal, -rayDirection), 5.0f);

    if (jitter.x < r0 * 2 - 1) {
        *outDir = reflect(rayDirection, normal);
        // *outDir = *outDir + jitter * (100 - material.shininess) / 100.0;
        *albedo = specular + diffuse;
        // *albedo = specular;
    }
    else {
        *outDir = normal + jitter;
        *albedo = diffuse;
    }

    *outDir = normalize(*outDir);

}

static __forceinline__ __device__ void ilum4Albedo(float3* outDir, float3* albedo, float3* emmision, material material, float3 jitter, float3 normal, CustomPayload *payload) {
    float3 diffuse = material.diffuse;
    float3 specular = material.specular;
    *emmision = material.emission;

    float3 rayDirection = optixGetWorldRayDirection();

    // SPECULAR REFLECTION POSIBILITY
    float r0 = (1 - material.ior) / (1 + material.ior);
    r0 = r0 * r0;
    r0 = r0 + (1 - r0) * pow(1.0f - dot(normal, -rayDirection), 5.0f);
    float sin_theta;
    if (payload->inside)
        sin_theta = 1 / material.ior * sqrt(1 - pow(dot(normal, -rayDirection), 2.0f));
    else 
        sin_theta = material.ior * sqrt(1 - pow(dot(normal, -rayDirection), 2.0f));


    if (jitter.x < r0 * 2 - 1 || sin_theta > 1.0) {
        *outDir = reflect(rayDirection, normal);
        // *outDir = *outDir + jitter * (100 - material.shininess) / 100.0;
        *albedo = specular + diffuse;
        // *albedo = specular;
    }
    else {
        *outDir = -normal + normalize(rayDirection + reflect(normal, -rayDirection)) * 1 / sqrt(1 - pow(sin_theta, 2.0f));
        *albedo = diffuse;
        payload->inside = !payload->inside;
    }

    *outDir = normalize(*outDir);

}

static __forceinline__ __device__ void ilum3Albedo(float3* outDir, float3* albedo, float3* emmision, material material, float3 jitter, float3 normal) {
    *albedo = material.ambient;
    *emmision = material.emission;
    float3 rayDirection = optixGetWorldRayDirection();

    *outDir = reflect(rayDirection, normal);
    *outDir = *outDir + jitter * (100 - material.shininess) / 100.0;
    *outDir = normalize(*outDir);

}

extern "C" __global__ void __closesthit__ch()
{
    // PREPARE STUFF
    CustomPayload* payload = getPayload();
    payload->depth++;
    float3 subpixel_jitter = {1, 1, 1};
    HitGroupData* hg_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

    // GET IN TRIANGLE COORDS
    float u = optixGetTriangleBarycentrics().x;
    float v = optixGetTriangleBarycentrics().y;

    // RAY INFO
    float3 rayDirection = optixGetWorldRayDirection();
    float3 outDirection;
    float3 albedo;
    float3 emmision;

    // DATA ABOUT THE TRIANGLE
    float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    unsigned int primitiveIdx = optixGetPrimitiveIndex();
    uint3 triangleIdx = hg_data->indices[primitiveIdx];

    // VERTEX POS
    float3 v1 = hg_data->vertices[triangleIdx.x];
    float3 v2 = hg_data->vertices[triangleIdx.y];
    float3 v3 = hg_data->vertices[triangleIdx.z];

    // MAKE SURFACE NORMAL
    float3 normal = normalize(cross(v1 - v2, v3 - v2));
    if (dot(normal, -rayDirection) < 0) normal = -normal;
    material mater = hg_data->materials[hg_data->material_indices[primitiveIdx]];


    if (payload->depth < params.max_depth) {

        // MAKE NEW PAYLOAD
        CustomPayload retrivedPayload;
        retrivedPayload.depth = payload->depth;
        retrivedPayload.inside = payload->inside;
        unsigned int u1, u2;
        packPointer(&retrivedPayload, u1, u2);
        // CALC A RANDOM NUMBER
        while (length(subpixel_jitter) > 1)
            subpixel_jitter = make_float3(rnd(payload->seed) * 2 - 1, rnd(payload->seed) * 2 - 1, rnd(payload->seed) * 2 - 1);
        retrivedPayload.seed = payload->seed;

        // MAKE NEW DIRECTION
        switch (mater.illum)
        {
        case 1:
            ilum1Albedo(&outDirection, &albedo, &emmision, mater, subpixel_jitter, normal);
            break;
        case 2:
            ilum2Albedo(&outDirection, &albedo, &emmision, mater, subpixel_jitter, normal);
            break;
        case 3:
            ilum3Albedo(&outDirection, &albedo, &emmision, mater, subpixel_jitter, normal);
            break;
        case 4:
            ilum4Albedo(&outDirection, &albedo, &emmision, mater, subpixel_jitter, normal, &retrivedPayload);
            break;
        default:
            break;
        }
        
        // TRACE DIFFUSE
        
        optixTrace(
            params.handle,
            position,
            outDirection,
            0.001f,                // Min intersection distance
            1e16f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask(255), // Specify always visible
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            u1, u2);
            payload->color = albedo * (retrivedPayload.color + retrivedPayload.emmision);
            payload->emmision = emmision;
    }
    else {
        payload->color = { 0, 0, 0 };
        payload->emmision = emmision;
    }
    
}

extern "C" __global__ void __closesthit__ch__smoke()
{
    // PREPARE STUFF
    CustomPayload* payload = getPayload();
    payload->depth++;
    float3 subpixel_jitter = { 1, 1, 1 };
    HitGroupData* hg_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

    // GET IN TRIANGLE COORDS
    float u = optixGetTriangleBarycentrics().x;
    float v = optixGetTriangleBarycentrics().y;

    // RAY INFO
    float3 rayDirection = optixGetWorldRayDirection();
    float3 outDirection;
    float3 albedo;
    float3 emmision;

    // DATA ABOUT THE TRIANGLE
    float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;
    unsigned int primitiveIdx = optixGetPrimitiveIndex();
    uint3 triangleIdx = hg_data->indices[primitiveIdx];

    // VERTEX POS
    float3 v1 = hg_data->vertices[triangleIdx.x];
    float3 v2 = hg_data->vertices[triangleIdx.y];
    float3 v3 = hg_data->vertices[triangleIdx.z];

    // MAKE SURFACE NORMAL
    float3 normal = normalize(cross(v1 - v2, v3 - v2));
    if (dot(normal, -rayDirection) < 0) normal = -normal;
    material mater = hg_data->materials[hg_data->material_indices[primitiveIdx]];


    if (payload->depth < params.max_depth) {

        // MAKE NEW PAYLOAD
        CustomPayload retrivedPayload;
        retrivedPayload.depth = payload->depth;
        unsigned int u1, u2;
        packPointer(&retrivedPayload, u1, u2);
        // CALC A RANDOM NUMBER
        while (length(subpixel_jitter) > 1)
            subpixel_jitter = make_float3(rnd(payload->seed) * 2 - 1, rnd(payload->seed) * 2 - 1, rnd(payload->seed) * 2 - 1);
        float scatter_chance = min(params.air_density * optixGetRayTmax(), 1.0f);
        float scatter = rnd(payload->seed);
        retrivedPayload.seed = payload->seed;
        // MAKE NEW DIRECTION

        if (scatter_chance < scatter) {
            switch (mater.illum)
            {
            case 1:
                ilum1Albedo(&outDirection, &albedo, &emmision, mater, subpixel_jitter, normal);
                break;
            case 2:
                ilum2Albedo(&outDirection, &albedo, &emmision, mater, subpixel_jitter, normal);
                break;
            case 3:
                ilum3Albedo(&outDirection, &albedo, &emmision, mater, subpixel_jitter, normal);
                break;
            default:
                break;
            }
        }
        else {
            rayDirection = normalize(rayDirection);
            position = optixGetWorldRayOrigin() + scatter * (1.0f / params.air_density) * rayDirection;
            outDirection = normalize(subpixel_jitter);
            albedo = { 1.0, 1.0, 1.0 };
            emmision = { 0, 0, 0 };
        }
        // TRACE DIFFUSE

        optixTrace(
            params.handle,
            position,
            outDirection,
            0.001f,                // Min intersection distance
            1e16f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask(255), // Specify always visible
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            u1, u2);
        payload->color = albedo * (retrivedPayload.color + retrivedPayload.emmision);
        payload->emmision = emmision;
    }
    else {
        float scatter_chance = min(params.air_density * optixGetRayTmax(), 1.0f);
        float scatter = rnd(payload->seed);
        if (scatter_chance < scatter) {
            switch (mater.illum)
            {
            case 1:
                ilum1Albedo(&outDirection, &albedo, &emmision, mater, subpixel_jitter, normal);
                break;
            case 2:
                ilum2Albedo(&outDirection, &albedo, &emmision, mater, subpixel_jitter, normal);
                break;
            case 3:
                ilum3Albedo(&outDirection, &albedo, &emmision, mater, subpixel_jitter, normal);
                break;
            default:
                break;
            }
        }
        else {
            emmision = { 0, 0, 0 };
        }
        payload->color = { 0, 0, 0 };
        payload->emmision = emmision;
    }

}

extern "C" __global__ void __miss__ms__smoke()
{
    // PREPARE STUFF
    CustomPayload* payload = getPayload();
    payload->depth++;
    float3 subpixel_jitter = { 1, 1, 1 };
    HitGroupData* hg_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // RAY INFO
    float3 rayDirection = optixGetWorldRayDirection();
    float3 outDirection;
    float3 albedo;
    float3 emmision;

    // DATA ABOUT THE TRIANGLE
    float3 position = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirection;



    if (payload->depth < params.max_depth) {

        // MAKE NEW PAYLOAD
        CustomPayload retrivedPayload;
        retrivedPayload.depth = payload->depth;
        unsigned int u1, u2;
        packPointer(&retrivedPayload, u1, u2);
        // CALC A RANDOM NUMBER
        while (length(subpixel_jitter) > 1)
            subpixel_jitter = make_float3(rnd(payload->seed) * 2 - 1, rnd(payload->seed) * 2 - 1, rnd(payload->seed) * 2 - 1);
        float scatter_chance = min(params.air_density * optixGetRayTmax(), 1.0f);
        float scatter = rnd(payload->seed);
        retrivedPayload.seed = payload->seed;
        // MAKE NEW DIRECTION

        rayDirection = normalize(rayDirection);
        position = optixGetWorldRayOrigin() + scatter * (1.0f / params.air_density) * rayDirection;
        outDirection = normalize(subpixel_jitter);
        albedo = { 1.0, 1.0, 1.0 };
        emmision = { 0, 0, 0 };

        // TRACE DIFFUSE
        optixTrace(
            params.handle,
            position,
            outDirection,
            0.001f,                // Min intersection distance
            1e16f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask(255), // Specify always visible
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            u1, u2);
        payload->color = albedo * (retrivedPayload.color + retrivedPayload.emmision);
        payload->emmision = emmision;
    }
    else {
        //payload->color = { 0.9 * (1.1f - (float)idx.y / dim.y), 0.9f * (float) idx.y / dim.y, 0.9 };
        payload->color = { 0.0, 0.0, 0.0 };
        payload->emmision = { 0, 0, 0 };
    }


}

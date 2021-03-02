//
// Copyright (c) 2021 Margarit Alexandru.
// You can use anything from here but I would like a mention (not obligatory)

struct material {
    //std::string name;

    float3 ambient;
    float3 diffuse;
    float3 specular;
    float3 transmittance;
    float3 emission;
    float shininess;
    float ior;       // index of refraction
    float dissolve;  // 1 == opaque; 0 == fully transparent
    // illumination model (see http://www.fileformat.info/format/material/)
    int illum;

    int dummy;

    cudaTextureObject_t Kd_map;
    cudaTextureObject_t Ka_map;
    cudaTextureObject_t Ks_map;
    cudaTextureObject_t Ke_map;
};

struct Params
{
    uchar4*                image;
    int           image_width;
    int           image_height;
    float3                 cam_eye;
    float3                 cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
    int samples_per_launch;
    int max_depth;
    int random;
    int count;
    float air_density;
    material AIR;
};


struct RayGenData
{
    // No data needed
};


struct MissData
{
    float3 bg_color;
};


struct HitGroupData
{
    float2* texCoords;
    uint3* indices;
    material* materials;
    int* material_indices;
    float3* normals;
    float3* vertices;
};

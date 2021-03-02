//
// Copyright (c) 2021 Margarit Alexandru.
// You can use anything from here but I would like a mention (not obligatory)

#pragma once
#include <unordered_map>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "MagieTudorica.h"

void loadTex(cudaTextureObject_t* texObj, std::string path);

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;
    glm::vec3 normal;

    bool operator==(const Vertex& other) const {
        return pos == other.pos && color == other.color && texCoord == other.texCoord;
    }
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^
                (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
                (hash<glm::vec2>()(vertex.texCoord) << 1);
        }
    };
}


struct Vertex_float {
    float3 pos;
    float3 color;
    float2 texCoord;
    float3 normal;
    
    void operator=(const Vertex& other) {
        pos.x = other.pos.x;
        pos.y = other.pos.y;
        pos.z = other.pos.z;

        color.x = other.color.x;
        color.y = other.color.y;
        color.z = other.color.z;

        texCoord.x = other.texCoord.x;
        texCoord.y = other.texCoord.y;

        normal.x = other.normal.x;
        normal.y = other.normal.y;
        normal.z = other.normal.z;
    }
};

const std::string MODEL_PATH = "./../../models/CornellBox-Original.obj";
const std::string MATERIAL_PATH = "./../../models";
const std::string TEXTURE_PATH = "./../../textures/Coca-Cola 01.jpg";

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;


void copyMaterialfromTiny(material *myMaterial, tinyobj::material_t materialToCopy) {
    myMaterial->ambient.x = materialToCopy.ambient[0];
    myMaterial->ambient.y = materialToCopy.ambient[1];
    myMaterial->ambient.z = materialToCopy.ambient[2];

    myMaterial->diffuse.x = materialToCopy.diffuse[0];
    myMaterial->diffuse.y = materialToCopy.diffuse[1];
    myMaterial->diffuse.z = materialToCopy.diffuse[2];

    myMaterial->specular.x = materialToCopy.specular[0];
    myMaterial->specular.y = materialToCopy.specular[1];
    myMaterial->specular.z = materialToCopy.specular[2];

    myMaterial->transmittance.x = materialToCopy.transmittance[0];
    myMaterial->transmittance.y = materialToCopy.transmittance[1];
    myMaterial->transmittance.z = materialToCopy.transmittance[2];

    myMaterial->emission.x = materialToCopy.emission[0];
    myMaterial->emission.y = materialToCopy.emission[1];
    myMaterial->emission.z = materialToCopy.emission[2];

    myMaterial->shininess = materialToCopy.shininess;
    myMaterial->ior = materialToCopy.ior;
    myMaterial->dissolve = materialToCopy.dissolve;
    myMaterial->illum = materialToCopy.illum;
    myMaterial->dummy = materialToCopy.dummy;

    if (!materialToCopy.diffuse_texname.empty()) {
        loadTex(&myMaterial->Kd_map, materialToCopy.diffuse_texname);
    }
    else {
        myMaterial->Kd_map = 0;
    }

    if (!materialToCopy.ambient_texname.empty()) {
        loadTex(&myMaterial->Ka_map, materialToCopy.ambient_texname);
    }
    else {
        myMaterial->Ka_map = 0;
    }

    if (!materialToCopy.specular_texname.empty()) {
        loadTex(&myMaterial->Ks_map, materialToCopy.specular_texname);
    }
    else {
        myMaterial->Ks_map = 0;
    }

    if (!materialToCopy.emissive_texname.empty()) {
        loadTex(&myMaterial->Ka_map, materialToCopy.emissive_texname);
    }
    else {
        myMaterial->Ke_map = 0;
    }
}

void addVectorFromTiny(std::vector<material>* myMaterials, const std::vector<tinyobj::material_t>& otherMaterials) {
    material copy;
    for (int i = 0; i < otherMaterials.size(); i++) {
        copyMaterialfromTiny(&copy, otherMaterials[i]);
        myMaterials->push_back(copy);
    }
}


void configureCamera(sutil::Camera& cam, const uint32_t width, const uint32_t height)
{
    cam.setEye({ 0.00f, 1.00f, 4.00f }); //box
    //cam.setEye({ 6.0f, 2.00f, 7.0f }); //car
    //cam.setEye({ 5.0f, 4.00f, 0.0f }); //bulb
    //cam.setEye({ 0.00f, 1.00f, 8.00f });
    //cam.setLookat({ 0.0f, 0.0f, -3.5f });
    cam.setLookat({ 0.0f, 1.0f, 0.0f }); //car
    cam.setUp({ 0.0f, 1.0f, 0.0f });
    //cam.setFovY(60.0f);
    cam.setFovY(45.0f);
    cam.setAspectRatio((float)width / (float)height);
}


void printUsageAndExit(const char* argv0)
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for image output\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n";
    exit(1);
}


static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
        << message << "\n";
}

void takeCareOfCommandLine(std::string* outfile, int* width, int* height, int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h")
        {
            printUsageAndExit(argv[0]);
        }
        else if (arg == "--file" || arg == "-f")
        {
            if (i < argc - 1)
            {
                *outfile = argv[++i];
            }
            else
            {
                printUsageAndExit(argv[0]);
            }
        }
        else if (arg.substr(0, 6) == "--dim=")
        {
            const std::string dims_arg = arg.substr(6);
            sutil::parseDimensions(dims_arg.c_str(), *width, *height);
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit(argv[0]);
        }
    }
}

void loadModel(std::vector<float3>* vertices, std::vector<uint3>* indices,std::vector<float2>* texCoords, std::vector<int>* material_indices, std::vector<material>* myMaterials, std::vector<float3>* normals) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    int switcher = 0;
    std::vector<unsigned int> temp;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str(), MATERIAL_PATH.c_str())) {
        throw std::runtime_error(warn + err);
    }
    std::cout << "GATA OBJLOAD";
    addVectorFromTiny(myMaterials, materials);
    /*
    if (myMaterials->size() == 0) std::cout << "Error";
    else std::cout << "                          " << myMaterials->size() << std::endl;
    */
    std::unordered_map<Vertex, unsigned int> uniqueVertices{};

    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            Vertex vertex{};
            Vertex_float vertexFloat;

            vertex.pos = {
                attrib.vertices[3 * index.vertex_index + 0],
                attrib.vertices[3 * index.vertex_index + 1],
                attrib.vertices[3 * index.vertex_index + 2]
            };

            if (index.normal_index > 0) {
                vertex.normal = {
                    attrib.normals[3 * index.normal_index + 0],
                    attrib.normals[3 * index.normal_index + 1],
                    attrib.normals[3 * index.normal_index + 2]
                };
            }

            if (index.texcoord_index > 0)
                vertex.texCoord = {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    1 - attrib.texcoords[2 * index.texcoord_index + 1]
            };

            vertex.color = { 1.0f, 1.0f, 1.0f };

            if (uniqueVertices.count(vertex) == 0) {
                uniqueVertices[vertex] = static_cast<unsigned int>((*vertices).size());
                vertexFloat = vertex;
                (*vertices).push_back(vertexFloat.pos);
                (*texCoords).push_back(vertexFloat.texCoord);
                (*normals).push_back(vertexFloat.normal);
            }

            temp.push_back(uniqueVertices[vertex]);
        }
        for (int i = 0; i < shape.mesh.material_ids.size(); i++) {
            material_indices->push_back(shape.mesh.material_ids[i]);
            // std::cout << shape.mesh.material_ids[i] << std::endl;
        }
    }
    std::cout << "GATA COPY VERT";
    for (int i = 0; i < temp.size(); i += 3) {
        indices->push_back({  temp[i], temp[i + 1] , temp[i + 2] });
    }
    std::cout << "GATA COPY INDEX";
}

void loadTex(cudaTextureObject_t* texObj, std::string path) {
    // LOAD IMG
    int texWidth;
    int texHeight;
    int texChannels;
    
    stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    std::cout << texChannels << std::endl;
    if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
    }

    cudaArray_t cuArray;

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    

    int32_t pitch = texWidth * texChannels * sizeof(uint8_t);
    cudaMallocArray(&cuArray, &channelDesc, texWidth, texHeight);

    // Copy to device memory some data located at address h_data
    // in host memory 
    CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, pixels, pitch, pitch, texHeight,
        cudaMemcpyHostToDevice));

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc tex_desc;
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 1;
    tex_desc.maxAnisotropy = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode = cudaFilterModePoint;
    tex_desc.borderColor[0] = 1.0f;
    tex_desc.sRGB = 0;

    CUDA_CHECK(cudaCreateTextureObject(texObj, &resDesc, &tex_desc, NULL));
}


void serializeObj(std::string filename, const std::vector<float3>& vertices, const std::vector<uint3>& indices, const std::vector<float2>& texCoords, const std::vector<int>& material_indices, const std::vector<material>& myMaterials, const std::vector<float3>& normals) {
    std::ofstream out(filename, std::ofstream::binary);
    serialize(out, vertices);
    serialize(out, indices);
    serialize(out, texCoords);
    serialize(out, material_indices);
    serialize(out, myMaterials);
    serialize(out, normals);
}

void deSerializeObj(std::string filename, std::vector<float3>& vertices, std::vector<uint3>& indices, std::vector<float2>& texCoords, std::vector<int>& material_indices, std::vector<material>& myMaterials, std::vector<float3>& normals) {
    std::ifstream in(filename, std::ifstream::binary);
    deserialize(in, vertices);
    deserialize(in, indices);
    deserialize(in, texCoords);
    deserialize(in, material_indices);
    deserialize(in, myMaterials);
    deserialize(in, normals);
}
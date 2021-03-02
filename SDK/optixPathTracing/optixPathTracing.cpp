//
// Copyright (c) 2021 Margarit Alexandru.
// You can use anything from here but I would like a mention (not obligatory)

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include "optixPathTracing.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <cstdlib>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include "auxiliarsAndSetters.h"
#include <time.h>
#include <ctime>

material AIR{ { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, 0, 1, 0, 0, 0 };

struct PathTracerState
{
    OptixDeviceContext context = 0;

    OptixTraversableHandle         gas_handle = 0;  // Traversable handle for triangle AS
    CUdeviceptr                    d_gas_output_buffer = 0;  // Triangle AS memory
    CUdeviceptr                    d_vertices = 0;
    CUdeviceptr                    d_texCoords = 0;
    CUdeviceptr                    d_indices = 0;
    CUdeviceptr                    d_material_indices = 0;
    CUdeviceptr                    d_materials = 0;
    CUdeviceptr                    d_normals = 0;

    OptixModule                    ptx_module = 0;
    OptixPipelineCompileOptions    pipeline_compile_options = {};
    OptixPipeline                  pipeline = 0;

    OptixProgramGroup              raygen_prog_group = nullptr;
    OptixProgramGroup              miss_prog_group = nullptr;
    OptixProgramGroup              hitgroup_prog_group = nullptr;

    CUstream                       stream = 0;
    Params                         params;
    CUdeviceptr                    d_param = 0;

    OptixShaderBindingTable        sbt = {};
};

CUdeviceptr allocate_device(const void* data, const size_t size) {
    CUdeviceptr d_ptr = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ptr), size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_ptr),
        data,
        size,
        cudaMemcpyHostToDevice
    ));
    return d_ptr;
}

void loadSceneAndBVH(PathTracerState& state) {
    // ---------------------------------------------------
    // LOADING STUFF IN
    // ---------------------------------------------------
    // Triangle build input: simple list of three vertices
    std::vector<float3> vertices;
    std::vector<float3> normals;
    std::vector<uint3> indices;
    std::vector<float2> texCoords;
    std::vector<int> material_indices;
    std::vector<material> materials;
    std::vector<int> randoms;
    std::vector<cudaTextureObject_t> textures;

    char choice;
    std::cout << "LOAD -> l" << std::endl;
    std::cin >> choice;
    std::cout << std::endl;
    if (choice == 'l') {
        loadModel(&vertices, &indices, &texCoords, &material_indices, &materials, &normals);

        serializeObj("Modelu.meu", vertices, indices, texCoords, material_indices, materials, normals);
    }
    else {
        deSerializeObj("Modelu.meu", vertices, indices, texCoords, material_indices, materials, normals);
    }

    state.d_texCoords = allocate_device(texCoords.data(), texCoords.size() * sizeof(float2));

    state.d_indices = allocate_device(indices.data(), indices.size() * sizeof(int3));

    state.d_material_indices = allocate_device(material_indices.data(), material_indices.size() * sizeof(int));

    state.d_materials = allocate_device(materials.data(), materials.size() * sizeof(material));

    state.d_normals = allocate_device(normals.data(), normals.size() * sizeof(float3));

    state.d_vertices = allocate_device(vertices.data(), sizeof(float3) * vertices.size());

    // ---------------------------------------------------
    // END
    // ---------------------------------------------------

    {
        // Use default options for simplicity.  In a real use case we would want to
        // enable compaction, etc
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        // Our build input is a simple list of non-indexed triangle vertices
        const uint32_t triangle_input_flags[3] = { OPTIX_GEOMETRY_FLAG_NONE, OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT, OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS };
        OptixBuildInput triangle_input = {};
        triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
        triangle_input.triangleArray.vertexBuffers = &state.d_vertices;
        triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_input.triangleArray.numIndexTriplets = static_cast<uint32_t>(indices.size());
        triangle_input.triangleArray.indexBuffer = state.d_indices;
        triangle_input.triangleArray.flags = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords = 1;
        triangle_input.triangleArray.preTransform = 0;
        triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);


        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            state.context,
            &accel_options,
            &triangle_input,
            1, // Number of build inputs
            &gas_buffer_sizes
        ));
        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_temp_buffer_gas),
            gas_buffer_sizes.tempSizeInBytes
        ));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&state.d_gas_output_buffer),
            gas_buffer_sizes.outputSizeInBytes
        ));

        OPTIX_CHECK(optixAccelBuild(
            state.context,
            0,                  // CUDA stream
            &accel_options,
            &triangle_input,
            1,                  // num build inputs
            d_temp_buffer_gas,
            gas_buffer_sizes.tempSizeInBytes,
            state.d_gas_output_buffer,
            gas_buffer_sizes.outputSizeInBytes,
            &state.gas_handle,
            nullptr,            // emitted property list
            0                   // num emitted properties
        ));

        // We can now free the scratch space buffer used during build and the vertex
        // inputs, since they are not needed by our trivial shading method
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
    }
}

void createModule(PathTracerState& state, char* log) {
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues = 6;
    state.pipeline_compile_options.numAttributeValues = 3;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    state.pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    const std::string ptx = sutil::getPtxString(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixPathTracing.cu");
    size_t sizeof_log = sizeof(log);

    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        state.context,
        &module_compile_options,
        &state.pipeline_compile_options,
        ptx.c_str(),
        ptx.size(),
        log,
        &sizeof_log,
        &state.ptx_module
    ));
}

void createProgramGroups(PathTracerState& state, char* log) {
    OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc = {}; //
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = state.ptx_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &raygen_prog_group_desc,
        1,   // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &state.raygen_prog_group
    ));

    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = state.ptx_module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &miss_prog_group_desc,
        1,   // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &state.miss_prog_group
    ));

    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &hitgroup_prog_group_desc,
        1,   // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &state.hitgroup_prog_group
    ));
}

void linkPipeline(PathTracerState& state, char* log) {
    const uint32_t    max_trace_depth = 31;
    OptixProgramGroup program_groups[] = { state.raygen_prog_group, state.miss_prog_group, state.hitgroup_prog_group };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        log,
        &sizeof_log,
        &state.pipeline
    ));

    OptixStackSizes stack_sizes = {};
    for (auto& prog_group : program_groups)
    {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
        0,  // maxCCDepth
        0,  // maxDCDEpth
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state, &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(state.pipeline, direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state, continuation_stack_size,
        1  // maxTraversableDepth
    ));
}

void setUpShaderBindingTable(PathTracerState& state, char* log) {
    CUdeviceptr  raygen_record;
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr miss_record;
    size_t      miss_record_size = sizeof(MissSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
    MissSbtRecord ms_sbt;
    ms_sbt.data = { 0.3f, 0.1f, 0.2f };
    OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_prog_group, &ms_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(miss_record),
        &ms_sbt,
        miss_record_size,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr hitgroup_record;
    size_t      hitgroup_record_size = sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
    HitGroupSbtRecord hg_sbt;

    hg_sbt.data.texCoords = (float2*)state.d_texCoords;
    hg_sbt.data.indices = (uint3*)state.d_indices;
    hg_sbt.data.material_indices = (int*)state.d_material_indices;
    hg_sbt.data.materials = (material*)state.d_materials;
    hg_sbt.data.normals = (float3*)state.d_normals;
    hg_sbt.data.vertices = (float3*)state.d_vertices;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroup_prog_group, &hg_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(hitgroup_record),
        &hg_sbt,
        hitgroup_record_size,
        cudaMemcpyHostToDevice
    ));

    state.sbt.raygenRecord = raygen_record;
    state.sbt.missRecordBase = miss_record;
    state.sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    state.sbt.missRecordCount = 1;
    state.sbt.hitgroupRecordBase = hitgroup_record;
    state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    state.sbt.hitgroupRecordCount = 1;
}

void launch(PathTracerState& state, int width, int height, sutil::CUDAOutputBuffer<uchar4> &output_buffer, char* log) {
    CUDA_CHECK(cudaStreamCreate(&state.stream));

    sutil::Camera cam;
    configureCamera(cam, width, height);

    state.params.image = output_buffer.map();
    state.params.image_width = width;
    state.params.image_height = height;
    state.params.handle = state.gas_handle;
    state.params.cam_eye = cam.eye();
    state.params.samples_per_launch = 1000;
    state.params.max_depth = 31;
    state.params.count = 1;
    state.params.random = rand();
    state.params.air_density = 0.00;
    state.params.AIR = AIR;
    cam.UVWFrame(state.params.cam_u, state.params.cam_v, state.params.cam_w);

    state.d_param = allocate_device(&state.params, sizeof(Params));

    OPTIX_CHECK(optixLaunch(state.pipeline, state.stream, state.d_param, sizeof(Params), &state.sbt, width, height, /*depth=*/1));
    CUDA_SYNC_CHECK();
}

void cleanup(PathTracerState &state, char* log) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_gas_output_buffer)));

    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(state.hitgroup_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.miss_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
    OPTIX_CHECK(optixModuleDestroy(state.ptx_module));

    OPTIX_CHECK(optixDeviceContextDestroy(state.context));
}

int main( int argc, char* argv[] )
{
    srand(time(0));
    std::string outfile;
    int         width  =  1024;
    int         height =  1024;

    takeCareOfCommandLine(&outfile, &width, &height, argc, argv);
    PathTracerState state;
    try
    {
        char log[2048]; // For error reporting from OptiX creation functions


        //
        // Initialize CUDA and create OptiX context
        //

        {
            // Initialize CUDA
            CUDA_CHECK(cudaFree(0));

            // Initialize the OptiX API, loading all API entry points
            OPTIX_CHECK(optixInit());

            // Specify context options
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction = &context_log_cb;
            options.logCallbackLevel = 4;

            // Associate a CUDA context (and therefore a specific GPU) with this
            // device context
            CUcontext cuCtx = 0;  // zero means take the current context
            OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &state.context));
        }

        //
        // accel handling
        //

        loadSceneAndBVH(state);

        //
        // Create module
        //

        createModule(state, log);

        //
        // Create program groups
        //

        createProgramGroups(state, log);

        //
        // Link pipeline
        //

        linkPipeline(state, log);

        //
        // Set up shader binding table
        //
        setUpShaderBindingTable(state, log);

        sutil::CUDAOutputBuffer<uchar4> output_buffer( sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height );

        //
        // launch
        //
        {
            std::clock_t start = std::clock();
            launch(state, width, height, output_buffer, log);

            output_buffer.unmap();
            //
            // Display results
            //
            {
                for (int i = 0; i < 40; i++) {
                    state.params.count++;
                    state.params.random = rand();

                    // COPY PARAMS

                    CUDA_CHECK(cudaMemcpy(
                        reinterpret_cast<void*>(state.d_param),
                        &state.params, sizeof(state.params),
                        cudaMemcpyHostToDevice
                    ));

                    // LAUNCH
                    OPTIX_CHECK(optixLaunch(state.pipeline, state.stream, state.d_param, sizeof(Params), &state.sbt, width, height, /*depth=*/1));
                    CUDA_SYNC_CHECK();

                }
                
                sutil::ImageBuffer buffer;
                buffer.data = output_buffer.getHostPointer();
                buffer.width = width;
                buffer.height = height;
                buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
                std::cout << "TIME : " << (clock() - start) / (double) CLOCKS_PER_SEC << std::endl;
                if (outfile.empty())
                    sutil::displayBufferWindow(argv[0], buffer);
                else
                    sutil::saveImage(outfile.c_str(), buffer, false);
                sutil::saveImage("Output_file", buffer, false);
                    
            }
        }

        //
        // Cleanup
        //
        {
            cleanup(state, log);
        }
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

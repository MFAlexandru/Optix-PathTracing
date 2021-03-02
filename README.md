# Optix-PathTracing

Implementation of path tracing using CUDA and Optix

# Motivation

The implementation is my own personal project, I am quiete inrested in computer graphics and I heard about the technique of raytracing in games so I tried to give it a shot. After a few months of learnig about it i finally tried it and made a CPU ray tracing program. When i tried adding some more complex geometry i ran into a bottle neck so i began to learn CUDA in hopes that I could speed up my program and after rewriting the whole thing in CUDA I found out about Optix. It took the implementation to a much higher level than before with the addition of Geometry accel. structs. Now I still make changes from time to time but I am more intrested in learnig more about real time rendering and Vulkan

# Instalation

Open the project with Cmake, chose a directory for build and run.
You will have to link with GLM, STB and tiny_object_loader

# Results

Currently the program can render dfferent materials, without textures, you can add fog in the whole scene and you can use different materials form fully diffuse to dielectrics and combinations of them. All objects must be 3D no simple planes if it is a dielectric.

# Examples of rendered images:

![picture](Screenshot%202021-03-02%20131329.png)
![picture](Screenshot%202021-03-02%20132404.png)

# References

https://github.com/nvpro-samples/optix_advanced_samples

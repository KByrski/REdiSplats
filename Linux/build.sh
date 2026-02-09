CUDA_DIRECTORY="$HOME/cuda-12.4"
COMPUTE="compute_70"
SM="sm_70"

OPTIX_DIRECTORY="$HOME/OptiX SDK 8.0.0"

"$CUDA_DIRECTORY/bin/nvcc" -std=c++17 -ptx -arch="$COMPUTE" source/shaders_SH0.cu -o output/shaders_SH0.ptx \
    -I"$OPTIX_DIRECTORY/include"

"$CUDA_DIRECTORY/bin/nvcc" -std=c++17 -ptx -arch="$COMPUTE" source/shaders_SH1.cu -o output/shaders_SH1.ptx \
    -I"$OPTIX_DIRECTORY/include"

"$CUDA_DIRECTORY/bin/nvcc" -std=c++17 -ptx -arch="$COMPUTE" source/shaders_SH2.cu -o output/shaders_SH2.ptx \
    -I"$OPTIX_DIRECTORY/include"

"$CUDA_DIRECTORY/bin/nvcc" -std=c++17 -ptx -arch="$COMPUTE" source/shaders_SH3.cu -o output/shaders_SH3.ptx \
    -I"$OPTIX_DIRECTORY/include"

"$CUDA_DIRECTORY/bin/nvcc" -std=c++17 -ptx -arch="$COMPUTE" source/shaders_SH4.cu -o output/shaders_SH4.ptx \
    -I"$OPTIX_DIRECTORY/include"

"$CUDA_DIRECTORY/bin/nvcc" -std=c++17 -rdc=true -Xcompiler -fPIC -c source/Auxiliary.cu -o output/Auxiliary.cu.o \
    -arch="$COMPUTE" -code="$SM" \
    -I"$CUDA_DIRECTORY/targets/x86_64-linux/include" \
    -I"$HOME/OptiX SDK 8.0.0"/include

"$CUDA_DIRECTORY/bin/nvcc" -std=c++17 -rdc=true -Xcompiler -fPIC -c source/device_variables.cu -o output/device_variables.cu.o \
    -arch="$COMPUTE" -code="$SM" \
    -I"$CUDA_DIRECTORY/targets/x86_64-linux/include" \
    -I"$HOME/OptiX SDK 8.0.0"/include

"$CUDA_DIRECTORY/bin/nvcc" -std=c++17 -rdc=true -Xcompiler -fPIC -c source/DumpParametersOptiX.cu -o output/DumpParametersOptiX.cu.o \
    -arch="$COMPUTE" -code="$SM" \
    -I"$CUDA_DIRECTORY/targets/x86_64-linux/include" \
    -I"$HOME/OptiX SDK 8.0.0"/include

"$CUDA_DIRECTORY/bin/nvcc" -std=c++17 -rdc=true -Xcompiler -fPIC -c source/DumpParametersToPLYFileOptiX.cu -o output/DumpParametersToPLYFileOptiX.cu.o \
    -arch="$COMPUTE" -code="$SM" \
    -I"$CUDA_DIRECTORY/targets/x86_64-linux/include" \
    -I"$HOME/OptiX SDK 8.0.0"/include

"$CUDA_DIRECTORY/bin/nvcc" -std=c++17 -rdc=true -Xcompiler -fPIC -c source/GetSceneExtentOptiX.cu -o output/GetSceneExtentOptiX.cu.o \
   -arch="$COMPUTE" -code="$SM" \
    -I"$CUDA_DIRECTORY/targets/x86_64-linux/include" \
    -I"$HOME/OptiX SDK 8.0.0"/include

"$CUDA_DIRECTORY/bin/nvcc" -std=c++17 -rdc=true -Xcompiler -fPIC -c source/InitializeOptiXOptimizer.cu -o output/InitializeOptiXOptimizer.cu.o \
    -arch="$COMPUTE" -code="$SM" \
    -I"$CUDA_DIRECTORY/targets/x86_64-linux/include" \
    -I"$HOME/OptiX SDK 8.0.0"/include

"$CUDA_DIRECTORY/bin/nvcc" -std=c++17 -rdc=true -Xcompiler -fPIC -c source/InitializeOptiXRenderer.cu -o output/InitializeOptiXRenderer.cu.o \
    -arch="$COMPUTE" -code="$SM" \
    -I"$CUDA_DIRECTORY/targets/x86_64-linux/include" \
    -I"$HOME/OptiX SDK 8.0.0"/include

"$CUDA_DIRECTORY/bin/nvcc" -std=c++17 -rdc=true -Xcompiler -fPIC -c source/RenderOptiX.cu -o output/RenderOptiX.cu.o \
    -arch="$COMPUTE" -code="$SM" \
    -I"$CUDA_DIRECTORY/targets/x86_64-linux/include" \
    -I"$HOME/OptiX SDK 8.0.0"/include

"$CUDA_DIRECTORY/bin/nvcc" -std=c++17 -rdc=true -Xcompiler -fPIC -c source/SetConfigurationOptiX.cu -o output/SetConfigurationOptiX.cu.o \
    -arch="$COMPUTE" -code="$SM" \
    -I"$CUDA_DIRECTORY/targets/x86_64-linux/include" \
    -I"$HOME/OptiX SDK 8.0.0"/include

"$CUDA_DIRECTORY/bin/nvcc" -std=c++17 -rdc=true -Xcompiler -fPIC -c source/UpdateGradientOptiX.cu -o output/UpdateGradient.cu.o \
    -arch="$COMPUTE" -code="$SM" \
    -I"$CUDA_DIRECTORY/targets/x86_64-linux/include" \
    -I"$HOME/OptiX SDK 8.0.0"/include

"$CUDA_DIRECTORY/bin/nvcc" -std=c++17 -rdc=true -Xcompiler -fPIC -c source/ZeroGradientOptiX.cu -o output/ZeroGradientOptiX.cu.o \
    -arch="$COMPUTE" -code="$SM" \
    -I"$CUDA_DIRECTORY/targets/x86_64-linux/include" \
    -I"$HOME/OptiX SDK 8.0.0"/include

"$CUDA_DIRECTORY/bin/nvcc" -std=c++17 -dlink \
    output/Auxiliary.cu.o \
    output/device_variables.cu.o \
    output/DumpParametersOptiX.cu.o \
    output/DumpParametersToPLYFileOptiX.cu.o \
    output/GetSceneExtentOptiX.cu.o \
    output/InitializeOptiXOptimizer.cu.o \
    output/InitializeOptiXRenderer.cu.o \
    output/RenderOptiX.cu.o \
    output/SetConfigurationOptiX.cu.o \
    output/UpdateGradient.cu.o \
    output/ZeroGradientOptiX.cu.o \
    -o output/dlink.o \
    -arch="$COMPUTE" -code="$SM" \
    -I"$CUDA_DIRECTORY/targets/x86_64-linux/include" \
    -I"$HOME/OptiX SDK 8.0.0"/include \
    -L"$CUDA_DIRECTORY/targets/x86_64-linux/lib" \
    -lcuda -lcudart -lcufft \
    -Xlinker -rpath="$CUDA_DIRECTORY/targets/x86_64-linux/lib"

g++ -std=c++17 -c source/LoadConfigFile.cpp -o output/LoadConfigFile.cpp.o \
    -I"$CUDA_DIRECTORY/targets/x86_64-linux/include" \
    -I"$OPTIX_DIRECTORY/include"

g++ -std=c++17 -c source/LoadSceneAndCameraCOLMAP.cpp -o output/LoadSceneAndCameraCOLMAP.cpp.o \
    -I"$CUDA_DIRECTORY/targets/x86_64-linux/include" \
    -I"$OPTIX_DIRECTORY/include"

g++ -std=c++17 -c source/LoadSceneAndCamera.cpp -o output/LoadSceneAndCamera.cpp.o \
    -I"$CUDA_DIRECTORY/targets/x86_64-linux/include" \
    -I"$OPTIX_DIRECTORY/include"

g++ -std=c++17 -c source/LoadPLYFile.cpp -o output/LoadPLYFile.cpp.o \
    -I"$CUDA_DIRECTORY/targets/x86_64-linux/include" \
    -I"$OPTIX_DIRECTORY/include"

g++ -std=c++17 -c source/Utils.cpp -o output/Utils.cpp.o \
    -I"$CUDA_DIRECTORY/targets/x86_64-linux/include" \
    -I"$OPTIX_DIRECTORY/include"

g++ -std=c++17 -o output/REdiSplats \
    output/Auxiliary.cu.o \
    output/device_variables.cu.o \
    output/DumpParametersOptiX.cu.o \
    output/DumpParametersToPLYFileOptiX.cu.o \
    output/GetSceneExtentOptiX.cu.o \
    output/InitializeOptiXOptimizer.cu.o \
    output/InitializeOptiXRenderer.cu.o \
    output/RenderOptiX.cu.o \
    output/SetConfigurationOptiX.cu.o \
    output/UpdateGradient.cu.o \
    output/ZeroGradientOptiX.cu.o \
    output/dlink.o \
    output/LoadConfigFile.cpp.o \
    output/LoadSceneAndCameraCOLMAP.cpp.o \
    output/LoadSceneAndCamera.cpp.o \
    output/LoadPLYFile.cpp.o \
    output/Utils.cpp.o \
    source/REdiSplats.cpp \
    -I"$CUDA_DIRECTORY/targets/x86_64-linux/include" \
    -I"$OPTIX_DIRECTORY/include" \
    -L"$CUDA_DIRECTORY/targets/x86_64-linux/lib" \
    -lcuda -lcufft -lcudart \
    -Wl,-rpath="$CUDA_DIRECTORY/targets/x86_64-linux/lib"
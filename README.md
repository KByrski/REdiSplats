# REdiSplats: Ray Tracing for Editable Gaussian Splatting
Krzysztof Byrski, Grzegorz Wilczyński1, Weronika Smolak-Dyżewska, Piotr Borycki, Dawid Baran, Sławomir Tadeja2, Przemysław Spurek <br>

| arXiv |
| :---- |
| REdiSplats: Ray Tracing for Editable Gaussian Splatting [https://arxiv.org/pdf/???](http://arxiv.org/abs/???)|

| ![](assets/ficus.gif) | ![](assets/lego_blender.gif) | ![](assets/lego_viewer.gif) |
|--------------|--------------|--------------|

<p align="center">
  <img src="assets/lego_blender_water.gif" width="50%">
</p>


1. Prerequisites:
-----------------
- Install Visual Studio 2019 Enterprise;
- Install CUDA Toolkit 12.4.1;
- Install NVIDIA OptiX SDK 8.0.0;

2. Compiling the CUDA static library:
------------------------------------
- Create the new CUDA 12.4 Runtime project and name it "RaySplattingFlatCUDA";
- Remove the newly created kernel.cu file with the code template;
- Add all the files from the directory "RaySplattingFlatCUDA" to the project;
- Change project's Configuration to "Release, x64";
- Add OptiX "include" directory path to the project's Include Directories. On our test system, we had to add the following path:

"C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0\include"

- In Properties -> Configuration Properties -> CUDA C/C++ -> Device -> Code Generation type the compute capability and microarchitecture version of your GPU. On our test system with RTX 4070 GPU we added "compute_89,sm_89";
- In Properties -> Configuration Properties -> General -> Configuration Type select "Static library (.lib)";
- For files: "shaders.cu" and "shadersMesh.cu" in Properties -> Configuration Properties -> CUDA C/C++ change the suffix of Compiler Output (obj/cubin) from ".obj" to ".ptx";
- For files: "shaders.cu" and "shadersMesh.cu" in Properties -> Configuration Properties -> CUDA C/C++ -> NVCC Compilation Type select "Generate device-only .ptx file (-ptx)";
- Make the following changes in the file kernel1.cu specifying the location of the compiled *.ptx shader files:

Line 256:
FILE *f = fopen("<location of the compiled *.ptx shader files>/shaders.cu.ptx", "rb");

Line 265:
f = fopen("<location of the compiled *.ptx shader files>/shaders.cu.ptx", "rb");

Line 4558:
FILE *f = fopen("<location of the compiled *.ptx shader files>/shadersMesh.cu.ptx", "rb");

Line 4567:
f = fopen("<location of the compiled *.ptx shader files>/shadersMesh.cu.ptx", "rb");

On our test system, we used the following paths as the string literal passed to the fopen function:

"C:/Users/\<Windows username>/source/repos/RaySplattingFlatCUDA/RaySplattingFlatCUDA/x64/Release/shaders.cu.ptx"
<br>
"C:/Users/\<Windows username>/source/repos/RaySplattingFlatCUDA/RaySplattingFlatCUDA/x64/Release/shadersMesh.cu.ptx"

- Build the project;

3. Compiling the Windows interactive optimizer application:
-----------------------------------------------------------
- Create the new Windows Desktop Application project and name it "RaySplattingFlatWindows";
- Remove the newly generated RaySplattingFlatWindows.cpp file with the code template;
- Add all the files from the directory "RaySplattingFlatWindows" to the project;
- Change project's Configuration to "Release, x64";
- In Properties -> Configuration Properties -> Linker -> Input -> Additional Dependencies add new lines:

"RaySplattingFlatCUDA.lib" <br>
"cuda.lib" <br>
"cudart.lib" <br>
"cufft.lib" <br>

- In Properties -> Configuration Properties -> Linker -> General -> Additional Library Directories add the "lib\x64" path of your CUDA toolkit. On our test system, we had to add the following path:

"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64"

- In Properties -> Configuration Properties -> Linker -> General -> Additional Library Directories add the path of the directory containing your compiled CUDA static library. On our test system, we had to add the following path:

"C:\Users\\\<Windows username>\source\repos\RaySplattingFlatCUDA\x64\Release"

4. Training the first model:
----------------------------
- Create the directory "dump" in the main RaySplattingFlatWindows project's directory and then create the subdirectory "dump\save" in the main RaySplattingFlatWindows project's directory. The application will store the checkpoints here. On our test system we created those directories in the following directory:

"C:\Users\\\<Windows username>\source\repos\RaySplattingFlatWindows\RaySplattingFlatWindows"

- Train the model with GaMeS for some small number of epochs (for example 100) on some dataset (for example: "drjohnson" from "Deep Blending" data sets) to obtain the *.ply file containing the pretrained Gaussians;
- Run 3DGS on the same dataset to obtain the cameras.json file (as we found out the cameras.json returned by the GaMeS incorrect);
- Copy the output file cameras.json to the dataset main directory;
- Convert all of the files in the subdirectory "images" located in the dataset main directory to 24-bit *.bmp file format without changing their names;
- Copy the configuration file "config.txt" to the project's main directory. On our test system we copied it to the following directory:

"C:\Users\\\<Windows username>\source\repos\RaySplattingFlatWindows\RaySplattingFlatWindows"

- In lines: 2 and 3 of the configuration file specify the location of the dataset main directory and the output GaMeS *.ply file obtained after short model pretraining;
- Run the "RaySplattingFlatWindows" project from the Visual Studio IDE;

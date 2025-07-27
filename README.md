# REdiSplats: Ray Tracing for Editable Gaussian Splatting
Krzysztof Byrski, Grzegorz Wilczyński, Weronika Smolak-Dyżewska, Piotr Borycki, Dawid Baran, Sławomir Tadeja, Przemysław Spurek <br>

| arXiv |
| :---- |
| REdiSplats: Ray Tracing for Editable Gaussian Splatting [https://arxiv.org/pdf/2503.12284](http://arxiv.org/abs/2503.12284)|

| ![](assets/ficus.gif) | ![](assets/lego_blender.gif) | ![](assets/lego_viewer.gif) |
|--------------|--------------|--------------|
| ![](assets/lego_blender_water.gif) | ![](assets/fox_up_down.gif) | ![](assets/glasses_shadow.gif) |



## Prerequisites:
-----------------
- Install Visual Studio 2019 Enterprisee;
- Install CUDA Toolkit 12.4.1;
- Install NVIDIA OptiX SDK 8.0.0;

## Compiling the CUDA static library:
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

## Compiling the Windows interactive optimizer application:
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

## Training the first model:
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

# Scripts
Scripts directory contains different scripts that allows to manipulate trained models.

### Setting up the environment
1. First create your conda environment
2. Install pytorch with CUDA support
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
3. Install Nvdiffrast
```bash
pip install ninja imageio PyOpenGL glfw xatlas gdown
pip install git+https://github.com/NVlabs/nvdiffrast/
```
4. Install other dependencies
```bash
pip install -r requirements.txt
```
## generate_mesh.py
This script allows you to get the MeshSplat representation of ReDiSplats model. Output is saved as a `.npz` file, it's output path is `/path/to/ply/meshsplat.npz`.

### Usage
```bash
python generate_mesh.py <ply_path> --opac_threshold <opac_thresh_val> --quant <quant_val>
```
where:
- `ply_path` - path to the input PLY file
- `opac_thresh_val` - opacity threshold for the mesh (e.g. if opac_thresh_val=0.5, only Gaussians with opacity greater than 0.5 will be used to generate the mesh), default value is 0.5
- `quant_val` - quantile value (the bigger the larger the MeshSplat size), default value is 4.0

## convert.py
This script allows you to get the Gaussian representation of ReDiSplats model from collection of frames `.obj` files. Output for each frame is saved as a `.ply` file.

### Usage
```bash
python convert.py --ply_path <ply_path> --output_path <output_path> --frames_path <frames_path> --quant <quant_val>
```
where:
- `ply_path` - path to the original PLY file
- `output_path` - path where the calculated `.ply` files for each frame should be saved
- `frames_path` - path to the directory where `.obj` files are stored
- `quant_val` - quantile value (the bigger the larger the MeshSplat size), default value is 4.0

## render_blender.py
This script allows you to render the MeshSplat representation of ReDiSplats model using Blender. Of course you have to have installed Blender already.

### Usage
```bash
blender --background --python path/to/render_blender.py -- --npz_path <path_to_npz_file> --cam_path <path_to_cameras_file>
```
where:
- `--npz_path` - path to the input `.npz` file
- `--cam_path` - path to the input cameras file

## render_nvdiffrast.py
This script allows you to render the MeshSplat representation of ReDiSplats model using Nvdiffrast.

### Usage
```bash
python render_nvdiffrast.py <npz_path> <cameras_path> --dp_layers <dp_layers_val>
```
where:
- `<npz_path>` - path to the input `.npz` file
- `<cameras_path>` - path to the input cameras file. If this is a path to a `.json` file, then the script will assume you are using Blender dataset. If it's a path to a dataset, then the script will assume you are using Colmap dataset.
- `<dp_layers_val>` - number of depth peeling layers, default value is 50. For NeRF Synthetic datasets use 50, for real scenes use at least 100 (recommended is 200).








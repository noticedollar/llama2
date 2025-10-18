# Optimizing AIMET Quantization workflow for LLaMA-2 7B for SnapDragon devices

## Platform requirements
This notebook is intended to run on a machine with:
  * Ubuntu 22.04
  * NVIDIA driver version equivalent to 525.60.13
  * NVIDIA A100 GPU
  * AIMETPRO version 1.35
  * QAIRT version = 2.35

## Install dependencies
Ensure that you have installed docker and the NVIDIA docker2 runtime: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#docker

## Docker Environment setup
After unpacking this notebook package, use the following command to launch the container:
```bash
docker run --rm --gpus all --name=aimet-dev-torch-gpu -v $PWD:$PWD -w $PWD -v /etc/localtime:/etc/localtime:ro -v /etc/timezone:/etc/timezone:ro --network=host --ulimit core=-1 --ipc=host --shm-size=8G --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -it artifacts.codelinaro.org/codelinaro-aimet/aimet-dev:latest.torch-gpu
```

*NOTE*: Alternatively you can download and build the AIMET development docker using [this Dockerfile](https://github.com/quic/aimet/blob/release-aimet-1.35.0/Jenkins/Dockerfile.torch-gpu).


### Install QPM CLI

1. Download QPM3 for Linux from the Qualcomm site from this link: https://qpm.qualcomm.com/#/main/tools/details/QPM3
2. Run the following command in the Docker container using the downloaded .deb package:
    ```bash
    dpkg -i QualcommPackageManager*.Linux-x86.deb
    ```
3. Check if the package was installed correctly:
    ```bash
    which qpm-cli
    ```
   The output should be `/usr/bin/qpm-cli`.

### Install AIMETPro

1. Download the latest AIMETPro 1.35.0 version for Linux from this link: https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AIMET_Pro
2. Create an installation path config JSON file named `qpm_aimetpro_config.json` with the following content:
    ```json
    {
        "CustomInstallPath" : "/tmp/aimetpro"
    }
    ```
3. Use the QPM CLI to extract tarballs from the downloaded QIK file:
    ```bash
    qpm-cli --extract ./Qualcomm_AIMET_Pro.1.35.0*.Linux-AnyCPU.qik --config qpm_aimetpro_config.json
    ```
4. Extract the needed torch-gpu-pt113-release variant:
    ```bash
    cd /tmp/aimetpro && tar -xvzf aimetpro-release-1.35.0_build-*.torch-gpu-pt113-release.tar.gz
    ```
5. Go to the folder to install required packages for the target variant (torch-gpu-pt113-release):
    ```bash
    cd aimetpro-release-1.35.0_build-*.torch-gpu-pt113-release
    python3 -m pip install cumm-cu117 spconv-cu117 torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
    python3 -m pip install pip/*.whl
    ```

### Install QAIRT

1. Download the latest QAIRT 2.35 for Linux from this link: https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK
2. Create an installation path config JSON file named `qpm_qairt_config.json` with the following content:
    ```json
    {
        "CustomInstallPath" : "/tmp/qairt"
    }
    ```
3. Use the QPM CLI to extract packages from the downloaded QIK file:
    ```bash
    qpm-cli --extract Qualcomm_AI_Runtime_SDK.2.35.*.Linux-AnyCPU.qik --config qpm_qairt_config.json
    ```

## Start Jupyter Notebook Instance

Once AIMETPro and QAIRT has been installed, create an Anaconda or Python virtual environment (detailed instructions for this can be found in the Step-2/3 README).

Set up the QAIRT SDK using the following command:
 ```bash
source <QNN_SDK_ROOT>/bin/envsetup.sh
 ```

Then, use the following command to launch the notebook:

```bash
./launch_nb.sh
```

Once the server starts, you will be presented a URL(s) that you may copy and paste into your web browser.

From the web browser, click on the Jupyter Notebook `qnn_model_prepare.ipynb`

*NOTE*: Make sure to update the `QNN_SDK_ROOT` and `model_id` variables inside of the Step-1 Notebook.
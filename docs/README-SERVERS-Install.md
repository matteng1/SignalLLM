# signal-cli-rest-api (needed)
* Create a signal api configuration directory to save configuration between docker file updates:
```shell
mkdir -p $HOME/.local/share/signal-api
```
* Run the docker container in ***normal*** or ***native*** mode for registration. Expose port 9922 to avoid collisions:<br>
*(Note: sudo because it's easier/faster than configuring docker networking correctly.)*
```shell
sudo docker run -d --name signal-api --restart=always -p 9922:8080 \
      -v $HOME/.local/share/signal-api:/home/.local/share/signal-cli \
      -e 'MODE=native' bbernhard/signal-cli-rest-api
```
* Link device to the Signal account meant to be used by the LLM:<br>
QR-code for app:<br>
http://localhost:9922/v1/qrcodelink?device_name=signal-api<br><br>
* Stop and delete container:
```shell
sudo docker stop signal-api
sudo docker rm signal-api
```
* Run the docker container in ***json-rpc*** mode:
```shell
sudo docker run -d --name signal-api --restart=always -p 9922:8080 \
      -v $HOME/.local/share/signal-api:/home/.local/share/signal-cli \
      -e 'MODE=json-rpc' bbernhard/signal-cli-rest-api
```
* Test connection:
```shell
curl -X GET -H 'accept: application/json' 'http://localhost:9922/v1/about'
```

***For more or better instructions go to https://github.com/bbernhard/signal-cli-rest-api***

# ollama docker image
* CPU only: <br>
*(Note: sudo because it's easier/faster than configuring docker networking correctly.)*
```shell
sudo docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```
* Nvidia GPU: <br>
*(Note: install [Nvidia container toolkit first](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation))*
```shell
sudo docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

* Test connection (Should output version number):
```shell
curl http://localhost:11434/api/version
```

* Download and/or run model interactively:
```shell
docker exec -it ollama ollama run "gemma3:12b"
```

# llama.cpp-server
*(Note: The compilation methods and optimizations may have changed.* <br>
*Below worked a few of weeks ago (Nvidia GPU).* <br><br>
* Probable prerequisites (Debian):
```shell
sudo apt update
sudo apt install pciutils build-essential cmake curl libcurl4-openssl-dev
```
* Create a directory to work in (can be anywhere):
```shell
mkdir ~/llamastuff && cd ~/llamastuff
```

* Compile server with cuda support (Nvidia GPU) and move binary for easier access (from shell).
```shell
# move to working directory
cd ~/llamastuff

# clone repository
git clone https://github.com/ggml-org/llama.cpp.git

# prepare build
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DLLAMA_CURL=ON

# build
cmake --build llama.cpp/build --config Release -j --clean-first

```
* Start the server (Nvidia GPU):<br>
*(Note: There is no "one size fits all" here. Try different settings. --ctx-size, --n-gpu-layers)*
```shell
./llama.cpp/build/bin/llama-server \
    --model ./gemma-3-12b-it-Q4_K_M.gguf \
    --threads 16 \
    --ctx-size 8192 \
    --n-gpu-layers 32 \
    --seed 3407 \
    --prio 2 \
    --temp 1.0 \
    --repeat-penalty 1.0 \
    --min-p 0.01 \
    --top-k 64 \
    --top-p 0.95
```
***For more or better instructions go to [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)***

# whisper.cpp-server - for STT
* Prerequisites for FFMPEG-build (Neededish for transcription of Signal audio format)
```shell
# Debian/Ubuntu
sudo apt install libavcodec-dev libavformat-dev libavutil-dev

# RHEL/Fedora
sudo dnf install libavcodec-free-devel libavformat-free-devel libavutil-free-devel
```
* Clone whisper repo and enter directory
```shell
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
```
* Compile with Nvidia GPU and FFMPEG support (recommended)
```shell
cmake -B build -DGGML_CUDA=1 -D WHISPER_FFMPEG=yes
cmake --build build -j --config Release
```
* Compile **without** Nvidia GPU (much slower) and FFMPEG support.
```shell
cmake -B build -D WHISPER_FFMPEG=yes
cmake --build build -j --config Release
```

* **Download preferred whisper model to models/ folder**
```shell
# Suitable model depends on preferred language, quality and so on. See link below for more information.
```
* Run server (from whisper.cpp folder). **Note port 8088**
```shell
# Settings: 8 threads, language EN, localhost, port 8088
./build/bin/whisper-server -t 8 -l en -m ../models/PREFERRED_WHISPER_MODEL.bin --host 127.0.0.1 --port 8088 --convert
```

***For more or better instructions go to [https://github.com/ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp)***

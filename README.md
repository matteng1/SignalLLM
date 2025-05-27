# SignalLLM
Python app for sending messages to a large language model using the Signal messaging app.<br>
Uses **signal-cli-rest-api** and an OpenAI chat compatible endpoint. Tested with **Ollama** and **llama.cpp-server**. <br>
Supports sending **images** if using a multimodal language model. <br><br>
**Really long conversations with memory enabled may cause OOMs or slowdowns.** <br>
To fix it just delete, edit or move files/memory/SIGNAL_USER_memory.json. Or send /reset.<br><br>
Memory file for each signal user is saved in ./files/memory/SIGNAL_USER_memory.json <br>
Attachments are saved in ./files/attachments/SIGNAL_USER/<br><br>
The LLM API key can be set as an environment variable.<br>
```shell
API_KEY="abcdef12345" python3 signal_llm.py
```
Docker instructions in [README-docker.md](README-docker.md).<br><br>

## Prerequisites
* Follow the instructions in [README-SERVERS-Install.md](docs/README-SERVERS-Install.md) to install signal-cli-rest-api and ollama **or** llamacpp-server.<br><br>
* Install prerequisites (Debian or similar distributions):
```shell
sudo apt-get install python3-aiohttp python3-websockets python3-aiofiles
```
* ***Clone this repository and enter directory.***
```shell
git clone --depth 1 https://github.com/matteng1/SignalLLM.git
cd SignalLLM
```
<br><br>
## Configuration
* Configure your settings in files/config/config.conf (see below for more information):
```javascript
{
    "signal_service": "127.0.0.1:9922",          // signal-cli-rest-api endpoint
    "phone_number": "+12345678910",              // Number of the linked Signal account
    "has_memory": true,                          // Remember previous messages
    "save_memory": true,                         // Continue conversation at a later run
    "save_attachments": true,                    // Save received attachments
    "llm_service_provider": "openai",            // Works with llama.cpp-server and ollama
    "llm_service_url": "http://localhost:11434", // Port 11434 for ollama. 8080 for llamacpp
    "llm_api_key": "",                           // API key.
    "llm_model_options": {"system_prompt": "","model":"gemma3:12b","keep_alive": 30}, // See below
}
```
#### llm_model_options:
**"system_prompt"**: System instructions. Can be a description of the chat companion. If running a multi-language model the language used in the system prompt will be used in the chat.<br>
"Du är en glad person som använder emojis alldeles för ofta." will make the model try to answer in swedish and maintain the described personality.<br><br>
**"model"**:         Which model to interact with. Supported in **ollama**<br><br>
**"keep_alive"**:    How long (in minutes) the model should be loaded in memory. For quick answers the default is set to 30 minutes. Supported in **ollama**<br><br><br>
## Run it
```shell
python3 signal_llm.py
```
## User commands
* **/reset**                           -    Reset memory for the sending user. Keep system prompt
* **/prompt You are a happy robot**    -    Change system prompt to "You are a happy robot" for the specific user. <br>
**Note: the message history is still saved. The model will most likely continue in the current persona if /reset is not sent afterwards.**
<br><br><br>

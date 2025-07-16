<p align="center">
<b>Chatbot Used Internal</b>
</p>

<p align=center>
<a href="https://github.com/Haste171/langchain-chatbot/releases"><img src="https://badgen.net/github/release/Haste171/langchain-chatbot">
<a href="https://gitHub.com/Haste171/langchain-chatbot/graphs/commit-activity"><img src="https://img.shields.io/badge/Maintained%3F-no-red.svg">
<a href="https://github.com/Haste171/langchain-chatbot/blob/master/LICENSE"><img src="https://img.shields.io/github/license/Haste171/langchain-chatbot">
<a href="https://discord.gg/KgmN4FPxxT"><img src="https://dcbadge.vercel.app/api/server/KgmN4FPxxT?compact=true&style=flat"></a>

</a>

<!-- *The LangChain Chatbot is an AI chat interface for the open-source library LangChain. It provides conversational answers to questions about vector ingested documents.* -->
<!-- *Existing repo development is at a freeze while we develop a langchain chat bot website :)* -->


# 🚀 Installation
## Setup Development Environment
```
git clone $REPO
```

### Provision Milvus
We use Milvus as vectore database.  

#### Install Milvus 
Here provide a brief way to install milvus with docker container running with the application.
```bash
curl -O https://raw.githubusercontent.com/milvus-io/milvus/v2.4.1/scripts/standalone_embed.sh
chmod +x standalone_embed.sh
./standalone_embed.sh start
```
#### Install Attu To Manage Milvus
```bash
docker run -d -p 8000:3000 -e MILVUS_URL="$IP:19530" --name attu zilliz/attu:v2.4.0
```
Replace $IP with the real ip of your local environment, like
```bash
ifconfig en0 | grep inet | awk '$1=="inet" {print $2}'
```
Open url `http://localhost:8000` with your browser

### Configure .env
Reference `example.env` to create `.env` file
```python
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
MILVUS_USER=root
MILVUS_PASSWORD=Milvus
MILVUS_DB_NAME=default
MILVUS_URL=http://localhost:19530
MILVUS_TOKEN=root:Milvus
SILICONFLOW_URL=https://api.siliconflow.cn/v1
SILICONFLOW_EMBEDDINGS_MODEL=BAAI/bge-large-zh-v1.5
SILICONFLOW_LLM_MODEL=deepseek-ai/DeepSeek-V3
SILICONFLOW_API_KEY=
```
#### Configure Variables of Milvus
Values of variables start with `MILVUS_` can remain unmodified if you provision Milvus with the above way, otherwise you need to modify them accordingly.  

#### Configure Variables of AI Models
We use `SILICONFLOW` as API provider for AI models, so you need to register and request an api key on `https://cloud.siliconflow.cn/account/ak`. then paste the api key to the value of `SILICONFLOW_API_KEY`.  
You can update `SILICONFLOW_EMBEDDINGS_MODEL` and `SILICONFLOW_LLM_MODEL` with your preferred models.

### Install Python Requirements
```bash
poetry install
```

### Run Program
#### Activate Environment
```bash
poetry shell
```
If you have updated `.env` file, you need to reload it to make effect   
```bash
source .env
```
#### Start Program
```bash
python3 startup.py
```


# 🔧 Key Features

✅ Interactive Ingestion UI for files 

✅ Chat UI with source, temperature, vector_k, and other parameter changing abilities

# 💻 Contributing
If you would like to contribute to the Program, please follow these steps:
1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Implement your changes 
4. Write tests for your changes and ensure that all tests pass
5. Submit a pull request

Some ideas that you can consider for contributing:
- Compatibility with many more files types 
- Compatibility with offline models (HuggingFace, Vicuna, Alpaca)

# 🔨 License

The Chatbot is released under the [MIT License](https://opensource.org/licenses/MIT).


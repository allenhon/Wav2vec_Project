# Wav2vec_Project
Ensure all requirements are installed. The requirements for all tasks in this project are in /asr
````
pip install -r requirements. txt
````
To run the API, first ensure that the asr_api.py file is running either in Docker or in Terminal
To run in Docker:

1) Ensure container is created
````
docker run -d -p 8001:8001 --name asr_api asr-api
````
2) In Docker Desktop, run the container.

Alternatively, you can just run the asr_api.py file in terminal

3) To text if the service is running properly,
````
curl  http://localhost:8001/ping
````
You should get a 'Pong' Response.

4) To process the mp3s, run cv-decode.py in terminal. Change the location of your python environment and cv-decode.py file correspondingly.
````
/Users/allen/Documents/Wav2vec_Project/.venv/bin/python /Users/allen/Documents/Wav2vec_Project/asr/cv-decode.py
````


For asr-train, the 'best' model is saved under wav2vec2-large-960h-cv. You may need the model.safetensors file to run local inference on your machine. I'm not sure of PyTorch automatically generates this file, but the configs are available in config.json.

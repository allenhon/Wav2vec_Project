from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from pydub import AudioSegment
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
import warnings
import numpy as np
from datasets import Dataset
warnings.filterwarnings("ignore")

app = FastAPI()

# Load ASR Model
MODEL_NAME = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
print ("Model loaded")

# Endpoint 1: Ping API
@app.get("/ping")
def ping():
    return {"message": "pong"}

# # Endpoint 2: ASR API for audio transcription
@app.post("/asr")
async def asr_endpoint(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as temp_file:
            temp_file.write(await file.read())
            print("File Read successfully")
        
        # Convert audio to 16kHz mono
        audio = AudioSegment.from_file(temp_filename)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(temp_filename, format="wav")

        print("Audio converted")
        
        # Load audio data from file into a format compatible with datasets
        audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)  # Ensure float32 type

        # Create a dataset from the audio data
        ds = Dataset.from_dict({
            "audio": [{"array": audio_data, "sampling_rate": 16000}]
        })

        # Tokenize input
        input_values = processor(ds[0]["audio"]["array"], return_tensors="pt", padding="longest").input_values  # Batch size 1

        # Retrieve logits
        logits = model(input_values).logits

        # Take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)

        # Get duration of the audio
        duration = len(audio) / 1000.0  # duration in seconds
        
        # Clean up temporary file
        os.remove(temp_filename)

        return JSONResponse(content={
            "transcription": transcription[0],  # Take the first (and only) transcription
            "duration": f"{duration:.2f}"
        })
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
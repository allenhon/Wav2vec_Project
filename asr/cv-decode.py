import requests
import pandas as pd
from time import sleep
from tqdm import tqdm 

# URL of the FastAPI endpoint
url = "http://127.0.0.1:8001/asr"  
CSV_FILE_PATH = "asr/common_voice/cv-valid-dev.csv"
audio_path="asr/common_voice/cv-valid-dev/"
mp3_file_path = "asr/common_voice/cv-valid-dev/cv-valid-dev/sample-000000.mp3"

AUDIO_FOLDER = "/asr/common_voice/cv-valid-dev/"

OUTPUT_CSV_PATH = "asr/common_voice/cv-valid-dev-with-text.csv"

def transcribe_cv_valid_dev():
    df = pd.read_csv(CSV_FILE_PATH)
    df["generated_text"] = ""

    for index,filename in tqdm(df['filename'].items(),total=len(df), desc="Transcribing"):
        fullpath=audio_path+filename
        # print (fullpath)
        
        # Open the file in binary mode
        with open(fullpath, "rb") as audio_file:
            # Create a dictionary of files to send in the request
            files = {"file": ("audio.mp3", audio_file, "audio/mp3")}

            # Send the POST request to the API
            response = requests.post(url, files=files)

        # Print the response from the server
        if response.status_code == 200:
            transcription=response.json().get("transcription")
            # print("Transcription:", transcription)
            # print("Duration:", response.json().get("duration"))
            df.at[index, "generated_text"] = transcription
        else:
            print("Error:", response.json())

        # Save the updated CSV file
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        # print(f"Transcription complete. Updated file saved at {OUTPUT_CSV_PATH}")

        # print (df)
        # sleep (10) # for debug
if __name__ == "__main__":
    transcribe_cv_valid_dev()
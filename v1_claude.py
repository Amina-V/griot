import json
import os
from dotenv import load_dotenv
from deepgram import (DeepgramClient, PrerecordedOptions, FileSource)
from litellm import completion

load_dotenv()
AUDIO_FILE = "/Applications/Griot/tape recorder story.mp3"
DEEPGRAM_API_KEY = os.getenv("DG_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

def main():
    try:
        # STEP 1 Create a Deepgram client using the API key
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)

        with open(AUDIO_FILE, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        # STEP 2: Configure Deepgram options for audio analysis
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
        )

        # STEP 3: Call the transcribe_file method with the text payload and options
        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
        
        # STEP 4: Extract the transcription text from the response
        transcription = response["results"]["channels"][0]["alternatives"][0]["transcript"]

        # STEP 5: Define the prompt for the blog post
        prompt = (
            f"You will be given a transcription from a video, which contains a personal narrative. Your task is to transform this transcription into a short form essay written in the first-person perspective. The essay should have a warm, personable, and comforting tone, creating a beautiful and flowing story from the text provided. Please note the following instructions:\n\n"
            f"1. Maintain Accuracy: Do not add any incorrect facts or details that are not present in the transcription. Only amplify the existing text to make it more engaging and readable.\n"
            f"2. First-Person Perspective: Write the essay as if it is being narrated by the person in the transcription.\n"
            f"3. Story Form: Structure the essay in a way that it reads like a story, with a clear beginning, middle, and end.\n"
            f"4. Warm and Comforting Tone: The tone should be warm and comforting, making the reader feel connected to the narrator's experiences and emotions.\n\n"
            f"Here is the transcription:\n\n"
            f"{transcription}"
        )
        
        # STEP 6: Call the LiteLLM API to generate the blog post
        response = completion(
            model="claude-2",
            messages=[{"content": prompt, "role": "user"}],
            api_key=CLAUDE_API_KEY
        )

        # STEP 7: Extract the generated blog post from the response
        blog_post = response.choices[0].message.content.strip()

        # STEP 8: Print or save the blog post
        print("Generated Blog Post:")
        print(blog_post)
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    main()

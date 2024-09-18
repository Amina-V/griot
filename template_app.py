import os
import json
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions
from litellm import completion

load_dotenv()

app = Flask(__name__)
CORS(app)

DEEPGRAM_API_KEY = os.getenv("DG_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
WEBFLOW_API_KEY = os.getenv("WEBFLOW_API_KEY")
WEBFLOW_SITE_ID = os.getenv("WEBFLOW_SITE_ID")
WEBFLOW_COLLECTION_ID = os.getenv("WEBFLOW_COLLECTION_ID")

def transcribe_audio(audio_file):
    deepgram = DeepgramClient(DEEPGRAM_API_KEY)
    with open(audio_file, "rb") as file:
        buffer_data = file.read()
    payload = {"buffer": buffer_data}
    options = PrerecordedOptions(model="nova-2", smart_format=True)
    response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
    return response["results"]["channels"][0]["alternatives"][0]["transcript"]

def generate_blog_post(transcription):
    prompt = f"""
    Transform this transcription into a short-form essay with a warm, personable tone:
    {transcription}
    
    Output a JSON object with 'title' and 'content' fields.
    """
    response = completion(model="claude-2", messages=[{"content": prompt, "role": "user"}], api_key=CLAUDE_API_KEY)
    return json.loads(response.choices[0].message.content.strip())

def generate_image(prompt):
    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    body = {
        "steps": 40,
        "width": 1024,
        "height": 1024,
        "seed": 0,
        "cfg_scale": 5,
        "samples": 1,
        "text_prompts": [{"text": prompt, "weight": 1}],
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {STABILITY_API_KEY}",
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))
    return response.json()["artifacts"][0]["base64"]

def post_to_webflow(title, content, image_data):
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {WEBFLOW_API_KEY}"
    }

    # First, upload the image
    image_upload_url = f"https://api.webflow.com/sites/{WEBFLOW_SITE_ID}/assets"
    image_upload_payload = {
        "fileName": f"{title.replace(' ', '_')}.png",
        "content": image_data
    }
    image_response = requests.post(image_upload_url, json=image_upload_payload, headers=headers)
    image_url = image_response.json()["url"]

    # Then, create the blog post
    create_item_url = f"https://api.webflow.com/collections/{WEBFLOW_COLLECTION_ID}/items"
    create_item_payload = {
        "fields": {
            "name": title,
            "slug": title.lower().replace(" ", "-"),
            "_archived": False,
            "_draft": False,
            "post-body": f'<img src="{image_url}" alt="{title}">\n\n{content}',
        }
    }
    response = requests.post(create_item_url, json=create_item_payload, headers=headers)
    
    return response.json()["_id"]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/record_story', methods=['POST'])
def record_story():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    audio_file = request.files['audio']
    audio_file.save('temp_audio.wav')
    try:
        transcription = transcribe_audio('temp_audio.wav')
        blog_post = generate_blog_post(transcription)
        image_data = generate_image(f"A warm, comforting image representing the story: {blog_post['title']}")
        
        post_id = post_to_webflow(blog_post['title'], blog_post['content'], image_data)
        
        return jsonify({
            'post_id': post_id,
            'title': blog_post['title'],
            'content': blog_post['content'][:200]  # Return a preview of the content
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
import os
import torch
from flask import Flask, request, render_template, send_file
from model import Generator
from transformers import BertTokenizer
from PIL import Image
import io
import base64

app = Flask(__name__)

# Initialize model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(latent_dim=100, text_embedding_dim=256).to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the latest checkpoint
checkpoint_dir = 'checkpoints'
if os.path.exists(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('generator_')]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
        generator.load_state_dict(torch.load(os.path.join(checkpoint_dir, latest_checkpoint), 
                                           map_location=device))
        generator.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    text = request.form['text']
    
    # Tokenize text
    text_tokens = tokenizer(text, padding='max_length', max_length=32, 
                           truncation=True, return_tensors='pt')
    # Convert to dictionary format
    text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
    
    # Generate image
    with torch.no_grad():
        z = torch.randn(1, 100).to(device)
        fake_image = generator(z, text_tokens)
        
        # Convert to PIL Image
        fake_image = fake_image.squeeze(0).cpu()
        fake_image = (fake_image + 1) / 2.0  # Scale from [-1,1] to [0,1]
        fake_image = fake_image.permute(1, 2, 0).numpy()
        fake_image = (fake_image * 255).astype('uint8')
        image = Image.fromarray(fake_image)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Convert to base64 for display
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()
        
    return {'image': img_base64}

if __name__ == '__main__':
    app.run(debug=True) 
# Text-to-Image GAN for Bird Generation

This project implements a Text-to-Image GAN model trained on the CUB-200-2011 dataset to generate bird images from text descriptions. The model uses BERT for text encoding and a conditional GAN architecture for image generation.

## Notes
1. The data has to be donwloaded first. The insrtuctions are given below.
2. I recommed training the model on Google Cloud or Cloud Servers and saving the model locally to run the app.

## Setup

1. Create a virtual environment and install dependencies:
```bash
pipenv install
```

2. Download and extract the CUB-200-2011 dataset:
```bash
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -xzf CUB_200_2011.tgz
mv CUB_200_2011 data/
```

## Training

To train the model, run:
```bash
python train.py --data_dir data/CUB_200_2011 --batch_size 64 --epochs 100
```

Training parameters can be adjusted using command-line arguments:
- `--data_dir`: Path to the CUB dataset
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.0002)
- `--latent_dim`: Dimension of the latent space (default: 100)
- `--text_embedding_dim`: Dimension of text embeddings (default: 256)
- `--save_interval`: Interval for saving checkpoints (default: 5)

## Web Interface

To start the web interface:
```bash
python app.py
```

Then open your browser and navigate to `http://localhost:5000`. Enter a text description of a bird, and the model will generate a corresponding image.

## Model Architecture

- **Generator**: Uses a series of transposed convolutions to generate images from noise and text embeddings
- **Discriminator**: Uses convolutional layers to classify real/fake images and their text descriptions
- **Text Encoder**: Uses BERT to encode text descriptions into embeddings

## Requirements

See `requirements.txt` for a complete list of dependencies. The main requirements are:
- PyTorch
- Transformers (for BERT)
- Flask
- Pillow
- NumPy
- tqdm 


## Output
<img width="516" alt="Result" src="https://github.com/user-attachments/assets/eec0884f-5074-4a6d-8c9c-55e9c1afa81f" />

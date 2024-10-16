import torch 
import os
import soundfile as sf
import pandas  as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datasets import load_dataset, Audio, Dataset
from transformers import EncodecModel, AutoProcessor
import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution

#EnCodec encoded version of speech commands validation data
X_test = torch.load('data_embeddings/valid_data_embed_SC_X.pt')
y_test = torch.load('data_embeddings/valid_data_embed_SC_y.pt')

class SpeechCommandTransformer(torch.nn.Module):
    # initialize
    def __init__(self, feature_size, seq_length, num_classes, model_dim=256, nhead=8, num_layers=3, dropout=0.1):
        super(SpeechCommandTransformer, self).__init__()
        # 
        self.embedding = torch.nn.Linear(feature_size, model_dim)
#         # Positional encoding
        self.pos_encoder = torch.nn.Parameter(torch.randn(1, seq_length, model_dim))
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = torch.nn.Linear(model_dim, num_classes)

    def forward(self, x):
        # Rearrange dimensions
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x += self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.output_layer(x)
        return x

#classifier model on the embedding space
model = SpeechCommandTransformer(feature_size=128, seq_length=75, num_classes=35, model_dim=256)
model.load_state_dict(torch.load("models/SC_transformer.pth"))


class encoder_classifier(torch.nn.Module):
    # initialize
    def __init__(self, classifier_model, encoder_model, encoder_processor):
        super(encoder_classifier, self).__init__()
        self.classifier_model = classifier_model
        self.encoder_model = encoder_model
        self.encoder_processor = encoder_processor

    def forward(self, x):
        x = self.encoder_processor(raw_audio=x, sampling_rate=24000, return_tensors="pt")
        x = self.encoder_model.encode(x["input_values"], x["padding_mask"])
        x = encodec_model.quantizer.decode(x.audio_codes[0].transpose(0, 1))
        y = self.classifier_model(x)
        return y

# Evaluate the initial model  
model.eval()
with torch.no_grad(): 
    y_pred = model(X_test) 
    _, predicted = torch.max(y_pred, dim=1) 
    accuracy = (predicted == y_test).float().mean() 
    print(f'Test Accuracy: {accuracy.item():.4f}')

print(y_test[:10])
print(predicted[:10])

#EnCodec model
encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz", cache_dir="./models")
encodec_processor = AutoProcessor.from_pretrained("facebook/encodec_24khz", cache_dir="./models")

model_enc_class = encoder_classifier(model, encodec_model, encodec_processor)

n_feats = 1000 #number of features to keep, max: 128*75=9600

with torch.no_grad(): 
    for i in range(5):
        x_audio, samplerate  = sf.read('noise.wav') #noise audio to get corresponding embedding space values
        x_audio = x_audio[:24000]
        inputs_0 = encodec_processor(raw_audio=x_audio, sampling_rate=24000, return_tensors="pt")
        encoder_outputs_0 = encodec_model.encode(inputs_0["input_values"], inputs_0["padding_mask"])
        white_embed = encodec_model.quantizer.decode(encoder_outputs_0.audio_codes[0].transpose(0, 1))

        sample_embed = torch.unsqueeze(X_test[i], 0)
        audio_values = encodec_model.decoder(sample_embed)
        
        sample_embed_c = torch.unsqueeze(X_test[i], 0)

        # Initialize the attribution algorithm with the model
        integrated_gradients = IntegratedGradients(model)
        xid = predicted[i]
        attributions_ig = integrated_gradients.attribute(sample_embed_c, target=xid, n_steps=200)
        attributions_ig = attributions_ig

        attributions_ig = attributions_ig[0].numpy()
        w_i = np.unravel_index(np.argsort(attributions_ig, axis=None), attributions_ig.shape)
        w_i = (w_i[0][:n_feats], w_i[1][:n_feats])


        white_embed[0][w_i] = sample_embed[0][w_i]
        y_pred_post = model(white_embed)
        _, predicted_post = torch.max(y_pred_post, dim=1)
        predicted_post = predicted_post.numpy()[0]
        print("class prediction for explanation: ", predicted_post)

        #to save the decoded explanation audio
        audio_values = encodec_model.decoder(white_embed)       
        sf.write('path/filename', audio_values.detach().numpy().reshape(-1), 24000) 


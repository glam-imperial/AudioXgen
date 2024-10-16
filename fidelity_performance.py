import torch 
import os
import soundfile as sf
import pandas  as pd
import librosa
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datasets import load_dataset, Audio, Dataset
from transformers import EncodecModel, AutoProcessor
import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution

#EnCodec encoded version of speech commands validation data
X_test_org = torch.load('data_embeddings/valid_data_embed_SC_X.pt')
y_test_org = torch.load('data_embeddings/valid_data_embed_SC_y.pt')
print("X_test shape: ", X_test_org.shape)
print("y_test shape: ", y_test_org.shape)

class SpeechCommandTransformer(torch.nn.Module):
    def __init__(self, feature_size, seq_length, num_classes, model_dim=256, nhead=8, num_layers=3, dropout=0.1):
        super(SpeechCommandTransformer, self).__init__()
        # 
        self.embedding = torch.nn.Linear(feature_size, model_dim)
        self.pos_encoder = torch.nn.Parameter(torch.randn(1, seq_length, model_dim))
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = torch.nn.Linear(model_dim, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x += self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.output_layer(x)
        return x

    
class encoder_classifier(torch.nn.Module):
    # initialize
    def __init__(self, classifier_model, encoder_model):
        super(encoder_classifier, self).__init__()
        self.classifier_model = classifier_model
        self.encoder_model = encoder_model

    def forward(self, x):
        x = self.encoder_model.encoder(x)
        y = self.classifier_model(x)
        return y

#classifier model on the embedding space
model = SpeechCommandTransformer(feature_size=128, seq_length=75, num_classes=35, model_dim=256)
model.load_state_dict(torch.load("models/SC_transformer.pth"))

X_test = X_test_org[:]
y_test = y_test_org[:]

# Evaluate the initial model 
model.eval()
with torch.no_grad(): 
    y_pred = model(X_test) 
    _, predicted = torch.max(y_pred, dim=1) 
    accuracy = (predicted == y_test).float().mean() 
    print(f'Test Accuracy: {accuracy.item():.4f}')

y_test_m = predicted.clone()

#EnCodec model
encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz", cache_dir="./models")
encodec_processor = AutoProcessor.from_pretrained("facebook/encodec_24khz", cache_dir="./models")
model_enc_class = encoder_classifier(model, encodec_model)

# Initialize the attribution algorithm with the model
integrated_gradients = IntegratedGradients(model)
integrated_gradients_enc = IntegratedGradients(model_enc_class)

n_steps = 50
keepImp = True #True when measuring fidelity by keeping the most important features, false for faithfulness for deleting most important ones.

accs_all = []
for n_feats in [0, 1600, 3600, 5600, 7600, 8600, 9600]: #number of features to keep or remove, max: 128*75=9600
    print("n_feats: ", n_feats)
    accs_feat = [n_feats]
    for seed in range(1):

        X_test = X_test_org[:]
        y_test = y_test_org[:]

        rand_list = np.arange(9600)
        np.random.shuffle(rand_list)
        rand_idxs = rand_list[:n_feats]
        rand_idxs_row = rand_idxs//75
        rand_idxs_col = rand_idxs%75

        method = "featatt" #random, featatt: feature attribution (IG) is ours
        removal_level = "latent" #input, latent: latent is ours


        with torch.no_grad(): 
            for i in tqdm(range(len(X_test))):
                x_audio, samplerate  = sf.read('noise.wav') #noise audio to get corresponding embedding space values
                x_audio = x_audio[:24000]
                inputs_0 = encodec_processor(raw_audio=x_audio, sampling_rate=24000, return_tensors="pt")
                encoder_outputs_0 = encodec_model.encode(inputs_0["input_values"], inputs_0["padding_mask"])
                white_embed = encodec_model.quantizer.decode(encoder_outputs_0.audio_codes[0].transpose(0, 1))
                sample_embed = torch.unsqueeze(X_test[i], 0)
                if removal_level == "input":
                    if method == "random":
                        audio_values = encodec_model.decoder(sample_embed)[0][0] 
                        rem_ratio = n_feats/9600
                        rem_n = int(rem_ratio*len(audio_values))
                        rem_idx = np.random.choice(len(audio_values), size=rem_n, replace=False)
                        audio_values[rem_idx] = 0
                        removed_inp = encodec_processor(raw_audio=audio_values, sampling_rate=24000, return_tensors="pt")
                        removed_embed = encodec_model.encode(removed_inp["input_values"], removed_inp["padding_mask"])
                        removed_embed = encodec_model.quantizer.decode(removed_embed.audio_codes[0].transpose(0, 1))
                        X_test[i] = removed_embed[0]
                    if method == "featatt":
                        audio_values = encodec_model.decoder(sample_embed)[0][0]
                        audio_values = encodec_processor(raw_audio=audio_values, sampling_rate=24000, return_tensors="pt")
                        audio_values = audio_values["input_values"]
                        xid = y_test_m[i]
                        attributions_ig = integrated_gradients_enc.attribute(audio_values, target=xid, n_steps=n_steps)
                        attributions_ig = attributions_ig[0][0].numpy()

                        if keepImp:
                            sort_idxs = np.argsort(attributions_ig) #ascending order
                            audio_values = audio_values[0][0]
                            rem_ratio = n_feats/9600
                        else:
                            sort_idxs = np.argsort(attributions_ig)[::-1] #descending order
                            audio_values = audio_values[0][0]
                            rem_ratio = 1 - n_feats/9600

                        rem_n = int(rem_ratio*len(audio_values))
                        sort_idxs = sort_idxs[:rem_n].copy()
                        audio_values[sort_idxs] = 0  #remove least important n_feats elements, so keeping most important 9600-n_feats

                        removed_inp = encodec_processor(raw_audio=audio_values, sampling_rate=24000, return_tensors="pt")
                        removed_embed = encodec_model.encode(removed_inp["input_values"], removed_inp["padding_mask"])
                        removed_embed = encodec_model.quantizer.decode(removed_embed.audio_codes[0].transpose(0, 1))
                        X_test[i] = removed_embed[0]
                        
                
                elif removal_level == "latent":
                    sample_embed_c = torch.unsqueeze(X_test[i], 0)

                    xid = y_test_m[i]
                    attributions_ig = integrated_gradients.attribute(sample_embed_c, target=xid, n_steps=n_steps)
                    attributions_ig = attributions_ig[0].numpy()
                    if method == "featatt":
                        w_i = np.unravel_index(np.argsort(attributions_ig, axis=None), attributions_ig.shape)
                        w_i = (w_i[0][:n_feats], w_i[1][:n_feats])


                    if method == "random":
                        w_i = (rand_idxs_row, rand_idxs_col)

                    if keepImp:
                        sample_embed[0][w_i] = white_embed[0][w_i]  #removing the least important n_feats elements, so keeping most important 9600-n_feats
                        X_test[i] = sample_embed[0]
                    else:
                        white_embed[0][w_i] = sample_embed[0][w_i] #keeping the least important n_feats elements, so deleting most important 9600-n_feats
                        X_test[i] = white_embed[0]


        # Evaluate the model after important deletion
        with torch.no_grad(): 
            y_pred = model(X_test) 
            _, predicted = torch.max(y_pred, dim=1) 
            accuracy = (predicted == y_test_m).float().mean() 
            print(f'Test Accuracy after important feature deletion: {accuracy.item():.4f}')

        accs_feat.append(accuracy.item())
    accs_feat = np.array(accs_feat)
    mean_acc = np.mean(accs_feat[1:])
    std_acc = np.std(accs_feat[1:])
    accs_feat = np.append(accs_feat, [mean_acc])
    accs_feat = np.append(accs_feat, [std_acc])
    accs_all.append(accs_feat)

accs_all = np.array(accs_all)
print(accs_all)

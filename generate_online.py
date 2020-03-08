import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
from data.dataloadertest import *
import json


def predict_captions(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}

    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (ids, images) in enumerate(dataloader):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(
                    images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, gen_i in enumerate(caps_gen):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen[ids[i]] = [gen_i.strip(), ]
            pbar.update()

    gen = evaluation.PTBTokenizer.tokenize(gen)
    result = []
    for ix, cap in gen.items():
        result.append({"image_id": int(ix), "caption": cap[0]})
    json.dump(result, open('result3_1.json', 'w'))


if __name__ == '__main__':

    testloader = loader()

    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--features_path', type=str,
                        default='/home/lkk/code/self-critical.pytorch/data/cocobu_att')
    parser.add_argument('--annotation_folder', type=str,
                        default='/home/lkk/code/meshed-memory-transformer/annotations')
    args = parser.parse_args()

    print('Meshed-Memory Transformer Evaluation')

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

    # Model and dataloaders
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': 40})
    decoder = MeshedDecoder(len(text_field.vocab), 54,
                            3, text_field.vocab.stoi['<pad>'])
    model = Transformer(
        text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    data = torch.load(
        '/home/lkk/code/meshed-memory-transformer/saved_models/m2_transformer_best.pth')
    model.load_state_dict(data['state_dict'])

    predict_captions(model, testloader, text_field)

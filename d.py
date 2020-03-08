import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (detections, captions) in enumerate(dataloader):
                detections, captions = detections.to(
                    device), captions.to(device)
                out = model(detections, captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(
                    out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(
                    images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim, text_field):
    # Training with cross-entropy
    model.train()
    scheduler.step()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, captions) in enumerate(dataloader):
            detections, captions = detections.to(device), captions.to(device)
            out = model(detections, captions)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, len(text_field.vocab)),
                           captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step()

    loss = running_loss / len(dataloader)
    return loss


def train_scst(model, dataloader, optim, cider, text_field):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    running_loss = .0
    seq_len = 20
    beam_size = 5

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, caps_gt) in enumerate(dataloader):
            detections = detections.to(device)
            outs, log_probs = model.beam_search(detections, seq_len, text_field.vocab.stoi['<eos>'],
                                                beam_size, out_size=beam_size)
            optim.zero_grad()

            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(
                *([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(
                evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[
                1].astype(np.float32)
            reward = torch.from_numpy(reward).to(
                device).view(detections.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--exp_name', type=str, default='m2_transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--features_path', type=str,
                        default='/home/lkk/code/self-critical.pytorch/data/cocobu_att')
    parser.add_argument('--annotation_folder', type=str,
                        default='/home/lkk/code/meshed-memory-transformer/annotations')
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    args = parser.parse_args()
    print(args)

    # Pipeline for image regions
    image_field = ImageDetectionsField(
        detections_path=args.features_path, max_detections=50, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/',
                   args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits

    if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open(
            'vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(
            open('vocab_%s.pkl' % args.exp_name, 'rb'))

    dict_dataset_train = train_dataset.image_dictionary(
        {'image': image_field, 'text': RawField()})
    ref_caps_train = list(train_dataset.text)  # 566435个句子
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))  # 3513886
    dict_dataset_val = val_dataset.image_dictionary(
        {'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary(
        {'image': image_field, 'text': RawField()})

    print("Training starts")
    for e in range(0, 0 + 100):
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                      drop_last=True)

        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=True,
                                           num_workers=args.workers)
        with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader_train)) as pbar:
            for it, *args in enumerate(dataloader_train):
                detections, selfbboxes, bboxes, captions = args[0][0].to(
                    device), args[0][1].to(device), args[0][2].to(device), args[0][3].to(device)

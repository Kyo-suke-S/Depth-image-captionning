from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import datetime
from tqdm import tqdm
import pickle
#import random
#from torch.utils.data.sampler import SubsetRandomSampler
import PIL
from PIL import Image
import skimage.transform #画像出力用ライブラリ
#from collections import deque

import torch
#from torch import nn
#import torch.nn.functional as F
from torch.utils.data import Subset
#from torch.nn.utils.rnn import pack_padded_sequence
#from torchvision import models
import torchvision.transforms as T
import torchvision.datasets as dataset
#from attention import Soft_Attention, Hard_Attention
from Captioning_models.Base_caption_model.base_caption_models import CNNEncoder_Atten
from Captioning_models.Depth_caption_model.depth_models import Depth_CNN_endoder, CD_RNNDecoderWithSoftAttention, MD_RNNDecoderWithSoftAttention, CD_RNNDecoderWithHardAttention
from Captioning_models.Depth_caption_model.DPT_model import DPT_Depthestimator
from Captioning_models.config import ConfigEval
from Captioning_models.evaluate_metrix import load_textfiles, score
import sys
import gc

import Captioning_models.util as util

def Cdepth_soft_evaluation():
    config = ConfigEval()

    # 辞書（単語→単語ID）の読み込み
    with open(config.ori_word_to_id_file, 'rb') as f:
        word_to_id = pickle.load(f)

    # 辞書（単語ID→単語）の読み込み
    with open(config.ori_id_to_word_file, 'rb') as f:
        id_to_word = pickle.load(f)

    # 辞書サイズを保存
    vocab_size = len(id_to_word)
    
    # 画像のtransformsを定義
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        # ImageNetデータセットの平均と標準偏差
        #T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])

    depth_transforms = T.Compose([T.Resize((224,224))])
    #validationデータセット
    #ランダムな4000個を使用
    val_dataset = dataset.CocoCaptions(root=config.val_img_directory, 
                                         annFile=config.rem_ori_val_anno_file,
                                         #annFile=config.remCOCO_ori_val_anno_file, 
                                         transform=transforms)
    
    make_refs_lambda = lambda x: util.make_refs_dep(x, word_to_id)
    #npy_file = config.index_dir
    #indeces = np.load(npy_file).tolist()
    #print(len(indeces))
    #subcoco = Subset(val_dataset, indeces)
    #print(f"coco4k : {len(coco4k)}")
    #print(len(subcoco))
    print(len(val_dataset))
    val_loader = torch.utils.data.DataLoader(
                        val_dataset,
                        #subcoco, 
                        batch_size=config.batch_size, 
                        num_workers=config.num_workers, 
                        collate_fn=make_refs_lambda)
    
    # モデルの定義
    encoder = CNNEncoder_Atten(config.enc_img_size)
    decoder = CD_RNNDecoderWithSoftAttention(config.dim_attention,
                                      config.dim_embedding, 
                                      config.dim_encoder,
                                      config.dim_hidden,
                                      vocab_size)
    depth_encoder = Depth_CNN_endoder(config.enc_img_size)
    dpt = DPT_Depthestimator()

    encoder.to(config.device)
    decoder.to(config.device)
    depth_encoder.to(config.device)
    dpt.to(config.device)
    dpt.load_weight()

    encoder.eval()
    decoder.eval()
    depth_encoder.eval()
    dpt.eval()


    scores = {"Bleu_1":[], "Bleu_2":[], "Bleu_3":[],"Bleu_4":[], "METEOR":[], "ROUGE_L":[],"CIDEr":[]}
    for i  in range(3):
        # モデルの学習済み重みパラメータをロード
        encoder.load_state_dict(
            torch.load(f'{config.save_directory_Cdep_s_ori}/depth_soft_encoder_best_ori{i}.pth'))
        decoder.load_state_dict(
            torch.load(f'{config.save_directory_Cdep_s_ori}/depth_soft_decoder_best_ori{i}.pth'))
        depth_encoder.load_state_dict(
            torch.load(f'{config.save_directory_Cdep_s_ori}/depth_soft_D_encoder_best_ori{i}.pth'))

        #coco4kでキャプションを評価
        ref_caps = []
        hypos_id = []
        for imgs, imgs_for_dep, captions in tqdm(val_loader):
            #captions: [[],[],..,[]]
            ref_caps.extend(captions)
            imgs = imgs.to(config.device)
            imgs_for_dep = imgs_for_dep.to(config.device)

            depth_maps = dpt(imgs_for_dep)
            depth_maps = depth_maps.unsqueeze(1)
            #depth_map = depth_map.detach()
            depth_maps = dpt.standardize_depth_map(depth_maps)
            depth_maps = depth_transforms(depth_maps)

            depth_features = depth_encoder(depth_maps)

            # エンコーダ・デコーダモデルによる予測
            feature = encoder(imgs)
            sampled_ids = decoder.batch_sample(feature, depth_features, word_to_id)
            hypos_id.append(sampled_ids)

        hypos_id = np.concatenate(hypos_id)
        hypos_word = []
        for ids in hypos_id:
            line = []
            for id in ids:
                w = id_to_word[id]
                if w == "<end>":
                    break
                line.append(w)
            hypos_word.append(" ".join(line))

        #print(f"ref: {len(ref_caps)}, hypo: {len(hypos_word)}")
        ref, hypo = load_textfiles(ref_caps,hypos_word)
        score_result = score(ref, hypo)
        #print(ref[10])
        #print(hypo[10])
        print(score_result)
        for mt, sc in score_result.items():
            scores[mt].append(sc)
        
    dire = config.save_directory_Cdep_s_ori+"/remORI_scores.pkl"
    with open(dire, "wb") as f:
        pickle.dump(scores, f)
       
    del encoder, decoder, depth_encoder, dpt
    gc.collect()

    return

def Cdepth_soft_sample(sample_pic):
    config = ConfigEval()

    # 辞書（単語→単語ID）の読み込み
    with open(config.word_to_id_file, 'rb') as f:
        word_to_id = pickle.load(f)

    # 辞書（単語ID→単語）の読み込み
    with open(config.id_to_word_file, 'rb') as f:
        id_to_word = pickle.load(f)

    # 辞書サイズを保存
    vocab_size = len(id_to_word)
    
    # 画像のtransformsを定義
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        # ImageNetデータセットの平均と標準偏差
        #T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])

    norm_trans = T.Compose([T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    depth_resize = T.Compose([T.Resize((224,224))])
    image_size = 384
    dep_trans = T.Compose([T.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        T.CenterCrop(image_size),
                                        #T.ToTensor(),
                                        T.Normalize(mean=0.5, std=0.5)])
    #validationデータセット
    #ランダムな4000個を使用
    #val_dataset = dataset.CocoCaptions(root=config.val_img_directory, 
    #                                     annFile=config.val_anno_file, 
    #                                     transform=transforms)
    
    #make_refs_lambda = lambda x: util.make_refs_dep(x, word_to_id)
    #indeces = np.load("np_val_index.npy").tolist()
    #coco4k = Subset(val_dataset, indeces)
    #print(f"coco4k : {len(coco4k)}")
    #val_loader = torch.utils.data.DataLoader(
    #                    coco4k, 
    #                    batch_size=config.batch_size, 
    #                    num_workers=config.num_workers, 
    #                    collate_fn=make_refs_lambda)
    
    # モデルの定義
    encoder = CNNEncoder_Atten(config.enc_img_size)
    decoder = CD_RNNDecoderWithSoftAttention(config.dim_attention,
                                      config.dim_embedding, 
                                      config.dim_encoder,
                                      config.dim_hidden,
                                      vocab_size)
    depth_encoder = Depth_CNN_endoder(config.enc_img_size)
    dpt = DPT_Depthestimator()

    encoder.to(config.device)
    decoder.to(config.device)
    depth_encoder.to(config.device)
    dpt.to(config.device)
    dpt.load_weight()

    encoder.eval()
    decoder.eval()
    depth_encoder.eval()
    dpt.eval()

    encoder.load_state_dict(
            torch.load(f'{config.save_directory_Cdep_soft}/depth_soft_encoder_best1.pth'))
    decoder.load_state_dict(
            torch.load(f'{config.save_directory_Cdep_soft}/depth_soft_decoder_best1.pth'))
    depth_encoder.load_state_dict(
            torch.load(f'{config.save_directory_Cdep_soft}/depth_soft_D_encoder_best1.pth'))


    #scores = {"Bleu_1":[], "Bleu_2":[], "Bleu_3":[],"Bleu_4":[], "METEOR":[], "ROUGE_L":[],"CIDEr":[]}
    if sample_pic == "sample1":
        img_directry = config.sample1_dir
    elif sample_pic == "sample2":
        img_directry = config.sample2_dir
    elif sample_pic == "sample3":
        img_directry = config.sample3_dir
    elif sample_pic == "airbus":
        img_directry = config.airbus_dir
    elif sample_pic == "cycling":
        img_directry = config.cycling_dir
    elif sample_pic == "dog":
        img_directry = config.dog_dir
    elif sample_pic == "football":
        img_directry = config.football_dir
    elif sample_pic == "soccer":
        img_directry = config.soccer_dir
    elif sample_pic == "river":
        img_directry = config.river_dir
    elif sample_pic == "seagull":
        img_directry = config.seagull_dir
    elif sample_pic == "bird":
        img_directry = config.bird_dir
    else:
        print("Input correct name")
        return

    for img_file in sorted(
        glob.glob(os.path.join(img_directry, '*.[jp][pn]g'))):

        img = Image.open(img_file).convert("RGB")
        #img = np.asarray(img)
        #print(type(img))
        img = transforms(img)
        img = img.unsqueeze(0)
        img_for_dep = img.detach().clone()
        img = norm_trans(img)
        img_for_dep = dep_trans(img_for_dep)
        img = img.to(config.device)
        img_for_dep = img_for_dep.to(config.device)

        depth_map = dpt(img_for_dep)
        depth_map = depth_map.unsqueeze(1)
        #depth_map = depth_map.detach()
        depth_map = dpt.standardize_depth_map(depth_map)
        depth_map = depth_resize(depth_map)

        feature = encoder(img)
        depth_features = depth_encoder(depth_map)
        sampled_ids, alphas = decoder.sample(feature, depth_features, word_to_id)

        img_plt = Image.open(img_file)
        img_plt = img_plt.resize([224, 224], Image.LANCZOS)
        plt.imshow(img_plt)
        plt.axis('off')
        #plt.show()
        plt.savefig(img_file[:-4] + f'_input.png', bbox_inches='tight')
        plt.clf()

        print(f'入力画像: {os.path.basename(img_file)}')

        sampled_caption = []
        c = 0
        for word_id, alpha in zip(sampled_ids, alphas):
            word = id_to_word[word_id]
            sampled_caption.append(word)

            alpha = alpha.view(
                config.enc_img_size, config.enc_img_size)
            alpha = alpha.to('cpu').numpy()
            alpha = skimage.transform.pyramid_expand(
                alpha, upscale=16, sigma=8)
            
            # タイムステップtの画像をプロット
            plt.imshow(img_plt)
            #plt.text(0, 1, f'{word}', color='black',
            #         backgroundcolor='white', fontsize=12)
            plt.imshow(alpha, alpha=0.8)
            plt.set_cmap(cm.Greys_r)
            plt.axis('off')
            #plt.show()
            plt.savefig(img_file[:-4] + f'_depth_soft_{word}_p{c}.png', bbox_inches='tight')
            plt.clf()
            c += 1
            
            if word == '<end>':
                break
        
        sentence = ' '.join(sampled_caption)
        print(f'出力キャプション: {sentence}')

        # 推定結果を書き込み
        gen_sentence_out = img_file[:-4] + '_depth_soft.txt'
        with open(gen_sentence_out, 'w') as f:
            print(sentence, file=f)

    del encoder, decoder, depth_encoder, dpt
    gc.collect()

    return


#-----------------------------------------------------------------------------------

def Cdepth_hard_evaluation():
    config = ConfigEval()

    # 辞書（単語→単語ID）の読み込み
    with open(config.ori_word_to_id_file, 'rb') as f:
        word_to_id = pickle.load(f)

    # 辞書（単語ID→単語）の読み込み
    with open(config.ori_id_to_word_file, 'rb') as f:
        id_to_word = pickle.load(f)

    # 辞書サイズを保存
    vocab_size = len(id_to_word)
    
    # 画像のtransformsを定義
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        # ImageNetデータセットの平均と標準偏差
        #T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])

    depth_transforms = T.Compose([T.Resize((224,224))])
    #validationデータセット
    #ランダムな4000個を使用
    val_dataset = dataset.CocoCaptions(root=config.val_img_directory,
                                         annFile=config.rem_ori_val_anno_file,
                                         #annFile=config.remCOCO_ori_val_anno_file, 
                                         transform=transforms)
    
    make_refs_lambda = lambda x: util.make_refs_dep(x, word_to_id)
    #npy_file = config.index_dir
    #indeces = np.load(npy_file).tolist()
    #print(len(indeces))
    #subcoco = Subset(val_dataset, indeces)
    #print(f"subcoco : {len(subcoco)}")
    print(len(val_dataset))
    val_loader = torch.utils.data.DataLoader(
                        val_dataset,
                        #subcoco, 
                        batch_size=config.batch_size, 
                        num_workers=config.num_workers, 
                        collate_fn=make_refs_lambda)
    
    # モデルの定義
    encoder = CNNEncoder_Atten(config.enc_img_size)
    decoder = CD_RNNDecoderWithHardAttention(config.dim_attention,
                                      config.dim_embedding, 
                                      config.dim_encoder,
                                      config.dim_hidden,
                                      vocab_size, config.device, config.dropout)
    depth_encoder = Depth_CNN_endoder(config.enc_img_size)
    dpt = DPT_Depthestimator()

    encoder.to(config.device)
    decoder.to(config.device)
    depth_encoder.to(config.device)
    dpt.to(config.device)
    dpt.load_weight()

    encoder.eval()
    decoder.eval()
    depth_encoder.eval()
    dpt.eval()


    scores = {"Bleu_1":[], "Bleu_2":[], "Bleu_3":[],"Bleu_4":[], "METEOR":[], "ROUGE_L":[],"CIDEr":[]}
    for i  in range(3):
        # モデルの学習済み重みパラメータをロード
        encoder.load_state_dict(
            torch.load(f'{config.save_directory_Cdep_hard_ori}/depth_hard_encoder_best_ori{i}.pth'))
        decoder.load_state_dict(
            torch.load(f'{config.save_directory_Cdep_hard_ori}/depth_hard_decoder_best_ori{i}.pth'))
        depth_encoder.load_state_dict(
            torch.load(f'{config.save_directory_Cdep_hard_ori}/depth_hard_D_encoder_best_ori{i}.pth'))

        #coco4kでキャプションを評価
        ref_caps = []
        hypos_id = []
        for imgs, imgs_for_dep, captions in tqdm(val_loader):
            #captions: [[],[],..,[]]
            ref_caps.extend(captions)
            imgs = imgs.to(config.device)
            imgs_for_dep = imgs_for_dep.to(config.device)

            depth_maps = dpt(imgs_for_dep)
            depth_maps = depth_maps.unsqueeze(1)
            #depth_map = depth_map.detach()
            depth_maps = dpt.standardize_depth_map(depth_maps)
            depth_maps = depth_transforms(depth_maps)

            depth_features = depth_encoder(depth_maps)

            # エンコーダ・デコーダモデルによる予測
            feature = encoder(imgs)
            sampled_ids = decoder.batch_sample(feature, depth_features, word_to_id)
            hypos_id.append(sampled_ids)

        hypos_id = np.concatenate(hypos_id)
        hypos_word = []
        for ids in hypos_id:
            line = []
            for id in ids:
                w = id_to_word[id]
                if w == "<end>":
                    break
                line.append(w)
            hypos_word.append(" ".join(line))

        #print(f"ref: {len(ref_caps)}, hypo: {len(hypos_word)}")
        ref, hypo = load_textfiles(ref_caps,hypos_word)
        score_result = score(ref, hypo)
        #print(ref[10])
        #print(hypo[10])
        print(score_result)
        for mt, sc in score_result.items():
            scores[mt].append(sc)
        
    dire = config.save_directory_Cdep_hard_ori+"/remORI_scores.pkl"
    with open(dire, "wb") as f:
        pickle.dump(scores, f)
       
    del encoder, decoder, depth_encoder, dpt
    gc.collect()

    return

def Cdepth_hard_sample(sample_pic):
    config = ConfigEval()

    # 辞書（単語→単語ID）の読み込み
    with open(config.word_to_id_file, 'rb') as f:
        word_to_id = pickle.load(f)

    # 辞書（単語ID→単語）の読み込み
    with open(config.id_to_word_file, 'rb') as f:
        id_to_word = pickle.load(f)

    # 辞書サイズを保存
    vocab_size = len(id_to_word)
    
    # 画像のtransformsを定義
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        # ImageNetデータセットの平均と標準偏差
        #T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])

    norm_trans = T.Compose([T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    depth_resize = T.Compose([T.Resize((224,224))])
    image_size = 384
    dep_trans = T.Compose([T.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        T.CenterCrop(image_size),
                                        #T.ToTensor(),
                                        T.Normalize(mean=0.5, std=0.5)])
    #validationデータセット
    #ランダムな4000個を使用
    #val_dataset = dataset.CocoCaptions(root=config.val_img_directory, 
    #                                     annFile=config.val_anno_file, 
    #                                     transform=transforms)
    
    #make_refs_lambda = lambda x: util.make_refs_dep(x, word_to_id)
    #indeces = np.load("np_val_index.npy").tolist()
    #coco4k = Subset(val_dataset, indeces)
    #print(f"coco4k : {len(coco4k)}")
    #val_loader = torch.utils.data.DataLoader(
    #                    coco4k, 
    #                    batch_size=config.batch_size, 
    #                    num_workers=config.num_workers, 
    #                    collate_fn=make_refs_lambda)
    
    # モデルの定義
    encoder = CNNEncoder_Atten(config.enc_img_size)
    decoder = CD_RNNDecoderWithHardAttention(config.dim_attention,
                                      config.dim_embedding, 
                                      config.dim_encoder,
                                      config.dim_hidden,
                                      vocab_size, config.device)
    depth_encoder = Depth_CNN_endoder(config.enc_img_size)
    dpt = DPT_Depthestimator()

    encoder.to(config.device)
    decoder.to(config.device)
    depth_encoder.to(config.device)
    dpt.to(config.device)
    dpt.load_weight()

    encoder.eval()
    decoder.eval()
    depth_encoder.eval()
    dpt.eval()

    encoder.load_state_dict(
            torch.load(f'{config.save_directory_Cdep_hard}/depth_hard_encoder_best1.pth'))
    decoder.load_state_dict(
            torch.load(f'{config.save_directory_Cdep_hard}/depth_hard_decoder_best1.pth'))
    depth_encoder.load_state_dict(
            torch.load(f'{config.save_directory_Cdep_hard}/depth_hard_D_encoder_best1.pth'))


    #scores = {"Bleu_1":[], "Bleu_2":[], "Bleu_3":[],"Bleu_4":[], "METEOR":[], "ROUGE_L":[],"CIDEr":[]}
    if sample_pic == "sample1":
        img_directry = config.sample1_dir
    elif sample_pic == "sample2":
        img_directry = config.sample2_dir
    elif sample_pic == "sample3":
        img_directry = config.sample3_dir
    elif sample_pic == "airbus":
        img_directry = config.airbus_dir
    elif sample_pic == "cycling":
        img_directry = config.cycling_dir
    elif sample_pic == "dog":
        img_directry = config.dog_dir
    elif sample_pic == "football":
        img_directry = config.football_dir
    elif sample_pic == "soccer":
        img_directry = config.soccer_dir
    elif sample_pic == "river":
        img_directry = config.river_dir
    elif sample_pic == "seagull":
        img_directry = config.seagull_dir
    elif sample_pic == "bird":
        img_directry = config.bird_dir
    else:
        print("Input correct name")
        return
    
    for img_file in sorted(
        glob.glob(os.path.join(img_directry, '*.[jp][pn]g'))):

        img = Image.open(img_file).convert("RGB")
        #img = np.asarray(img)
        #print(type(img))
        img = transforms(img)#224×224
        img = img.unsqueeze(0)#バッチ化
        img_for_dep = img.detach().clone()
        img = norm_trans(img)
        img_for_dep = dep_trans(img_for_dep)
        img = img.to(config.device)
        img_for_dep = img_for_dep.to(config.device)

        depth_map = dpt(img_for_dep)
        depth_map = depth_map.unsqueeze(1)
        #depth_map = depth_map.detach()
        depth_map = dpt.standardize_depth_map(depth_map)
        depth_map = depth_resize(depth_map)

        feature = encoder(img)
        depth_features = depth_encoder(depth_map)
        sampled_ids, alphas = decoder.sample(feature, depth_features, word_to_id)

        img_plt = Image.open(img_file)
        img_plt = img_plt.resize([224, 224], Image.LANCZOS)
        plt.imshow(img_plt)
        plt.axis('off')
        #plt.show()
        plt.savefig(img_file[:-4] + f'_input.png', bbox_inches='tight')
        plt.clf()
        plt.close()

        print(f'入力画像: {os.path.basename(img_file)}')

        sampled_caption = []
        c = 0
        for word_id, alpha in zip(sampled_ids, alphas):
            word = id_to_word[word_id]
            sampled_caption.append(word)
            #print(alpha)
            alpha = alpha.view(
                config.enc_img_size, config.enc_img_size)
            alpha = alpha.to('cpu').numpy()
            #print(alpha)
            alpha = skimage.transform.pyramid_expand(
                alpha, upscale=16, sigma=8)
            
            # タイムステップtの画像をプロット
            plt.imshow(img_plt)
            #plt.text(0, 1, f'{word}', color='black',
            #         backgroundcolor='white', fontsize=12)
            plt.imshow(alpha, alpha=0.8)
            plt.set_cmap(cm.Greys_r)
            plt.axis('off')
            #plt.show()
            plt.savefig(img_file[:-4] + f'_depth_hard_{word}_p{c}.png', bbox_inches='tight')
            plt.clf()
            plt.close()
            c += 1
            
            if word == '<end>':
                break
        
        sentence = ' '.join(sampled_caption)
        print(f'出力キャプション: {sentence}')

        # 推定結果を書き込み
        gen_sentence_out = img_file[:-4] + '_depth_hard.txt'
        with open(gen_sentence_out, 'w') as f:
            print(sentence, file=f)

    del encoder, decoder, depth_encoder, dpt
    gc.collect()

    return



def main():
    args = sys.argv
    if args[1] == "soft" and args[2] == "cnn":
        if args[3] == "score":
            Cdepth_soft_evaluation()
        elif args[3] == "sample":
            sample_pic = args[4]
            Cdepth_soft_sample(sample_pic)

    elif args[1] == "soft" and args[2] == "mlp":
        if args[3] == "score":
            pass
        elif args[3] == "sample":
            pass

    elif args[1] == "hard" and args[2] == "cnn":
        if args[3] == "score":
            Cdepth_hard_evaluation()
        elif args[3] == "sample":
            sample_pic = args[4]
            Cdepth_hard_sample(sample_pic)
            
    elif args[1] == "hard" and args[2] == "mlp":
        if args[3] == "score":
            pass
        elif args[3] == "sample":
            pass

if __name__ == "__main__":
    main()
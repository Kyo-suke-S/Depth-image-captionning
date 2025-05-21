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
from base_caption_models import CNNEncoder_Atten, RNNDecoderWithSoftAttention, RNNDecoderWithHardAttention
from config import ConfigEval
from evaluate_metrix import load_textfiles, score
import sys
import util

from nic import evaluation_nic

def Soft_evaluation():
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
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])
    #validationデータセット
    #ランダムな4000個を使用
    val_dataset = dataset.CocoCaptions(root=config.val_img_directory,
                                         annFile=config.rem_ori_val_anno_file, 
                                         #annFile=config.remCOCO_ori_val_anno_file, 
                                         transform=transforms)
    
    make_refs_lambda = lambda x: util.make_refs(x, word_to_id)
    #npy_file = config.index_dir
    #indeces = np.load(npy_file).tolist()#ランダムに取り出すためのインデックス
    #print(len(indeces))
    #subcoco = Subset(val_dataset, indeces)#オリジナルデータ500点だけではサブセットを作らない
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
    decoder = RNNDecoderWithSoftAttention(config.dim_attention,
                                      config.dim_embedding, 
                                      config.dim_encoder,
                                      config.dim_hidden,
                                      vocab_size)
    encoder.to(config.device)
    decoder.to(config.device)
    encoder.eval()
    decoder.eval()

    scores = {"Bleu_1":[], "Bleu_2":[], "Bleu_3":[],"Bleu_4":[], "METEOR":[], "ROUGE_L":[],"CIDEr":[]}
    for i  in range(3):
        # モデルの学習済み重みパラメータをロード
        encoder.load_state_dict(
            torch.load(f'{config.save_directory_soft_ori}/base_soft_encoder_best_ori{i}.pth'))
        decoder.load_state_dict(
            torch.load(f'{config.save_directory_soft_ori}/base_soft_decoder_best_ori{i}.pth'))

        #coco4kでキャプションを評価
        ref_caps = []
        hypos_id = []
        for imgs, captions in tqdm(val_loader):
            #captions: [[],[],..,[]]
            ref_caps.extend(captions)
            imgs = imgs.to(config.device)

            # エンコーダ・デコーダモデルによる予測
            feature = encoder(imgs)
            sampled_ids = decoder.batch_sample(feature, word_to_id)
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
        
    dire = config.save_directory_soft_ori+"/remORI_scores.pkl"
    with open(dire, "wb") as f:
        pickle.dump(scores, f)

            # 入力画像を表示
            #img_plt = Image.open(img_file)
            #img_plt = img_plt.resize([224, 224], Image.LANCZOS)
            #plt.imshow(img_plt)
            #plt.axis('off')
            #plt.show()
            #print(f'入力画像: {os.path.basename(img_file)}')

            # 画像キャプショニングの実行
            #sampled_caption = []
            #word = id_to_word[word_id]
            #sampled_caption.append(word)
            
            #alpha = alpha.view(
                #config.enc_img_size, config.enc_img_size)
            #alpha = alpha.to('cpu').numpy()
            #alpha = skimage.transform.pyramid_expand(
                #alpha, upscale=16, sigma=8)
            
            # タイムステップtの画像をプロット
            #plt.imshow(img_plt)
            #plt.text(0, 1, f'{word}', color='black',
            #         backgroundcolor='white', fontsize=12)
            #plt.imshow(alpha, alpha=0.8)
            #plt.set_cmap(cm.Greys_r)
            #plt.axis('off')
            #plt.show()
            
            #if word == '<end>':
                #break
        
        #sentence = ' '.join(sampled_caption)
        #print(f'出力キャプション: {sentence}')

        # 推定結果を書き込み
        #gen_sentence_out = img_file[:-4] + '_show_attend_and_tell(soft).txt'
        #with open(gen_sentence_out, 'w') as f:
            #print(sentence, file=f)

def Soft_sample(sample_pic):#キャプションとアテンションを可視化
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
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])
    
    # モデルの定義
    encoder = CNNEncoder_Atten(config.enc_img_size)
    decoder = RNNDecoderWithSoftAttention(config.dim_attention,
                                      config.dim_embedding, 
                                      config.dim_encoder,
                                      config.dim_hidden,
                                      vocab_size)
    encoder.to(config.device)
    decoder.to(config.device)
    encoder.eval()
    decoder.eval()

    # モデルの学習済み重みパラメータをロード
    encoder.load_state_dict(
        torch.load(f'{config.save_directory_soft}/base_soft_encoder_best1.pth'))
    decoder.load_state_dict(
        torch.load(f'{config.save_directory_soft}/base_soft_decoder_best1.pth'))
    
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


    # ディレクトリ内の画像を対象としてキャプショニング実行
    for img_file in sorted(
        glob.glob(os.path.join(img_directry, '*.[jp][pn]g'))):

        # 画像読み込み
        print(img_file)
        img = Image.open(img_file).convert("RGB")
        img = transforms(img)
        img = img.unsqueeze(0)
        img = img.to(config.device)

        # エンコーダ・デコーダモデルによる予測
        feature = encoder(img)
        sampled_ids, alphas = decoder.sample(feature, word_to_id)

        # 入力画像を表示
        img_plt = Image.open(img_file)
        img_plt = img_plt.resize([224, 224], Image.LANCZOS)
        plt.imshow(img_plt)
        plt.axis('off')
        #plt.show()
        plt.savefig(img_file[:-4] + f'_input.png', bbox_inches='tight')
        plt.clf()
        plt.close()

        print(f'入力画像: {os.path.basename(img_file)}')


        # 画像キャプショニングの実行
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
            plt.savefig(img_file[:-4] + f'_base_soft_{word}_p{c}.png', bbox_inches='tight')
            plt.clf()
            plt.close()
            c += 1
            
            if word == '<end>':
                break
        
        sentence = ' '.join(sampled_caption)
        print(f'出力キャプション: {sentence}')

        # 推定結果を書き込み
        gen_sentence_out = img_file[:-4] + '_base_soft.txt'
        with open(gen_sentence_out, 'w') as f:
            print(sentence, file=f)

#---------------------------------------------------------------------------------
def Hard_evaluation():
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
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])
    #validationデータセット
    #ランダムな4000個を使用
    val_dataset = dataset.CocoCaptions(root=config.val_img_directory, 
                                         annFile=config.rem_ori_val_anno_file,
                                         #annFile=config.remCOCO_ori_val_anno_file, 
                                         transform=transforms)#
    
    make_refs_lambda = lambda x: util.make_refs(x, word_to_id)
    #indeces = np.load(config.index_dir).tolist()
    #subcoco = Subset(val_dataset, indeces)
    #print(f"subcoco : {len(subcoco)}")
    print(f"val_dataset : {len(val_dataset)}")
    val_loader = torch.utils.data.DataLoader(
                        val_dataset,
                        #subcoco, 
                        batch_size=config.batch_size, 
                        num_workers=config.num_workers, 
                        collate_fn=make_refs_lambda)
    
    # モデルの定義
    encoder = CNNEncoder_Atten(config.enc_img_size)
    decoder = RNNDecoderWithHardAttention(config.dim_attention,
                                      config.dim_embedding, 
                                      config.dim_encoder,
                                      config.dim_hidden,
                                      vocab_size,config.device, config.dropout)
    encoder.to(config.device)
    decoder.to(config.device)
    encoder.eval()
    decoder.eval()

    scores = {"Bleu_1":[], "Bleu_2":[], "Bleu_3":[],"Bleu_4":[], "METEOR":[], "ROUGE_L":[],"CIDEr":[]}
    for i  in range(3):
        # モデルの学習済み重みパラメータをロード
        encoder.load_state_dict(
            torch.load(f'{config.save_directory_hard_ori}/base_hard_encoder_best_ori{i}.pth'))
        decoder.load_state_dict(
            torch.load(f'{config.save_directory_hard_ori}/base_hard_decoder_best_ori{i}.pth'))

        #coco4kでキャプションを評価
        ref_caps = []
        hypos_id = []
        for imgs, captions in tqdm(val_loader):
            #captions: [[],[],..,[]]
            ref_caps.extend(captions)
            imgs = imgs.to(config.device)

            # エンコーダ・デコーダモデルによる予測
            feature = encoder(imgs)
            sampled_ids = decoder.batch_sample(feature, word_to_id)
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
        
    dire = config.save_directory_hard_ori+"/remORI_scores.pkl"
    with open(dire, "wb") as f:
        pickle.dump(scores, f)

            # 入力画像を表示
            #img_plt = Image.open(img_file)
            #img_plt = img_plt.resize([224, 224], Image.LANCZOS)
            #plt.imshow(img_plt)
            #plt.axis('off')
            #plt.show()
            #print(f'入力画像: {os.path.basename(img_file)}')

            # 画像キャプショニングの実行
            #sampled_caption = []
            #word = id_to_word[word_id]
            #sampled_caption.append(word)
            
            #alpha = alpha.view(
                #config.enc_img_size, config.enc_img_size)
            #alpha = alpha.to('cpu').numpy()
            #alpha = skimage.transform.pyramid_expand(
                #alpha, upscale=16, sigma=8)
            
            # タイムステップtの画像をプロット
            #plt.imshow(img_plt)
            #plt.text(0, 1, f'{word}', color='black',
            #         backgroundcolor='white', fontsize=12)
            #plt.imshow(alpha, alpha=0.8)
            #plt.set_cmap(cm.Greys_r)
            #plt.axis('off')
            #plt.show()
            
            #if word == '<end>':
                #break
        
        #sentence = ' '.join(sampled_caption)
        #print(f'出力キャプション: {sentence}')

        # 推定結果を書き込み
        #gen_sentence_out = img_file[:-4] + '_show_attend_and_tell(soft).txt'
        #with open(gen_sentence_out, 'w') as f:
            #print(sentence, file=f)


def Hard_smaple(sample_pic):
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
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])
    
    # モデルの定義
    encoder = CNNEncoder_Atten(config.enc_img_size)
    decoder = RNNDecoderWithHardAttention(config.dim_attention,
                                      config.dim_embedding, 
                                      config.dim_encoder,
                                      config.dim_hidden,
                                      vocab_size,
                                      config.device, config.dropout)
    encoder.to(config.device)
    decoder.to(config.device)
    encoder.eval()
    decoder.eval()

    # モデルの学習済み重みパラメータをロード
    encoder.load_state_dict(
        torch.load(f'{config.save_directory_hard}/base_hard_encoder_best1.pth'))
    decoder.load_state_dict(
        torch.load(f'{config.save_directory_hard}/base_hard_decoder_best1.pth'))
    
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

    # ディレクトリ内の画像を対象としてキャプショニング実行
    for img_file in sorted(
        glob.glob(os.path.join(img_directry, '*.[jp][pn]g'))):

        # 画像読み込み
        img = Image.open(img_file)
        img = transforms(img)
        img = img.unsqueeze(0)
        img = img.to(config.device)

        # エンコーダ・デコーダモデルによる予測
        feature = encoder(img)
        sampled_ids, alphas = decoder.sample(feature, word_to_id)

        # 入力画像を表示
        img_plt = Image.open(img_file)
        img_plt = img_plt.resize([224, 224], Image.LANCZOS)
        plt.imshow(img_plt)
        plt.axis('off')
        #plt.show()
        plt.savefig(img_file[:-4] + f'_input.png', bbox_inches='tight')
        plt.clf()
        plt.close()
        print(f'入力画像: {os.path.basename(img_file)}')

        # 画像キャプショニングの実行
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
            plt.savefig(img_file[:-4] + f'_base_hard_{word}_p{c}.png', bbox_inches='tight')
            plt.clf()
            plt.close()
            c += 1
            
            if word == '<end>':
                break
        
        sentence = ' '.join(sampled_caption)
        print(f'出力キャプション: {sentence}')

        # 推定結果を書き込み
        gen_sentence_out = img_file[:-4] + '_base_hard.txt'
        with open(gen_sentence_out, 'w') as f:
            print(sentence, file=f)

def main():
    args = sys.argv
    if args[1] == "soft":
        if args[2] == "score":
            Soft_evaluation()
        elif args[2] == "sample":
            sample_pic = args[3]
            Soft_sample(sample_pic)
    elif args[1] == "hard":
        if args[2] == "score":
            Hard_evaluation()
        elif args[2] == "sample":
            sample_pic = args[3]
            Hard_smaple(sample_pic)
    elif args[1] == "nic":
        evaluation_nic()

if __name__ == "__main__":
    main()


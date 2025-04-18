import sentencepiece as spm
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np

class SentencePieceTokenizer:
    def __init__(self, args):
        self.sp_model:spm.SentencePieceProcessor = spm.SentencePieceProcessor()
        self.special_tokens = {
            "pad": "<pad>",
            "unk": "<unk>",
            "bos": "<s>",
            "eos": "</s>",
        }
        self.special_tokens_int = {
            "pad": 0,
            "unk": 1,
            "bos": 2,
            "eos": 3,
        }
        self.train_tokenizer = args.train_tokenizer
        self.tokenizer_save_model_path = args.tokenizer_save_model_path
        self.tokenizer_train_input_path = args.tokenizer_train_input_path
        self.tokenizer_mode = args.tokenizer_mode #['bpe','unigram','char','word']
        self.tokenizer_vocab_size = args.tokenizer_vocab_size
        self.character_coverage = args.tokenizer_character_coverage
        self.split_digits = args.tokenizer_split_digits
        self.num_threads = args.tokenizer_training_num_threads
        self.user_defined_symbols = args.tokenizer_user_defined_symbols
        self.max_sentence_length = args.tokenizer_max_sentence_length

        self.model_path = args.tokenizer_model_path
        
        if self.model_path:
            self.load(self.model_path)


    def load(self, model_path: str):
        self.sp_model.load(model_path)
        
        # 验证特殊 Token 一致性
        expected_specials = list(self.special_tokens.values())
        actual_specials = [self.sp_model.id_to_piece(i) for i in [0, 1, 2, 3]]
        assert actual_specials == expected_specials, "Special tokens mismatch!"

    def train(self):
        train_args = {
        'input': self.tokenizer_train_input_path,
        'model_prefix': self.tokenizer_save_model_path,
        'vocab_size': self.tokenizer_vocab_size,      
        'model_type': self.tokenizer_mode,            
        'character_coverage': self.character_coverage,     
        'pad_id': 0,                    # 填充符号ID
        'unk_id': 1,                    # 未知符号ID
        'bos_id': 2,                    # 句子开始符号ID
        'eos_id': 3,                    # 句子结束符号ID
        'num_threads': self.num_threads,         
        'split_digits': self.split_digits,         
        'user_defined_symbols': self.user_defined_symbols,
         "max_sentence_length": self.max_sentence_length,
        }
        try:
            spm.SentencePieceTrainer.train(**train_args)
        except Exception as e:
            raise e
        
        self.load(self.tokenizer_save_model_path)


    def encode(
        self,
        text: str|List[str],
        text_target: List[str] = None,
        prepare_decoder_input_ids_from_labels: bool = False,
        add_bos: bool = False,
        add_eos: bool = True,
        max_length: Optional[int] = None,
        padding: Optional[str] = None,
        truncation: bool = False,
        return_tensors: Optional[str]  = None,
    ) -> Dict[str, torch.Tensor]:
        # Encode
        if return_tensors == 'pt':
            if isinstance(text, str):
                text = [text]
            with ThreadPoolExecutor() as executor:
                batch_input_ids = list(executor.map(
                    lambda t: self.sp_model.encode(t, add_bos=add_bos, add_eos=add_eos),
                    text
                ))
            

            # 截断处理
            if truncation and max_length:
                batch_input_ids = [id[:max_length] for id in batch_input_ids]

             # 自动填充逻辑
            if padding == 'max_length' and max_length is not None:
                padded_input_ids = [id + [self.special_tokens_int["pad"]] * (max_length - len(id)) for id in batch_input_ids]
            elif padding == 'longest':
                max_len = max(len(ids) for ids in batch_input_ids)
                padded_input_ids = [
                    ids + [self.special_tokens_int["pad"]] * (max_len - len(ids))
                    for ids in batch_input_ids
                ]

            # 转换为Tensor列表
            padded_ids = torch.tensor(padded_input_ids)

             # 生成注意力掩码
            attention_mask = (padded_ids != self.special_tokens_int["pad"]).type(torch.LongTensor)

            return_dict = {
            "input_ids": padded_ids,
            "attention_mask": attention_mask,
            }

            # label输入的处理
            labels = None
            decoder_input_ids = None
            if text_target is not None:
                with ThreadPoolExecutor() as executor:
                    batch_label_ids = list(executor.map(
                        lambda t: self.sp_model.encode(t, add_bos=False, add_eos=True),
                        text_target
                    ))
                
                # Truncate labels
                if truncation and max_length is not None:
                    batch_label_ids = [ids[:max_length] for ids in batch_label_ids]
                
                # Pad labels
                if padding == 'max_length' and max_length is not None:
                    padded_label_ids = [
                        ids + [self.special_tokens_int["pad"]] * (max_length - len(ids))
                        for ids in batch_label_ids
                    ]
                elif padding == 'longest':
                    max_len = max(len(ids) for ids in batch_label_ids)
                    padded_label_ids = [
                        ids + [self.special_tokens_int["pad"]] * (max_len - len(ids))
                        for ids in batch_label_ids
                    ]
                else:
                    padded_label_ids = batch_label_ids
                
                labels_tensor = torch.tensor(padded_label_ids)
                labels = torch.where(labels_tensor != self.special_tokens_int["pad"], labels_tensor, -100)
                
                # Generate decoder_input_ids from labels if required
                if prepare_decoder_input_ids_from_labels:
                    batch_decoder_input_ids = []
                    for label_ids in batch_label_ids:
                        if len(label_ids) == 0:
                            decoder_ids = [self.special_tokens_int["bos"]]
                        else:
                            decoder_ids = [self.special_tokens_int["bos"]] + label_ids[:-1]
                        batch_decoder_input_ids.append(decoder_ids)
                    
                    # Truncate decoder_input_ids
                    if truncation and max_length is not None:
                        batch_decoder_input_ids = [ids[:max_length] for ids in batch_decoder_input_ids]
                    
                    # Pad decoder_input_ids
                    if padding == 'max_length' and max_length is not None:
                        padded_decoder_ids = [
                            ids + [self.special_tokens_int["pad"]] * (max_length - len(ids))
                            for ids in batch_decoder_input_ids
                        ]
                    elif padding == 'longest':
                        max_len = max(len(ids) for ids in batch_decoder_input_ids)
                        padded_decoder_ids = [
                            ids + [self.special_tokens_int["pad"]] * (max_len - len(ids))
                            for ids in batch_decoder_input_ids
                        ]
                    else:
                        padded_decoder_ids = batch_decoder_input_ids
                    
                    decoder_input_ids = torch.tensor(padded_decoder_ids)

            if labels is not None:
                return_dict["labels"] = labels

            if decoder_input_ids is not None:
                return_dict["decoder_input_ids"] = decoder_input_ids

        if return_tensors == "pt":
            return return_dict
        else:
            raise ValueError(f"Unsupported tensor type: {return_tensors}")
    
    def decode(self, ids: List[int]) -> str:
        # 类型转换处理
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().tolist()  # 确保转移到CPU并转为列表
        elif isinstance(ids, np.ndarray):
            ids = ids.tolist()
        
        # 确保是平面列表（处理单样本情况）
        if isinstance(ids[0], (list, np.ndarray, torch.Tensor)):
            ids = ids[0]  # 自动取第一个样本
        
        return self.sp_model.decode(ids)
    
    def get_vocab_size(self) -> int:
        return self.sp_model.get_piece_size()

    def get_vocab(self) -> Dict[str, int]:
        return {self.sp_model.id_to_piece(i): i for i in range(self.get_vocab_size())}



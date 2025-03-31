import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Full parameters")

    # 训练tokenizer参数
    parser.add_argument("--train_tokenizer", action="store_true", help="是否训练新的 tokenizer")
    parser.add_argument("--tokenizer_save_model_path", type=str, default="tokenizer_model/tokenizer_model", help="保存模型的路径前缀")
    parser.add_argument("--tokenizer_train_input_path", type=str, help="训练 tokenizer 的输入文件路径")
    parser.add_argument("--tokenizer_mode", type=str, choices=["bpe", "unigram", "char", "word"], default="bpe", help="分词模式")
    parser.add_argument("--tokenizer_vocab_size", type=int, default=16000, help="词汇表大小")
    parser.add_argument("--tokenizer_character_coverage", type=float, default=1.0, help="字符覆盖范围")
    parser.add_argument("--tokenizer_split_digits", action="store_true", help="是否将数字拆分为单独 token")
    parser.add_argument("--tokenizer_training_num_threads", type=int, default=12, help="训练线程数")
    parser.add_argument("--tokenizer_user_defined_symbols", type=str, default="(,),:,-", help="用户定义的特殊符号")
    parser.add_argument("--tokenizer_max_sentence_length", type=int, default=1000, help="最大句子长度")

    # 加载tokenizer参数
    parser.add_argument("--tokenizer_model_path", type=str, default="tokenizer_model/tokenizer_model", help="加载的tokenizer模型路径")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
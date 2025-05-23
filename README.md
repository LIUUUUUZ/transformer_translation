# transformer_translation
从头构建Transformer模型进行翻译任务，后续会加上RoPE的对比以及NSA等新技巧的优化

# 主要计划
* 由torch等中基础模块实现类似GPT的Transformer模型的泰译英任务
* 2025.5.4 新增计划:
* 1.使用OpenAi的Tokenizer和预训练GPT2模型的0-shot和few-shot在泰译英任务上的主观感受以及BLEU分数。
* 2.想方法测试预训练Tokenizer和本地训练的BPE Tokenizer在泰译英任务上的编码优劣。
* 3.通过OpenAi提供的GPT2模型框架与预训练权重在其在原始Tokenizer上进行泰译英任务的微调并测试模型的翻译效果。
* 4.通过OpenAi提供的GPT2模型框架在BPE Tokenizer上进行泰译英任务的训练并测试模型的翻译效果，尽量去寻找已经完善的GPT2训练框架。
* 5.探究本地数据集中训练集和测试集的overlap程度。
* 6.将RoPE替换4中训练的GPT2中的PE部分，测试模型的性能变化以及如果可能测试模型的外推性。
* 7.将NSA框架添加进GPT2中进行微调参数，测试NSA框架在输入长度较短的情况下的提升幅度。

# 当前计划
* 构建分词器（tokenizer） √ 
* 构建GPT翻译任务的训练与测试数据集 √
* ~~理解翻译任务的损失函数并构建GPT Decoder模型与训练框架~~
* 加载OpenAi的预训练模型与预训练Tokenizer进行0-shot和few-shot实验，得到其在泰译英任务上的主观感受以及BLEU分数。
* 通过n-gram方法探究本地数据集中训练集和测试集的overlap程度。

# 进度
* 针对泰译英训练数据集(80万对句子组)用BPE训练了词汇表与分词模型，并以此实现此项目中GPT2的tokenizer，输出格式与AutoTokenizer中GPT2的一致，目前只支持输入max_lenght，padding，truncation和输出pt格式。

* 依照huggingface的[Transformer入门手册](https://transformers.run/c3/2022-03-24-transformers-note-7/#1-%E5%87%86%E5%A4%87%E6%95%B0%E6%8D%AE)中的数据格式完成训练与测试数据的dataloader(暂时不支持val数据集)， 且在tokenizer的encode中加入了text_target参数用于同步处理目标翻译句子的编码和prepare_decoder_input_ids_from_labels参数控制生成句子最前面的序列起始符是bos token的目标翻译句子， 其中text_target生成的label句子中填充 token 对应的标签都被自动设置为 -100。

# 参考论文
1.[Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
2.[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

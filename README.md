# transformer_translation
从头构建Transformer模型进行翻译任务，后续会加上RoPE的对比以及NSA等新技巧的优化

# 主要计划
* 由torch等中基础模块实现类似GPT的Transformer模型的泰译英任务

# 当前计划
* 构建分词器（tokenizer） √ 
* 构建GPT翻译任务的训练与测试数据集

# 进度
* 针对泰译英训练数据集(80万对句子组)用BPE训练了词汇表与分词模型，并以此实现此项目中GPT2的tokenizer，输出格式与AutoTokenizer中GPT2的一致，目前只支持输入max_lenght，padding，truncation和输出pt格式。
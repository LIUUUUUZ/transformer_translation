from torch.utils.data import DataLoader

def collote_fn(batch_samples, SPtokenizer, padding_mode, max_input_length):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample['translation']["th"])
        batch_targets.append(sample['translation']["en"])
    batch_data = SPtokenizer.encode(
    batch_inputs, 
    text_target=batch_targets,
    prepare_decoder_input_ids_from_labels= True,
    padding= padding_mode, 
    max_length=max_input_length, 
    truncation=True,
    return_tensors="pt" 
    )
    return batch_data

def get_dataloader(dataset_en_th):
    train_dataloader = DataLoader(dataset_en_th["train"], batch_size=32, shuffle=True, collate_fn=collote_fn)
    test_dataloader = DataLoader(dataset_en_th["test"], batch_size=32, shuffle=False, collate_fn=collote_fn)
    return train_dataloader, test_dataloader
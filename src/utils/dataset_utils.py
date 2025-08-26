from types import SimpleNamespace
from typing import Any, Tuple

from torch.utils.data import DataLoader

from src.dataset import LAIONPOPTextImageDataset
from src.flow.dataset import KaggleImageFolderImagenet, ImageNet21kFolder


def create_datasets_and_loaders(config: SimpleNamespace, accelerator) -> Tuple[Any, Any, DataLoader, DataLoader]:
    """Create dataset, val_dataset and corresponding data loaders based on config and accelerator.

    Returns (dataset, val_dataset, dataloader, val_loader).
    """
    dataset_choice = getattr(config, 'dataset', 'laion_pop')
    if str(dataset_choice).lower() == 'imagenet64_kaggle':
        H, W = tuple(getattr(config, 'input_size', (256, 256)))
        res = int(H)
        dataset = KaggleImageFolderImagenet(
            split='train',
            resolution=res,
            kaggle_dataset_id=getattr(config, 'kaggle_dataset_id', 'ayaroshevskiy/downsampled-imagenet-64x64'),
            max_samples=getattr(config, 'max_samples', None)
        )
        val_dataset = KaggleImageFolderImagenet(
            split='val', resolution=res,
            kaggle_dataset_id=getattr(config, 'kaggle_dataset_id', 'ayaroshevskiy/downsampled-imagenet-64x64')
        )
    elif str(dataset_choice).lower() == 'imagenet21k_folder':
        root = getattr(config, 'imagenet21k_root', None)
        if not root:
            raise ValueError("--imagenet21k_root must be provided for imagenet21k_folder dataset")
        H, W = tuple(getattr(config, 'input_size', (256, 256)))
        res = int(H)
        dataset = ImageNet21kFolder(root_dir=root, split='train', resolution=res, max_samples=getattr(config, 'max_samples', None))
        val_dataset = ImageNet21kFolder(root_dir=root, split='val', resolution=res)
    else:
        dataset = LAIONPOPTextImageDataset(
            vocab_size=getattr(config, 'vocab_size', 32000),
            tokenizer_path=getattr(config, 'tokenizer_path', "gs://t5-data/vocabs/cc_en.32000/sentencepiece.model"),
            max_text_len=getattr(config, 'max_seq_len', 64),
            image_size=tuple(getattr(config, 'input_size', (256, 256))),
            cache_dir=getattr(config, 'cache_dir', "./laion_pop_cache"),
            max_samples=getattr(config, 'max_samples', None),
            use_cogvlm_captions=getattr(config, 'use_cogvlm_captions', True),
            min_resolution=getattr(config, 'min_resolution', 512),
            num_workers=getattr(config, 'num_workers', 4),
            ignore_pad=getattr(config, 'ignore_pad', False)
        )
        # No explicit val set; reuse train dataset for a quick sanity val (not ideal)
        val_dataset = dataset

    train_sampler, val_sampler = accelerator.build_samplers(dataset, val_dataset)
    pin_mem = True if accelerator.device.type == 'cuda' else False

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(getattr(config, 'num_workers', 8) or 8),
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=True,
        pin_memory=pin_mem
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=int(getattr(config, 'num_workers', 8) or 8),
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=False,
        pin_memory=pin_mem
    )

    return dataset, val_dataset, dataloader, val_loader



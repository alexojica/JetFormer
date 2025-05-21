import sys
import platform
import os
import pathlib # For robust path normalization
from typing import Optional # Added for type hinting

if platform.system() == "Windows":
    class MockResource:
        RLIMIT_NOFILE = 7 # RLIMIT_NOFILE is usually an int, value doesn't strictly matter for this mock
        
        def getrlimit(self, resource_type):
            if resource_type == self.RLIMIT_NOFILE:
                # Return some plausible default values for open file limits (soft, hard)
                return (1024, 4096) 
            raise ValueError(f"MockResource: Unsupported resource type: {resource_type}")

        def setrlimit(self, resource_type, limits):
            # This mock won't actually change system limits on Windows
            # It just needs to exist to prevent an AttributeError
            print(f"MockResource: setrlimit for {resource_type} called with {limits}, but not implemented on Windows.")
            pass

    sys.modules['resource'] = MockResource()

import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

class TFDSImagenet64(Dataset):
    """PyTorch Dataset wrapper for TFDS downsampled_imagenet/64x64."""
    def __init__(self, split='train', data_dir=None, manual_tar_dir=None, max_samples: Optional[int] = None):
        """
        Args:
            split (str): Dataset split, e.g., 'train', 'validation'.
            data_dir (str, optional): Root directory for TFDS to store *prepared* data.
                                     Defaults to TFDS's default path if None.
            manual_tar_dir (str, optional): Absolute or relative path to a directory 
                                          containing the manually downloaded .tar files 
                                          (e.g., train_64x64.tar, valid_64x64.tar).
                                          If None, TFDS might attempt standard download procedures.
            max_samples (int, optional): Maximum number of samples to load from the split. 
                                         If None, all samples are loaded.
        """
        super().__init__()
        # Ensure TensorFlow does not allocate GPU memory if PyTorch is also using it.
        tf.config.set_visible_devices([], 'GPU')

        if not manual_tar_dir:
            raise ValueError("manual_tar_dir must be specified and point to the directory with .tar files.")

        # Ensure manual_tar_dir is an absolute, normalized path
        resolved_manual_tar_dir = str(pathlib.Path(manual_tar_dir).expanduser().resolve())

        # data_dir for the builder is where the *prepared* dataset will live or be looked for.
        # If None, TFDS uses its default (e.g., ~/tensorflow_datasets).
        resolved_builder_data_dir = None
        if data_dir:
            resolved_builder_data_dir = str(pathlib.Path(data_dir).expanduser().resolve())
        
        self.builder = tfds.builder(
            'downsampled_imagenet/64x64',
            data_dir=resolved_builder_data_dir 
        )
        
        # Explicitly configure manual_dir for download_and_prepare
        # This tells TFDS where to find the .tar archives.
        download_config = tfds.download.DownloadConfig(
            manual_dir=resolved_manual_tar_dir,
        )
        
        print(f"TFDSImagenet64: Using builder data_dir (for prepared data): {resolved_builder_data_dir if resolved_builder_data_dir else 'TFDS Default'}")
        print(f"TFDSImagenet64: Using manual_tar_dir for DownloadConfig (for .tar files): {resolved_manual_tar_dir}")
        
        # This will use the .tar files from resolved_manual_tar_dir
        # and prepare the dataset into resolved_builder_data_dir.
        self.builder.download_and_prepare(download_config=download_config)
        
        self.tf_dataset = self.builder.as_dataset(split=split, as_supervised=False)
        
        if max_samples is not None and max_samples > 0:
            self.tf_dataset = self.tf_dataset.take(max_samples)
            print(f"Loading up to {max_samples} samples from {split} split of downsampled_imagenet/64x64 into memory...")
        else:
            print(f"Loading all samples from {split} split of downsampled_imagenet/64x64 into memory...")
            
        self.examples = []
        for example in tfds.as_numpy(self.tf_dataset):
            self.examples.append(example)
        print(f"Loaded {len(self.examples)} examples.")

        # Clean up TensorFlow objects to prevent pickling issues with DataLoader workers
        del self.tf_dataset
        del self.builder

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_uint8 = example['image'] # (64, 64, 3) uint8 [0, 255]
        label = example.get('label', -1)       # scalar int64, defaults to -1 if not present

        # Preprocessing: value_range(-1, 1)
        image_float32 = image_uint8.astype(np.float32)
        image_normalized = (image_float32 / 127.5) - 1.0 # Converts [0, 255] to [-1, 1]
        
        image_tensor = torch.from_numpy(image_normalized)
        
        if label != -1:
            label_tensor = torch.tensor(label, dtype=torch.long)
            return {"image": image_tensor, "label": label_tensor}
        else:
            # If no label (or label was defaulted to -1), return only image
            # This aligns with downsampled_imagenet/64x64 typically only having 'image' feature
            # and big_vision's config doing 'keep("image")'
            return {"image": image_tensor}


if __name__ == '__main__':
    print("Testing TFDSImagenet64 Dataset...")
    
    # IMPORTANT: Create a folder (e.g., './my_manual_tars') and place your
    # train_64x64.tar and valid_64x64.tar (or equivalent) files there.
    test_manual_tar_dir = "./local_imagenet64_tars/" # <--- SET THIS TO YOUR TAR DIRECTORY
    
    # This is where TFDS will store the PROCESSED dataset (TFRecords, etc.)
    # It can be the TFDS default, or a custom path.
    # Example: test_prepared_data_dir = "C:/Users/aojic/tensorflow_datasets_prepared"
    test_prepared_data_dir = None # Let TFDS use its default for prepared data for this test
    test_max_samples = 100 # Test loading a small subset

    if not os.path.exists(test_manual_tar_dir) or not any(f.endswith('.tar') for f in os.listdir(test_manual_tar_dir)):
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR: test_manual_tar_dir '{test_manual_tar_dir}' does not exist or contains no .tar files.")
        print(f"Please create it and place your downsampled_imagenet 64x64 .tar files inside.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit(1)
        
    try:
        print(f"Attempting to load with manual_tar_dir: {test_manual_tar_dir}")
        print(f"TFDS will prepare data into: {test_prepared_data_dir if test_prepared_data_dir else 'TFDS default location'}")
        
        train_dataset = TFDSImagenet64(
            split='train[:1%]', 
            data_dir=test_prepared_data_dir, 
            manual_tar_dir=test_manual_tar_dir,
            max_samples=test_max_samples # Pass max_samples for testing
        ) 
        print(f"Number of training samples (requested up to {test_max_samples}, from 1% split): {len(train_dataset)}")
        
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            image_sample = sample['image']
            print(f"Image sample shape: {image_sample.shape}, dtype: {image_sample.dtype}")
            print(f"Image sample min: {image_sample.min()}, max: {image_sample.max()}")

            if 'label' in sample:
                label_sample = sample['label']
                print(f"Label sample: {label_sample}, dtype: {label_sample.dtype}")
            else:
                print("Label not found in sample (as might be expected for this dataset).")

            # Test DataLoader
            # Note: collate_fn might need adjustment if labels are sometimes missing
            def collate_fn_maybe_label(batch):
                images = torch.stack([item['image'] for item in batch])
                if 'label' in batch[0]:
                    labels = torch.stack([item['label'] for item in batch])
                    return {'image': images, 'label': labels}
                return {'image': images}

            train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_maybe_label)
            batch_sample = next(iter(train_dataloader))
            images_batch = batch_sample['image']
            print(f"Images batch shape: {images_batch.shape}")
            if 'label' in batch_sample:
                labels_batch = batch_sample['label']
                print(f"Labels batch shape: {labels_batch.shape}")
            else:
                print("Labels not in batch.")
        else:
            print("Dataset is empty, cannot test sample.")

    except Exception as e:
        print(f"An error occurred during dataset testing: {e}")
        print("Please ensure tensorflow-datasets is installed and configured correctly.")
        print("For downsampled_imagenet/64x64, manual download of .tar files into")
        print("the 'manual_dir' (e.g., YOUR_TFDS_DATA_DIR/downloads/manual/) is required.")
        import traceback
        traceback.print_exc() 
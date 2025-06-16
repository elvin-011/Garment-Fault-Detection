import os
from pathlib import Path
import shutil
import json
import logging
import numpy as np
import cv2
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import concurrent.futures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreparator:
    def __init__(self, raw_data_dir='data/raw', processed_data_dir='data/processed'):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.dataset_paths = {
            'TILDA': self.raw_data_dir / 'tilda',
            'TFD': self.raw_data_dir / 'tfd',
            'AITEX': self.raw_data_dir / 'aitex',
            'DAGM': self.raw_data_dir / 'DAGM 2007',
            'FabricSpotDefect': self.raw_data_dir / 'fabricspotdefect',
            'DeepFashion2': self.raw_data_dir / 'deepfashion2',
            'FabricDefect': self.raw_data_dir / 'fabric_defect',
            'IndoFashion': self.raw_data_dir / 'indo_fashion'
        }
        
        # Create processed data directory structure
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'val', 'test']:
            (self.processed_data_dir / split).mkdir(exist_ok=True)
            (self.processed_data_dir / split / 'images').mkdir(exist_ok=True)
            (self.processed_data_dir / split / 'labels').mkdir(exist_ok=True)
    
    def _safe_read_json(self, file_path):
        """Safely read a JSON file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {file_path}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return None
    
    def _safe_write_json(self, file_path, data):
        """Safely write data to a JSON file with error handling"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error writing to {file_path}: {str(e)}")
            return False
    
    def _safe_read_image(self, file_path):
        """Safely read an image file with error handling"""
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                logger.error(f"Failed to read image: {file_path}")
                return None
            return img
        except Exception as e:
            logger.error(f"Error reading image {file_path}: {str(e)}")
            return None
    
    def _safe_write_image(self, file_path, img):
        """Safely write an image file with error handling"""
        try:
            success = cv2.imwrite(str(file_path), img)
            if not success:
                logger.error(f"Failed to write image: {file_path}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error writing image {file_path}: {str(e)}")
            return False
    
    def _safe_read_text(self, file_path):
        """Safely read a text file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {str(e)}")
            return None
    
    def prepare_surface_defect_datasets(self):
        """Prepare surface defect datasets (TILDA, TFD, AITEX, DAGM)"""
        logger.info("Preparing surface defect datasets...")
        
        # Prepare TILDA dataset
        self._prepare_tilda()
        
        # Prepare TFD dataset
        self._prepare_tfd()
        
        # Prepare AITEX dataset
        self._prepare_aitex()
        
        # Prepare DAGM dataset
        self._prepare_dagm()
    
    def prepare_anomaly_datasets(self):
        """Prepare anomaly detection datasets (FabricDefect, FabricSpotDefect)"""
        logger.info("Preparing anomaly detection datasets...")
        
        # Prepare FabricDefect dataset
        self._prepare_fabric_defect()
        
        # Prepare FabricSpotDefect dataset
        self._prepare_fabric_spot_defect()
    
    def prepare_structure_datasets(self):
        """Prepare structural analysis datasets (IndoFashion, DeepFashion2)"""
        logger.info("Preparing structural analysis datasets...")
        
        # Prepare IndoFashion dataset
        self._prepare_indofashion()
        
        # Prepare DeepFashion2 dataset
        self._prepare_deepfashion2()
    
    def _prepare_tilda(self):
        """Prepare TILDA dataset"""
        logger.info("Preparing TILDA dataset...")
        tilda_path = self.dataset_paths['TILDA']
        if not tilda_path.exists():
            logger.warning(f"TILDA dataset not found at {tilda_path}")
            return
        
        # Create class mapping
        class_mapping = {
            'good': 0,
            'hole': 1,
            'objects': 2,
            'oil spot': 3,
            'thread error': 4
        }
        
        # Process each class
        for class_name, class_id in class_mapping.items():
            class_path = tilda_path / class_name
            if not class_path.exists():
                continue
            
            # Get all images for this class
            images = list(class_path.glob('*.png'))
            
            # Split into train/val/test
            train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
            
            # Process each split
            for split, imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
                for img_path in tqdm(imgs, desc=f"Processing TILDA {class_name} {split}"):
                    # Copy and resize image
                    img = cv2.imread(str(img_path))
                    img = cv2.resize(img, (64, 64))
                    
                    # Save image
                    save_path = self.processed_data_dir / split / 'images' / f"tilda_{class_name}_{img_path.stem}.png"
                    cv2.imwrite(str(save_path), img)
                    
                    # Create label
                    label = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'dataset': 'TILDA',
                        'original_path': str(img_path)
                    }
                    
                    # Save label
                    label_path = self.processed_data_dir / split / 'labels' / f"tilda_{class_name}_{img_path.stem}.json"
                    with open(label_path, 'w') as f:
                        json.dump(label, f)
    
    def _prepare_tfd(self):
        """Prepare TFD dataset"""
        logger.info("Preparing TFD dataset...")
        tfd_path = self.dataset_paths['TFD']
        if not tfd_path.exists():
            logger.warning(f"TFD dataset not found at {tfd_path}")
            return
        
        # Get all images
        images_path = tfd_path / 'Images'
        if not images_path.exists():
            return
        
        images = list(images_path.glob('*.png'))
        
        # Split into train/val/test
        train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
        
        # Process each split
        for split, imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            for img_path in tqdm(imgs, desc=f"Processing TFD {split}"):
                # Copy and resize image
                img = cv2.imread(str(img_path))
                img = cv2.resize(img, (256, 256))
                
                # Save image
                save_path = self.processed_data_dir / split / 'images' / f"tfd_{img_path.stem}.png"
                cv2.imwrite(str(save_path), img)
                
                # Get corresponding mask
                mask_path = tfd_path / 'masks' / f"{img_path.stem}.txt"
                if mask_path.exists():
                    # Read mask
                    with open(mask_path, 'r') as f:
                        mask_data = f.read().strip().split()
                    
                    # Create label
                    label = {
                        'dataset': 'TFD',
                        'original_path': str(img_path),
                        'mask_path': str(mask_path),
                        'grid_size': [int(mask_data[0]), int(mask_data[1])] if len(mask_data) >= 2 else None
                    }
                    
                    # Save label
                    label_path = self.processed_data_dir / split / 'labels' / f"tfd_{img_path.stem}.json"
                    with open(label_path, 'w') as f:
                        json.dump(label, f)
    
    def _prepare_aitex(self):
        """Prepare AITEX dataset"""
        logger.info("Preparing AITEX dataset...")
        aitex_path = self.dataset_paths['AITEX']
        if not aitex_path.exists():
            logger.warning(f"AITEX dataset not found at {aitex_path}")
            return
        
        # Get all images and masks
        images_path = aitex_path / 'images'
        masks_path = aitex_path / 'masks'
        
        if not (images_path.exists() and masks_path.exists()):
            return
        
        images = list(images_path.glob('*.png'))
        
        # Split into train/val/test
        train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
        
        # Process each split
        for split, imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            for img_path in tqdm(imgs, desc=f"Processing AITEX {split}"):
                # Get corresponding mask
                mask_path = masks_path / f"{img_path.stem}_mask.png"
                if not mask_path.exists():
                    continue
                
                # Read and resize image
                img = cv2.imread(str(img_path))
                img = cv2.resize(img, (256, 4096))
                
                # Read and resize mask
                mask = cv2.imread(str(mask_path))
                mask = cv2.resize(mask, (256, 4096))
                
                # Save image
                save_path = self.processed_data_dir / split / 'images' / f"aitex_{img_path.stem}.png"
                cv2.imwrite(str(save_path), img)
                
                # Save mask
                mask_save_path = self.processed_data_dir / split / 'labels' / f"aitex_{img_path.stem}_mask.png"
                cv2.imwrite(str(mask_save_path), mask)
                
                # Create label
                label = {
                    'dataset': 'AITEX',
                    'original_path': str(img_path),
                    'mask_path': str(mask_path)
                }
                
                # Save label
                label_path = self.processed_data_dir / split / 'labels' / f"aitex_{img_path.stem}.json"
                with open(label_path, 'w') as f:
                    json.dump(label, f)
    
    def _prepare_dagm(self):
        """Prepare DAGM dataset"""
        logger.info("Preparing DAGM dataset...")
        dagm_path = self.dataset_paths['DAGM']
        if not dagm_path.exists():
            logger.warning(f"DAGM dataset not found at {dagm_path}")
            return
        
        # Process each class
        for class_num in range(1, 11):
            class_name = f'Class{class_num}'
            class_path = dagm_path / class_name
            
            if not class_path.exists():
                continue
            
            # Process train and test sets
            for split in ['Train', 'Test']:
                split_path = class_path / split
                if not split_path.exists():
                    continue
                
                # Get all images
                images = list(split_path.glob('*.PNG'))
                
                # Process each image
                for img_path in tqdm(images, desc=f"Processing DAGM {class_name} {split}"):
                    # Get corresponding label
                    label_path = split_path / 'Label' / f"{img_path.stem}_label.PNG"
                    if not label_path.exists():
                        continue
                    
                    # Read and resize image
                    img = cv2.imread(str(img_path))
                    img = cv2.resize(img, (512, 512))
                    
                    # Read and resize label
                    label_img = cv2.imread(str(label_path))
                    label_img = cv2.resize(label_img, (512, 512))
                    
                    # Determine target split
                    target_split = 'train' if split == 'Train' else 'test'
                    
                    # Save image
                    save_path = self.processed_data_dir / target_split / 'images' / f"dagm_{class_name}_{img_path.stem}.png"
                    cv2.imwrite(str(save_path), img)
                    
                    # Save label
                    label_save_path = self.processed_data_dir / target_split / 'labels' / f"dagm_{class_name}_{img_path.stem}_label.png"
                    cv2.imwrite(str(label_save_path), label_img)
                    
                    # Create label
                    label = {
                        'dataset': 'DAGM',
                        'class_name': class_name,
                        'original_path': str(img_path),
                        'label_path': str(label_path)
                    }
                    
                    # Save label
                    label_path = self.processed_data_dir / target_split / 'labels' / f"dagm_{class_name}_{img_path.stem}.json"
                    with open(label_path, 'w') as f:
                        json.dump(label, f)
    
    def _prepare_fabric_defect(self):
        """Prepare FabricDefect dataset"""
        logger.info("Preparing FabricDefect dataset...")
        fabric_defect_path = self.dataset_paths['FabricDefect']
        if not fabric_defect_path.exists():
            logger.warning(f"FabricDefect dataset not found at {fabric_defect_path}")
            return
        
        # Process H5 file
        h5_path = fabric_defect_path / 'matchingtDATASET_train_64.h5'
        if not h5_path.exists():
            logger.warning(f"H5 file not found at {h5_path}")
            return
        
        try:
            with h5py.File(h5_path, 'r') as f:
                # Get all classes and angles
                classes = list(f.keys())
                if not classes:
                    logger.warning("No classes found in H5 file")
                    return
                
                angles = list(f[classes[0]].keys())
                if not angles:
                    logger.warning("No angles found in H5 file")
                    return
                
                # Process each class
                for class_name in classes:
                    try:
                        # Get all samples for this class
                        samples = []
                        for angle in angles:
                            try:
                                dataset = f[class_name][angle]
                                for i in range(dataset.shape[0]):
                                    samples.append((angle, i, dataset[i]))
                            except Exception as e:
                                logger.error(f"Error processing angle {angle} for class {class_name}: {str(e)}")
                                continue
                        
                        if not samples:
                            logger.warning(f"No samples found for class {class_name}")
                            continue
                        
                        # Split into train/val/test
                        train_samples, temp_samples = train_test_split(samples, test_size=0.3, random_state=42)
                        val_samples, test_samples = train_test_split(temp_samples, test_size=0.5, random_state=42)
                        
                        # Process each split
                        for split, split_samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
                            for angle, idx, sample in tqdm(split_samples, desc=f"Processing FabricDefect {class_name} {split}"):
                                try:
                                    # Convert sample to image
                                    img = (sample[0] * 255).astype(np.uint8)
                                    
                                    # Save image
                                    save_path = self.processed_data_dir / split / 'images' / f"fabric_defect_{class_name}_{angle}_{idx}.png"
                                    if not self._safe_write_image(save_path, img):
                                        continue
                                    
                                    # Create label
                                    label = {
                                        'dataset': 'FabricDefect',
                                        'class_name': class_name,
                                        'angle': angle,
                                        'sample_index': idx
                                    }
                                    
                                    # Save label
                                    label_path = self.processed_data_dir / split / 'labels' / f"fabric_defect_{class_name}_{angle}_{idx}.json"
                                    self._safe_write_json(label_path, label)
                                except Exception as e:
                                    logger.error(f"Error processing sample {idx} for class {class_name} angle {angle}: {str(e)}")
                                    continue
                    except Exception as e:
                        logger.error(f"Error processing class {class_name}: {str(e)}")
                        continue
        except Exception as e:
            logger.error(f"Error reading H5 file {h5_path}: {str(e)}")
            return
    
    def _prepare_fabric_spot_defect(self):
        """Prepare FabricSpotDefect dataset"""
        logger.info("Preparing FabricSpotDefect dataset...")
        fabric_spot_path = self.dataset_paths['FabricSpotDefect']
        if not fabric_spot_path.exists():
            logger.warning(f"FabricSpotDefect dataset not found at {fabric_spot_path}")
            return
        
        # Process original images
        original_path = fabric_spot_path / 'Orginal'
        if original_path.exists():
            # Process each split
            for split in ['train', 'val', 'test']:
                split_path = original_path / split
                if not split_path.exists():
                    continue
                
                # Get all images
                images = list(split_path.glob('*.jpg'))
                
                # Process each image
                for img_path in tqdm(images, desc=f"Processing FabricSpotDefect original {split}"):
                    # Read and resize image
                    img = self._safe_read_image(img_path)
                    if img is None:
                        continue
                    
                    img = cv2.resize(img, (416, 416))
                    
                    # Save image
                    save_path = self.processed_data_dir / split / 'images' / f"fabric_spot_original_{img_path.stem}.jpg"
                    if not self._safe_write_image(save_path, img):
                        continue
                    
                    # Get COCO annotation
                    coco_path = split_path / '_annotations.coco.json'
                    if coco_path.exists():
                        coco_data = self._safe_read_json(coco_path)
                        if coco_data is None:
                            continue
                        
                        # Find annotations for this image
                        image_id = None
                        for img_data in coco_data.get('images', []):
                            if img_data.get('file_name') == img_path.name:
                                image_id = img_data.get('id')
                                break
                        
                        if image_id is not None:
                            # Get annotations for this image
                            annotations = [ann for ann in coco_data.get('annotations', []) 
                                         if ann.get('image_id') == image_id]
                            
                            # Create label
                            label = {
                                'dataset': 'FabricSpotDefect',
                                'original_path': str(img_path),
                                'annotations': annotations
                            }
                            
                            # Save label
                            label_path = self.processed_data_dir / split / 'labels' / f"fabric_spot_original_{img_path.stem}.json"
                            self._safe_write_json(label_path, label)
        
        # Process augmented images
        augmented_path = fabric_spot_path / 'Augmented' / 'YOLOv8'
        if augmented_path.exists():
            # Process each split
            for split in ['train', 'val', 'test']:
                images_path = augmented_path / split / 'images'
                labels_path = augmented_path / split / 'labels'
                
                if not (images_path.exists() and labels_path.exists()):
                    continue
                
                # Get all images
                images = list(images_path.glob('*.jpg'))
                
                # Process each image
                for img_path in tqdm(images, desc=f"Processing FabricSpotDefect augmented {split}"):
                    # Read and resize image
                    img = cv2.imread(str(img_path))
                    img = cv2.resize(img, (416, 416))
                    
                    # Save image
                    save_path = self.processed_data_dir / split / 'images' / f"fabric_spot_augmented_{img_path.stem}.jpg"
                    cv2.imwrite(str(save_path), img)
                    
                    # Get YOLO label
                    label_path = labels_path / f"{img_path.stem}.txt"
                    if label_path.exists():
                        # Read YOLO label
                        with open(label_path, 'r') as f:
                            yolo_labels = f.readlines()
                        
                        # Create label
                        label = {
                            'dataset': 'FabricSpotDefect',
                            'original_path': str(img_path),
                            'yolo_labels': [line.strip() for line in yolo_labels]
                        }
                        
                        # Save label
                        label_save_path = self.processed_data_dir / split / 'labels' / f"fabric_spot_augmented_{img_path.stem}.json"
                        with open(label_save_path, 'w') as f:
                            json.dump(label, f)
    
    def _prepare_indofashion(self):
        """Efficiently prepare IndoFashion dataset using split-specific metadata and class mapping."""
        logger.info("Preparing IndoFashion dataset (efficient)...")
        indofashion_path = self.dataset_paths['IndoFashion']
        if not indofashion_path.exists():
            logger.warning(f"IndoFashion dataset not found at {indofashion_path}")
            return

        images_path = indofashion_path / 'images'
        metadata_files = {
            'train': indofashion_path / 'train_data.json',
            'val': indofashion_path / 'val_data.json',
            'test': indofashion_path / 'test_data.json'
        }

        split_metadata = {}
        class_mapping = {}
        current_class_idx = 0

        # Load metadata and build class mapping (JSONL support)
        for split, json_path in metadata_files.items():
            if json_path.exists():
                try:
                    data = []
                    with open(json_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                data.append(json.loads(line))
                    split_metadata[split] = data
                    for item in data:
                        class_label = item.get('class_label', '')
                        if class_label not in class_mapping:
                            class_mapping[class_label] = current_class_idx
                            current_class_idx += 1
                except Exception as e:
                    logger.error(f"Error loading {json_path}: {str(e)}")
                    continue

        # Save class mapping
        class_mapping_path = self.processed_data_dir / 'indo_fashion_classes.json'
        with open(class_mapping_path, 'w', encoding='utf-8') as f:
            json.dump({'class_mapping': class_mapping, 'classes': list(class_mapping.keys())}, f, indent=2)

        def process_item(item, split):
            try:
                img_rel_path = item['image_path']
                img_path = indofashion_path / img_rel_path
                if not img_path.exists():
                    logger.warning(f"Image not found: {img_path}")
                    return
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"Error reading image: {img_path}")
                    return
                # Resize maintaining aspect ratio
                h, w = img.shape[:2]
                max_size = 512
                if h > w:
                    new_h = max_size
                    new_w = int(w * (max_size / h))
                else:
                    new_w = max_size
                    new_h = int(h * (max_size / w))
                img = cv2.resize(img, (new_w, new_h))
                # Save image
                save_path = self.processed_data_dir / split / 'images' / f"indofashion_{Path(img_rel_path).stem}.jpg"
                cv2.imwrite(str(save_path), img)
                # Save label
                class_label = item.get('class_label', '')
                class_idx = class_mapping.get(class_label, 0)
                label = {
                    'dataset': 'IndoFashion',
                    'original_path': str(img_path),
                    'brand': item.get('brand', ''),
                    'product_title': item.get('product_title', ''),
                    'class_label': class_label,
                    'class_idx': class_idx
                }
                label_path = self.processed_data_dir / split / 'labels' / f"indofashion_{Path(img_rel_path).stem}.json"
                with open(label_path, 'w', encoding='utf-8') as f:
                    json.dump(label, f)
            except Exception as e:
                logger.error(f"Error processing IndoFashion item: {str(e)}")

        # Process each split in parallel
        for split, metadata in split_metadata.items():
            logger.info(f"Processing IndoFashion {split} split with {len(metadata)} items...")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                list(tqdm(executor.map(lambda item: process_item(item, split), metadata), total=len(metadata), desc=f"IndoFashion {split}"))
    
    def _prepare_deepfashion2(self):
        """Prepare DeepFashion2 dataset"""
        logger.info("Preparing DeepFashion2 dataset...")
        deepfashion2_path = self.dataset_paths['DeepFashion2']
        if not deepfashion2_path.exists():
            logger.warning(f"DeepFashion2 dataset not found at {deepfashion2_path}")
            return
        
        # Process each split
        for split in ['train', 'val', 'test']:
            split_path = deepfashion2_path / ('validation' if split == 'val' else split)
            if not split_path.exists():
                continue
            
            # Get all images
            images_path = split_path / 'image'
            if not images_path.exists():
                continue
            
            images = list(images_path.glob('*.jpg'))
            
            # Process each image
            for img_path in tqdm(images, desc=f"Processing DeepFashion2 {split}"):
                # Read and resize image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Resize maintaining aspect ratio
                h, w = img.shape[:2]
                max_size = 512
                if h > w:
                    new_h = max_size
                    new_w = int(w * (max_size / h))
                else:
                    new_w = max_size
                    new_h = int(h * (max_size / w))
                
                img = cv2.resize(img, (new_w, new_h))
                
                # Save image
                save_path = self.processed_data_dir / split / 'images' / f"deepfashion2_{img_path.stem}.jpg"
                cv2.imwrite(str(save_path), img)
                
                # Get annotation
                if split != 'test':  # Test set doesn't have annotations
                    annos_path = split_path / 'annos'
                    if annos_path.exists():
                        ann_path = annos_path / f"{img_path.stem}.json"
                        if ann_path.exists():
                            with open(ann_path, 'r') as f:
                                ann = json.load(f)
                            
                            # Create label
                            label = {
                                'dataset': 'DeepFashion2',
                                'original_path': str(img_path),
                                'annotation': ann
                            }
                            
                            # Save label
                            label_path = self.processed_data_dir / split / 'labels' / f"deepfashion2_{img_path.stem}.json"
                            with open(label_path, 'w') as f:
                                json.dump(label, f)

    def prepare_deepfashion2_split(self, split='val'):
        logger.info(f"Preparing DeepFashion2 {split} split...")
        deepfashion2_path = self.dataset_paths['DeepFashion2']
        if not deepfashion2_path.exists():
            logger.warning(f"DeepFashion2 dataset not found at {deepfashion2_path}")
            return

        split_path = deepfashion2_path / ('validation' if split == 'val' else split)
        if not split_path.exists():
            logger.warning(f"Split path not found: {split_path}")
            return

        images_path = split_path / 'image'
        if not images_path.exists():
            logger.warning(f"Images path not found: {images_path}")
            return

        images = list(images_path.glob('*.jpg'))
        for img_path in tqdm(images, desc=f"Processing DeepFashion2 {split}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            max_size = 512
            if h > w:
                new_h = max_size
                new_w = int(w * (max_size / h))
            else:
                new_w = max_size
                new_h = int(h * (max_size / w))
            img = cv2.resize(img, (new_w, new_h))
            save_path = self.processed_data_dir / split / 'images' / f"deepfashion2_{img_path.stem}.jpg"
            cv2.imwrite(str(save_path), img)
            if split != 'test':
                annos_path = split_path / 'annos'
                if annos_path.exists():
                    ann_path = annos_path / f"{img_path.stem}.json"
                    if ann_path.exists():
                        with open(ann_path, 'r') as f:
                            ann = json.load(f)
                        label = {
                            'dataset': 'DeepFashion2',
                            'original_path': str(img_path),
                            'annotation': ann
                        }
                        label_path = self.processed_data_dir / split / 'labels' / f"deepfashion2_{img_path.stem}.json"
                        with open(label_path, 'w') as f:
                            json.dump(label, f)

def main():
    try:
        preparator = DataPreparator()
        
        # Prepare all datasets
        # preparator.prepare_surface_defect_datasets()
        # preparator.prepare_anomaly_datasets()
        # preparator.prepare_structure_datasets()
        # preparator.prepare_deepfashion2_split('val')
        preparator.prepare_deepfashion2_split('test')
        logger.info("Data preparation completed successfully!")
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise  # Re-raise the exception for debugging

if __name__ == "__main__":
    main() 
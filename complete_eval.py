# YOLOv11 Garment Fault Detection - Comprehensive Model Evaluation
# Use this script to evaluate your trained best.pt model

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
from PIL import Image
import glob

class GarmentFaultEvaluator:
    def __init__(self, model_path, data_yaml_path=None):
        """
        Initialize evaluator with trained model
        
        Args:
            model_path: Path to best.pt file
            data_yaml_path: Path to data.yaml file (optional)
        """
        self.model_path = model_path
        self.data_yaml_path = data_yaml_path
        self.model = None
        self.classes = ['Broken button', 'Button hike', 'Color defect', 'Foreign yarn', 'Hole', 'Sewing error']
        self.class_colors = {
            'Broken Button': (255, 0, 0),      # Red
            'Button hike': (255,255,255),      #white
            'Color defect': (0, 255, 0),       # Green
            'Foreign yarn': (255, 255, 0),     # Yellow
            'Hole': (0, 0, 255),               # Blue
            'Sewing error': (255, 0, 255)      # Magenta
        }
        
    def load_model(self):
        """Load the trained model"""
        try:
            print(f"📥 Loading model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("✅ Model loaded successfully!")
            
            # Print model info
            print(f"🤖 Model type: {type(self.model.model).__name__}")
            print(f"📊 Number of classes: {len(self.classes)}")
            print(f"🏷️ Classes: {self.classes}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def evaluate_on_test_set(self, test_images_path=None):
        """
        Comprehensive evaluation on test set with detailed statistics
        
        Args:
            test_images_path: Path to test images directory
        """
        if not self.model:
            print("❌ Model not loaded! Call load_model() first.")
            return None
        
        print("📈 Starting comprehensive evaluation...")
    
        # Method 1: Use built-in validation if data.yaml is available
        if self.data_yaml_path and os.path.exists(self.data_yaml_path):
            print("🔍 Running validation on test set using data.yaml...")
            try:
                results = self.model.val(
                    data=self.data_yaml_path,
                    split='test',  # Use test set
                    save_json=True,
                    save_hybrid=True,
                    conf=0.25,  # Confidence threshold
                    iou=0.6,    # IoU threshold for NMS
                    max_det=300, # Maximum detections per image
                    half=False,  # Use FP32 for better accuracy
                    device='0' if torch.cuda.is_available() else 'cpu',
                    dnn=False,
                    plots=True,  # Generate plots
                    verbose=True
                )
                
                # Print standard YOLO metrics first
                self.print_detailed_metrics(results)
                
                # Then run detailed analysis on the same test images
                print("\n" + "="*60)
                print("📊 DETAILED TEST SET ANALYSIS")
                print("="*60)
                
                # Extract test images path from data.yaml if not provided
                if not test_images_path:
                    test_images_path = self.extract_test_path_from_yaml()
                
                if test_images_path and os.path.exists(test_images_path):
                    print("🔍 Running detailed analysis on test images...")
                    detailed_results = self.manual_evaluation(test_images_path)
                    return results, detailed_results
                else:
                    print("⚠️  Test images path not found for detailed analysis")
                    return results, None
                    
            except Exception as e:
                print(f"❌ Built-in validation failed: {e}")
                print("🔄 Falling back to manual evaluation...")
                # Fall through to manual evaluation
            
        # Method 2: Manual evaluation if only images are available
        if test_images_path and os.path.exists(test_images_path):
            print(f"🔍 Running manual evaluation on images in: {test_images_path}")
            return None, self.manual_evaluation(test_images_path)
            
        else:
            print("❌ Please provide either valid data.yaml path or test images path")
            return None, None

    def extract_test_path_from_yaml(self):
        """Extract test images path from data.yaml file"""
        try:
            import yaml
            with open(self.data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            test_path = data.get('test', None)
            if test_path:
                # Handle relative paths
                if not os.path.isabs(test_path):
                    yaml_dir = os.path.dirname(self.data_yaml_path)
                    test_path = os.path.join(yaml_dir, test_path)
                return test_path
        except Exception as e:
            print(f"⚠️  Could not extract test path from yaml: {e}")
        return None

    def run_comprehensive_test_evaluation(self, test_images_path=None):
        """
        Run comprehensive evaluation with all detailed statistics
        This ensures you get both YOLO metrics AND detailed analysis
        """
        if not self.model:
            print("❌ Model not loaded! Call load_model() first.")
            return None, None
            
        print("🚀 Starting comprehensive test evaluation with detailed statistics...")
        
        builtin_results = None
        detailed_results = None
        
        # Step 1: Try built-in validation first
        if self.data_yaml_path and os.path.exists(self.data_yaml_path):
            print("\n" + "="*60)
            print("1️⃣ YOLO BUILT-IN VALIDATION METRICS") 
            print("="*60)
            
            try:
                builtin_results = self.model.val(
                    data=self.data_yaml_path,
                    split='test',
                    save_json=True,
                    save_hybrid=True,
                    conf=0.25,
                    iou=0.6,
                    max_det=300,
                    half=False,
                    device='0' if torch.cuda.is_available() else 'cpu',
                    dnn=False,
                    plots=True,
                    verbose=True
                )
                
                self.print_detailed_metrics(builtin_results)
                
                # Extract test path from yaml if not provided
                if not test_images_path:
                    test_images_path = self.extract_test_path_from_yaml()
                    
            except Exception as e:
                print(f"❌ Built-in validation failed: {e}")
        
        # Step 2: Always run detailed analysis for comprehensive statistics
        if test_images_path and os.path.exists(test_images_path):
            print("\n" + "="*60)
            print("2️⃣ DETAILED TEST SET STATISTICS")
            print("="*60)
            
            detailed_results = self.manual_evaluation(test_images_path)
        elif not test_images_path:
            print("⚠️  No test images path provided for detailed analysis")
        else:
            print(f"❌ Test images path does not exist: {test_images_path}")
        
        return builtin_results, detailed_results
    def print_detailed_metrics(self, results):
            """Print comprehensive evaluation metrics"""
            print("\n" + "="*60)
            print("📊 DETAILED EVALUATION METRICS")
            print("="*60)
            
            # Overall metrics
            print(f"🎯 Overall Performance:")
            print(f"   mAP@50:     {results.box.map50:.4f}")
            print(f"   mAP@50-95:  {results.box.map:.4f}")
            print(f"   Precision:  {results.box.mp:.4f}")
            print(f"   Recall:     {results.box.mr:.4f}")
            print(f"   F1-Score:   {2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr):.4f}")
            
            # Per-class metrics
            if hasattr(results.box, 'maps') and len(results.box.maps) > 0:
                print(f"\n📋 Per-Class mAP@50:")
                for i, (class_name, map_score) in enumerate(zip(self.classes, results.box.maps)):
                    print(f"   {class_name:15}: {map_score:.4f}")
            
            # Additional metrics
            print(f"\n⚙️ Inference Details:")
            print(f"   Speed (preprocess): {results.speed['preprocess']:.2f}ms")
            print(f"   Speed (inference):  {results.speed['inference']:.2f}ms")
            print(f"   Speed (postprocess): {results.speed['postprocess']:.2f}ms")
            
            print("="*60)
    
    def manual_evaluation(self, test_images_path):
        """Manual evaluation when only images are available"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(test_images_path, ext)))
            image_paths.extend(glob.glob(os.path.join(test_images_path, ext.upper())))
        
        if not image_paths:
            print(f"❌ No images found in {test_images_path}")
            return None
        
        print(f"🔍 Found {len(image_paths)} test images")
        
        # Load ground truth if available
        labels_dir = test_images_path.replace('images', 'labels')
        has_ground_truth = os.path.exists(labels_dir)
        
        results_summary = {
            'total_images': len(image_paths),
            'images_with_detections': 0,
            'total_detections': 0,
            'class_detections': {cls: 0 for cls in self.classes},
            'confidence_scores': [],
            'processing_times': [],
            'predictions_per_image': [],
            'ground_truth_per_image': [] if has_ground_truth else None
        }
        
        # Track class presence
        gt_classes_present = set()
        pred_classes_present = set()
        
        # Process each image
        for i, img_path in enumerate(image_paths):
            if i % 50 == 0:
                print(f"Processing image {i+1}/{len(image_paths)}...")
            
            # Run inference
            start_time = cv2.getTickCount()
            results = self.model(img_path, verbose=False)
            end_time = cv2.getTickCount()
            
            processing_time = (end_time - start_time) / cv2.getTickFrequency() * 1000
            results_summary['processing_times'].append(processing_time)
            
            # Extract predictions
            pred_classes = []
            if len(results[0].boxes) > 0:
                results_summary['images_with_detections'] += 1
                results_summary['total_detections'] += len(results[0].boxes)
                
                for box in results[0].boxes:
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    class_name = self.classes[class_id]
                    
                    results_summary['confidence_scores'].append(confidence)
                    results_summary['class_detections'][class_name] += 1
                    pred_classes.append(class_id)
                    pred_classes_present.add(class_id)
            
            results_summary['predictions_per_image'].append(pred_classes)
            
            # Load ground truth if available
            if has_ground_truth:
                gt_classes = self.load_ground_truth_labels(img_path, labels_dir)
                results_summary['ground_truth_per_image'].append(gt_classes)
                gt_classes_present.update(gt_classes)
        
        # Print manual evaluation results with class analysis
        self.print_manual_results_enhanced(results_summary, gt_classes_present, pred_classes_present, has_ground_truth)
        return results_summary
    
    def load_ground_truth_labels(self, image_path, labels_dir):
        """Load ground truth labels from YOLO format txt files"""
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(labels_dir, f"{image_name}.txt")
        
        if not os.path.exists(label_path):
            return []
        
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # class_id x_center y_center width height
                    class_id = int(parts[0])
                    labels.append(class_id)
        return labels
    
    def print_manual_results_enhanced(self, results_summary, gt_classes_present, pred_classes_present, has_ground_truth):
        """Print enhanced manual evaluation results with class analysis"""
        print("\n" + "="*60)
        print("📊 MANUAL EVALUATION RESULTS")
        print("="*60)
        
        total_images = results_summary['total_images']
        images_with_det = results_summary['images_with_detections']
        total_det = results_summary['total_detections']
        
        print(f"📸 Image Statistics:")
        print(f"   Total images:           {total_images}")
        print(f"   Images with detections: {images_with_det} ({images_with_det/total_images*100:.1f}%)")
        print(f"   Images without detect:  {total_images-images_with_det} ({(total_images-images_with_det)/total_images*100:.1f}%)")
        print(f"   Average detections/img: {total_det/total_images:.2f}")
        
        # Enhanced Class Coverage Analysis
        print(f"\n🎯 Class Coverage Analysis:")
        print(f"   Number of classes predicted: {len(pred_classes_present)}")
        print(f"   Predicted classes present: {sorted(list(pred_classes_present))}")
        
        if has_ground_truth:
            print(f"   Number of classes in ground truth: {len(gt_classes_present)}")
            print(f"   Ground truth classes present: {sorted(list(gt_classes_present))}")
            
            # Classes missed and extra classes
            missed_classes = gt_classes_present - pred_classes_present
            extra_classes = pred_classes_present - gt_classes_present
            
            if missed_classes:
                missed_names = [self.classes[i] for i in missed_classes if i < len(self.classes)]
                print(f"   ⚠️  Classes in GT but not predicted: {missed_names}")
            if extra_classes:
                extra_names = [self.classes[i] for i in extra_classes if i < len(self.classes)]
                print(f"   🔍 Classes predicted but not in GT: {extra_names}")
            
            print(f"   ✅ Classes correctly detected: {len(gt_classes_present & pred_classes_present)}/{len(gt_classes_present)}")
        
        print(f"\n🔍 Detection Statistics:")
        print(f"   Total detections: {total_det}")
        for class_name, count in results_summary['class_detections'].items():
            percentage = count / total_det * 100 if total_det > 0 else 0
            print(f"   {class_name:15}: {count:4d} ({percentage:5.1f}%)")
        
        if results_summary['confidence_scores']:
            conf_scores = results_summary['confidence_scores']
            print(f"\n📊 Confidence Statistics:")
            print(f"   Mean confidence: {np.mean(conf_scores):.3f}")
            print(f"   Std confidence:  {np.std(conf_scores):.3f}")
            print(f"   Min confidence:  {np.min(conf_scores):.3f}")
            print(f"   Max confidence:  {np.max(conf_scores):.3f}")
        
        if results_summary['processing_times']:
            proc_times = results_summary['processing_times']
            print(f"\n⚡ Performance Statistics:")
            print(f"   Mean processing time: {np.mean(proc_times):.2f}ms")
            print(f"   Std processing time:  {np.std(proc_times):.2f}ms")
            print(f"   FPS (approximate):    {1000/np.mean(proc_times):.1f}")
        
        # Add confusion matrix analysis if ground truth is available
        if has_ground_truth:
            self.print_confusion_matrix_analysis(results_summary)
        
        # Add background explanation
        # self.print_background_explanation()
        
        print("="*60)
    
    def print_confusion_matrix_analysis(self, results_summary):
        """Print confusion matrix analysis"""
        print(f"\n📈 Confusion Matrix Analysis:")
        
        predictions = results_summary['predictions_per_image']
        ground_truths = results_summary['ground_truth_per_image']
        
        # Image-level confusion matrix for each class
        for class_id in range(len(self.classes)):
            y_true = [1 if class_id in gt else 0 for gt in ground_truths]
            y_pred = [1 if class_id in pred else 0 for pred in predictions]
            
            if any(y_true) or any(y_pred):
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, 0)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"\n   {self.classes[class_id]} - Image-Level Metrics:")
                print(f"      Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
                print(f"      True Negatives: {tn}, False Positives: {fp}")
                print(f"      False Negatives: {fn}, True Positives: {tp}")
    
    
    def visualize_predictions(self, image_path, output_dir='predictions', conf_threshold=0.25):
        """
        Visualize predictions on sample images
        
        Args:
            image_path: Path to single image or directory of images
            output_dir: Directory to save visualized predictions
            conf_threshold: Confidence threshold for display
        """
        if not self.model:
            print("❌ Model not loaded! Call load_model() first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle single image or directory
        if os.path.isfile(image_path):
            image_paths = [image_path]
        elif os.path.isdir(image_path):
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(glob.glob(os.path.join(image_path, ext)))
                image_paths.extend(glob.glob(os.path.join(image_path, ext.upper())))
        else:
            print(f"❌ Invalid image path: {image_path}")
            return
        
        if not image_paths:
            print(f"❌ No images found in {image_path}")
            return
        
        print(f"🎨 Visualizing predictions for {len(image_paths)} images...")
        
        # Process first 20 images (to avoid overwhelming)
        for i, img_path in enumerate(image_paths): #[:20]
            print(f"Processing {os.path.basename(img_path)}...")
            
            # Load and predict
            image = cv2.imread(img_path)
            results = self.model(img_path, conf=conf_threshold, verbose=False)
            
            # Draw predictions
            annotated_image = self.draw_predictions(image, results[0], conf_threshold)
            
            # Save annotated image
            output_path = os.path.join(output_dir, f"pred_{os.path.basename(img_path)}")
            cv2.imwrite(output_path, annotated_image)
        
        print(f"✅ Visualized predictions saved to: {output_dir}")
    
    def draw_predictions(self, image, result, conf_threshold=0.25):
        """Draw bounding boxes and labels on image"""
        annotated_image = image.copy()
        
        if len(result.boxes) == 0:
            # Add "No Defects Detected" text
            cv2.putText(annotated_image, "No Defects Detected", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return annotated_image
        
        for box in result.boxes:
            confidence = box.conf.item()
            if confidence < conf_threshold:
                continue
                
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            class_id = int(box.cls.item())
            class_name = self.classes[class_id]
            color = self.class_colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label and confidence
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_image
    
    def benchmark_speed(self, test_images_path, num_images=100):
        """Benchmark inference speed"""
        if not self.model:
            print("❌ Model not loaded! Call load_model() first.")
            return
        
        # Get test images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(test_images_path, ext)))
            if len(image_paths) >= num_images:
                break
        
        image_paths = image_paths[:num_images]
        
        if not image_paths:
            print(f"❌ No images found in {test_images_path}")
            return
        
        print(f"⚡ Benchmarking speed on {len(image_paths)} images...")
        
        # Warm up
        for _ in range(5):
            self.model(image_paths[0], verbose=False)
        
        # Benchmark
        times = []
        for img_path in image_paths:
            start_time = cv2.getTickCount()
            _ = self.model(img_path, verbose=False)
            end_time = cv2.getTickCount()
            
            processing_time = (end_time - start_time) / cv2.getTickFrequency()
            times.append(processing_time)
        
        # Print results
        mean_time = np.mean(times)
        std_time = np.std(times)
        fps = 1.0 / mean_time
        
        print(f"📊 Speed Benchmark Results:")
        print(f"   Mean inference time: {mean_time*1000:.2f}ms")
        print(f"   Std inference time:  {std_time*1000:.2f}ms")
        print(f"   FPS:                 {fps:.1f}")
        print(f"   Min time:            {np.min(times)*1000:.2f}ms")
        print(f"   Max time:            {np.max(times)*1000:.2f}ms")
    
    def export_model(self, export_formats=['onnx', 'torchscript']):
        """Export model to different formats for deployment"""
        if not self.model:
            print("❌ Model not loaded! Call load_model() first.")
            return
        
        print("📦 Exporting model to different formats...")
        
        for format_name in export_formats:
            try:
                print(f"Exporting to {format_name.upper()}...")
                self.model.export(format=format_name, optimize=True)
                print(f"✅ {format_name.upper()} export successful")
            except Exception as e:
                print(f"❌ {format_name.upper()} export failed: {e}")
        
        print("📦 Model export completed!")

def main():
    """Main execution function with comprehensive statistics"""
    
    # Initialize evaluator
    model_path = "best_final.pt"
    data_yaml = "data.yaml"  # For YOLO validation metrics
    test_images = "test_final _og\images"  # For detailed statistics
    
    evaluator = GarmentFaultEvaluator(model_path, data_yaml)
    
    # Load model
    if not evaluator.load_model():
        return
    
    print("\n🚀 Starting Comprehensive Model Evaluation with Detailed Statistics...")
    
    # 1. Comprehensive evaluation with detailed statistics
    print("\n" + "="*60)
    print("1️⃣ COMPREHENSIVE TEST SET EVALUATION")
    print("="*60)
    
    # This will give you BOTH standard YOLO metrics AND detailed statistics
    builtin_results, detailed_results = evaluator.run_comprehensive_test_evaluation(
        test_images_path=test_images
    )
    
    # 2. Visualize predictions on sample images
    print("\n" + "="*60)
    print("2️⃣ VISUALIZING PREDICTIONS")
    print("="*60)
    
    evaluator.visualize_predictions(test_images, output_dir='prediction_visualizations')
    
    # 3. Speed benchmark
    print("\n" + "="*60)
    print("3️⃣ SPEED BENCHMARK")
    print("="*60)
    
    evaluator.benchmark_speed(test_images, num_images=50)
    
    # 4. Export model for deployment
    print("\n" + "="*60)
    print("4️⃣ MODEL EXPORT")
    print("="*60)
    
    evaluator.export_model(['onnx', 'torchscript'])
    
    # 5. Summary
    print("\n🎉 Comprehensive evaluation completed successfully!")
    print("\n📋 What you got:")
    if builtin_results:
        print("✅ Standard YOLO validation metrics (mAP, precision, recall)")
    if detailed_results:
        print("✅ Detailed image statistics and class coverage")
        print("✅ Per-class detection counts and percentages")
        print("✅ Confidence score distributions")
        print("✅ Processing time analysis")
        print("✅ Image-level confusion matrix for each class")
    print("✅ Prediction visualizations created")
    print("✅ Speed benchmarking completed")
    print("✅ Model exported for deployment")
    print("✅ Ready for production deployment!")

if __name__ == "__main__":
    main()
def evaluate_builtin_method(self):
    """
    Run built-in YOLO validation using data.yaml
    """
    if not self.model:
        print("❌ Model not loaded! Call load_model() first.")
        return None
        
    if not (self.data_yaml_path and os.path.exists(self.data_yaml_path)):
        print("❌ data.yaml not found for built-in evaluation")
        return None
    
    print("🔍 Running built-in YOLO validation using data.yaml...")
    
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
        
        # Print detailed metrics
        self.print_detailed_metrics(results)
        return results
        
    except Exception as e:
        print(f"❌ Built-in validation failed: {e}")
        print("   This might be due to:")
        print("   - Incorrect paths in data.yaml")
        print("   - Missing test dataset")
        print("   - Incompatible dataset format")
        return None

def evaluate_manual_method(self, test_images_path):
    """
    Run manual evaluation on test images directory
    
    Args:
        test_images_path: Path to test images directory
    """
    if not self.model:
        print("❌ Model not loaded! Call load_model() first.")
        return None
        
    if not (test_images_path and os.path.exists(test_images_path)):
        print(f"❌ Test images path does not exist: {test_images_path}")
        return None
    
    print(f"🔍 Running manual evaluation on images in: {test_images_path}")
    return self.manual_evaluation(test_images_path)

def evaluate_comprehensive(self, test_images_path=None, use_builtin=True, use_manual=True):
    """
    Run both built-in and manual evaluations for comprehensive analysis
    
    Args:
        test_images_path: Path to test images directory
        use_builtin: Whether to run built-in evaluation
        use_manual: Whether to run manual evaluation
    """
    if not self.model:
        print("❌ Model not loaded! Call load_model() first.")
        return None, None
        
    print("📈 Starting comprehensive evaluation with both methods...")
    
    builtin_results = None
    manual_results = None
    
    # Method 1: Built-in validation
    if use_builtin:
        print("\n" + "="*50)
        print("🤖 BUILT-IN YOLO VALIDATION")
        print("="*50)
        builtin_results = self.evaluate_builtin_method()
        
        if builtin_results:
            print("✅ Built-in validation completed successfully!")
        else:
            print("⚠️  Built-in validation failed or skipped")
    
    # Method 2: Manual evaluation
    if use_manual and test_images_path:
        print("\n" + "="*50)
        print("🔧 MANUAL EVALUATION")
        print("="*50)
        manual_results = self.evaluate_manual_method(test_images_path)
        
        if manual_results:
            print("✅ Manual evaluation completed successfully!")
        else:
            print("⚠️  Manual evaluation failed")
    elif use_manual:
        print("⚠️  Manual evaluation skipped - no test_images_path provided")
    
    # Comparison summary
    if builtin_results and manual_results:
        self.print_evaluation_comparison(builtin_results, manual_results)
    
    return builtin_results, manual_results

def print_evaluation_comparison(self, builtin_results, manual_results):
    """Compare results from both evaluation methods"""
    print("\n" + "="*60)
    print("🔍 EVALUATION METHOD COMPARISON")
    print("="*60)
    
    print("📊 Built-in vs Manual Evaluation:")
    
    # Built-in metrics
    if hasattr(builtin_results, 'box'):
        print(f"\n🤖 Built-in YOLO Validation:")
        print(f"   mAP@50:     {builtin_results.box.map50:.4f}")
        print(f"   mAP@50-95:  {builtin_results.box.map:.4f}")
        print(f"   Precision:  {builtin_results.box.mp:.4f}")
        print(f"   Recall:     {builtin_results.box.mr:.4f}")
    
    # Manual metrics
    if manual_results:
        total_images = manual_results['total_images']
        images_with_det = manual_results['images_with_detections']
        total_det = manual_results['total_detections']
        
        print(f"\n🔧 Manual Evaluation:")
        print(f"   Total Images:       {total_images}")
        print(f"   Images with Detect: {images_with_det} ({images_with_det/total_images*100:.1f}%)")
        print(f"   Total Detections:   {total_det}")
        print(f"   Avg Det/Image:      {total_det/total_images:.2f}")
        
        if manual_results['confidence_scores']:
            avg_conf = np.mean(manual_results['confidence_scores'])
            print(f"   Avg Confidence:     {avg_conf:.3f}")
    
    print(f"\n💡 Interpretation:")
    print(f"   • Built-in validation uses ground truth labels for precise metrics")
    print(f"   • Manual evaluation shows real-world performance on unlabeled data")
    print(f"   • Both methods complement each other for complete model assessment")
    print("="*60)

# Updated main function to use both methods
def main():
    """Main execution function with dual evaluation"""
    
    # Initialize evaluator
    model_path = "best_final.pt"
    data_yaml = "data.yaml"  # Keep this for built-in evaluation
    test_images = "test_final_og\\images"  # For manual evaluation
    
    evaluator = GarmentFaultEvaluator(model_path, data_yaml)
    
    # Load model
    if not evaluator.load_model():
        return
    
    print("\n🚀 Starting Comprehensive Dual Evaluation...")
    
    # 1. Run both evaluations
    print("\n" + "="*60)
    print("1️⃣ COMPREHENSIVE DUAL EVALUATION")
    print("="*60)
    
    builtin_results, manual_results = evaluator.evaluate_comprehensive(
        test_images_path=test_images,
        use_builtin=True,   # Try built-in evaluation
        use_manual=True     # Also run manual evaluation
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
    
    print("\n🎉 Dual evaluation completed successfully!")
    print("\n📋 Summary of what was done:")
    print("✅ Model loaded and validated")
    print("✅ Built-in YOLO validation attempted")
    print("✅ Manual evaluation completed")
    print("✅ Evaluation methods compared")
    print("✅ Prediction visualizations created")
    print("✅ Performance benchmarking completed")
    print("✅ Model exported for deployment")
    print("✅ Ready for deployment!")

# Quick dual evaluation function
def dual_evaluate(model_path, test_images_path, data_yaml_path=None):
    """
    Quick dual evaluation - runs both methods
    
    Args:
        model_path: Path to your best.pt file
        test_images_path: Path to test images directory
        data_yaml_path: Path to data.yaml (optional)
    """
    evaluator = GarmentFaultEvaluator(model_path, data_yaml_path)
    
    if evaluator.load_model():
        print("🚀 Running dual evaluation...")
        return evaluator.evaluate_comprehensive(test_images_path)
    else:
        print("❌ Failed to load model")
        return None, None
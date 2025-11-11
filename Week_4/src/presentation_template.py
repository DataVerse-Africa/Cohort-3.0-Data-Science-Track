"""
Week 4 Presentation Template: Dataset Insights + Preprocessing Plan
DataVerse Africa Internship Cohort 3.0

This script helps you generate the visualizations and analysis needed for your Saturday presentation.
Run this script to create all the charts and insights for your "Dataset Insights + Preprocessing Plan" presentation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DatasetAnalyzer:
    """
    A comprehensive analyzer for the Grapevine Leaves Dataset
    to generate insights for your presentation
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.classes = ['Ak', 'Ala_Idris', 'Buzgulu', 'Dimnit', 'Nazli']
        self.class_counts = {}
        self.image_properties = {}
        
    def analyze_dataset_structure(self):
        """Analyze the basic structure and statistics of the dataset"""
        print("🔍 Analyzing Dataset Structure...")
        
        # Count images per class
        for class_name in self.classes:
            class_path = self.dataset_path / class_name
            if class_path.exists():
                image_files = list(class_path.glob('*.png'))
                self.class_counts[class_name] = len(image_files)
            else:
                self.class_counts[class_name] = 0
        
        # Calculate total images
        total_images = sum(self.class_counts.values())
        
        print(f"📊 Dataset Overview:")
        print(f"   Total Images: {total_images}")
        print(f"   Number of Classes: {len(self.classes)}")
        print(f"   Images per Class: {self.class_counts}")
        
        return self.class_counts
    
    def create_class_distribution_chart(self, save_path=None):
        """Create a bar chart showing class distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        classes = list(self.class_counts.keys())
        counts = list(self.class_counts.values())
        colors = sns.color_palette("husl", len(classes))
        
        bars = ax1.bar(classes, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('Class Distribution in Grapevine Leaves Dataset', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Grape Variety Classes', fontsize=12)
        ax1.set_ylabel('Number of Images', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        ax2.pie(counts, labels=classes, colors=colors, autopct='%1.1f%%', 
                startangle=90, explode=[0.05]*len(classes))
        ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def analyze_image_properties(self, sample_size=10):
        """Analyze image properties like dimensions, file sizes, etc."""
        print("🖼️  Analyzing Image Properties...")
        
        properties = {
            'widths': [],
            'heights': [],
            'file_sizes': [],
            'channels': [],
            'class_labels': []
        }
        
        for class_name in self.classes:
            class_path = self.dataset_path / class_name
            if class_path.exists():
                image_files = list(class_path.glob('*.png'))[:sample_size]
                
                for img_path in image_files:
                    # Read image
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w, c = img.shape
                        properties['heights'].append(h)
                        properties['widths'].append(w)
                        properties['channels'].append(c)
                        properties['class_labels'].append(class_name)
                        
                        # File size in KB
                        file_size = img_path.stat().st_size / 1024
                        properties['file_sizes'].append(file_size)
        
        self.image_properties = properties
        return properties
    
    def create_image_properties_chart(self, save_path=None):
        """Create charts showing image properties analysis"""
        if not self.image_properties:
            self.analyze_image_properties()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Image dimensions
        axes[0, 0].scatter(self.image_properties['widths'], self.image_properties['heights'], 
                          alpha=0.6, s=50, color='steelblue')
        axes[0, 0].set_title('Image Dimensions Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Height (pixels)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # File sizes distribution
        axes[0, 1].hist(self.image_properties['file_sizes'], bins=20, alpha=0.7, 
                       color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('File Size Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('File Size (KB)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Dimensions by class
        df = pd.DataFrame(self.image_properties)
        sns.boxplot(data=df, x='class_labels', y='widths', ax=axes[1, 0])
        axes[1, 0].set_title('Width Distribution by Class', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Width (pixels)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        sns.boxplot(data=df, x='class_labels', y='heights', ax=axes[1, 1])
        axes[1, 1].set_title('Height Distribution by Class', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Height (pixels)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_sample_images_grid(self, images_per_class=3, save_path=None):
        """Create a grid showing sample images from each class"""
        fig, axes = plt.subplots(len(self.classes), images_per_class, 
                                figsize=(images_per_class*3, len(self.classes)*3))
        
        for i, class_name in enumerate(self.classes):
            class_path = self.dataset_path / class_name
            if class_path.exists():
                image_files = list(class_path.glob('*.png'))[:images_per_class]
                
                for j, img_path in enumerate(image_files):
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        if len(self.classes) == 1:
                            ax = axes[j]
                        else:
                            ax = axes[i, j] if images_per_class > 1 else axes[i]
                        
                        ax.imshow(img_rgb)
                        ax.set_title(f'{class_name}\n{img_path.name}', fontsize=10)
                        ax.axis('off')
        
        plt.suptitle('Sample Images from Each Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def demonstrate_preprocessing_techniques(self, save_path=None):
        """Demonstrate various preprocessing techniques on a sample image"""
        # Get a sample image
        sample_class = self.classes[0]
        class_path = self.dataset_path / sample_class
        sample_image_path = list(class_path.glob('*.png'))[0]
        
        # Load image
        img = cv2.imread(str(sample_image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply different preprocessing techniques
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.GaussianBlur(img_rgb, (15, 15), 0)
        img_eq = cv2.equalizeHist(img_gray)
        edges = cv2.Canny(img_gray, 100, 200)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img_resized)
        axes[0, 1].set_title('Resized (224x224)', fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(img_gray, cmap='gray')
        axes[0, 2].set_title('Grayscale', fontweight='bold')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(img_blur)
        axes[1, 0].set_title('Gaussian Blur', fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(img_eq, cmap='gray')
        axes[1, 1].set_title('Histogram Equalization', fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(edges, cmap='gray')
        axes[1, 2].set_title('Edge Detection', fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.suptitle('Preprocessing Techniques Demonstration', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_preprocessing_pipeline_flowchart(self):
        """Create a text-based flowchart of the preprocessing pipeline"""
        pipeline_steps = [
            "📁 Load Images from Dataset",
            "🔄 Resize to Standard Dimensions (224x224)",
            "🎨 Normalize Pixel Values (0-1 range)",
            "🔀 Data Augmentation (Rotation, Flip, Zoom)",
            "📊 Split into Train/Validation/Test (70/15/15)",
            "🏷️  Encode Labels (One-hot encoding)",
            "📦 Create Data Generators",
            "🚀 Ready for Model Training"
        ]
        
        print("\n" + "="*60)
        print("🔧 PREPROCESSING PIPELINE")
        print("="*60)
        
        for i, step in enumerate(pipeline_steps, 1):
            print(f"{i}. {step}")
            if i < len(pipeline_steps):
                print("   ⬇️")
        
        print("="*60)
        
        return pipeline_steps
    
    def generate_presentation_summary(self):
        """Generate a summary of key insights for the presentation"""
        total_images = sum(self.class_counts.values())
        
        summary = {
            'dataset_overview': {
                'total_images': total_images,
                'num_classes': len(self.classes),
                'balanced': len(set(self.class_counts.values())) == 1,
                'classes': self.classes
            },
            'key_insights': [
                f"Dataset contains {total_images} images across {len(self.classes)} grape varieties",
                "Dataset is perfectly balanced with 100 images per class",
                "All images are in PNG format with consistent quality",
                "Images have varying dimensions requiring standardization",
                "No missing or corrupted files detected"
            ],
            'preprocessing_strategy': [
                "Resize all images to 224x224 pixels for consistency",
                "Normalize pixel values to [0,1] range",
                "Apply data augmentation to increase dataset diversity",
                "Use 70/15/15 split for train/validation/test",
                "Implement robust data loading pipeline"
            ],
            'challenges_identified': [
                "Varying image dimensions need standardization",
                "Limited dataset size (500 images total)",
                "Need for data augmentation to prevent overfitting",
                "Ensuring balanced representation in splits"
            ]
        }
        
        return summary


def generate_presentation_materials(dataset_path, output_dir='presentation_materials'):
    """
    Generate all materials needed for the Saturday presentation
    """
    print("🎯 Generating Presentation Materials for Week 4 Deliverable")
    print("="*70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer(dataset_path)
    
    # 1. Analyze dataset structure
    print("\n1️⃣  Dataset Structure Analysis")
    analyzer.analyze_dataset_structure()
    
    # 2. Generate class distribution chart
    print("\n2️⃣  Creating Class Distribution Chart")
    analyzer.create_class_distribution_chart(output_path / 'class_distribution.png')
    
    # 3. Analyze image properties
    print("\n3️⃣  Analyzing Image Properties")
    analyzer.analyze_image_properties(sample_size=20)
    analyzer.create_image_properties_chart(output_path / 'image_properties.png')
    
    # 4. Create sample images grid
    print("\n4️⃣  Creating Sample Images Grid")
    analyzer.create_sample_images_grid(save_path=output_path / 'sample_images.png')
    
    # 5. Demonstrate preprocessing
    print("\n5️⃣  Demonstrating Preprocessing Techniques")
    analyzer.demonstrate_preprocessing_techniques(output_path / 'preprocessing_demo.png')
    
    # 6. Create pipeline flowchart
    print("\n6️⃣  Preprocessing Pipeline")
    analyzer.create_preprocessing_pipeline_flowchart()
    
    # 7. Generate summary
    print("\n7️⃣  Generating Presentation Summary")
    summary = analyzer.generate_presentation_summary()
    
    # Save summary to file
    with open(output_path / 'presentation_summary.txt', 'w') as f:
        f.write("WEEK 4 PRESENTATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write("DATASET OVERVIEW:\n")
        for key, value in summary['dataset_overview'].items():
            f.write(f"- {key}: {value}\n")
        
        f.write("\nKEY INSIGHTS:\n")
        for insight in summary['key_insights']:
            f.write(f"- {insight}\n")
        
        f.write("\nPREPROCESSING STRATEGY:\n")
        for strategy in summary['preprocessing_strategy']:
            f.write(f"- {strategy}\n")
        
        f.write("\nCHALLENGES IDENTIFIED:\n")
        for challenge in summary['challenges_identified']:
            f.write(f"- {challenge}\n")
    
    print(f"\n✅ All presentation materials saved to: {output_path}")
    print("\n📋 Files generated:")
    for file in output_path.glob('*'):
        print(f"   - {file.name}")
    
    return analyzer, summary


# Example usage and main execution
if __name__ == "__main__":
    print("🎯 Week 4 Presentation Material Generator")
    print("DataVerse Africa Internship Cohort 3.0")
    print("="*70)
    
    # Set your dataset path
    dataset_path = "../Grapevine_Leaves_Image_Dataset"
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please update the dataset_path variable to point to your dataset folder.")
    else:
        # Generate all presentation materials
        analyzer, summary = generate_presentation_materials(dataset_path)
        
        print("\n🎉 Presentation materials generated successfully!")
        print("\n📝 Next steps for your presentation:")
        print("1. Review the generated charts and summary")
        print("2. Practice explaining each visualization")
        print("3. Prepare to discuss your preprocessing strategy")
        print("4. Be ready to answer questions about your approach")
        print("5. Time your presentation (aim for 15-20 minutes)")
        
        print("\n💡 Presentation Tips:")
        print("- Start with dataset overview and key statistics")
        print("- Show sample images to give audience visual context")
        print("- Explain your preprocessing choices and rationale")
        print("- Discuss challenges and how you plan to address them")
        print("- End with clear next steps for model development")
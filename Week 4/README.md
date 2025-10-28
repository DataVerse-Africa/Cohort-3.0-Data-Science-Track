# Week 4: Deep Learning & Computer Vision
## DataVerse Africa Internship Cohort 3.0

### 🎯 **Project: Crop Disease Detection System**

Welcome to Week 4 of the DataVerse Africa Internship! This week marks our transition into the exciting world of Deep Learning and Computer Vision. We'll be working on building a **Crop Disease Detection System** using grapevine leaf images.

---

## 📅 **Weekly Schedule**

### **Week 4 – Vision & Foundation**

| Day | Focus | Activities |
|-----|-------|------------|
| **Monday (Strategist)** | AI & CV Impact | How AI & CV Improve African Food Security, Healthcare and Finance |
| **Tuesday** | Neural Networks Basics | Backpropagation, OpenCV image preprocessing |
| **Thursday** | Advanced Workshop | Image dataset exploration, preprocessing pipeline design |
| **Saturday** | Presentation | "Dataset Insights + Preprocessing Plan" |
| **Self-study** | TensorFlow Basics | Image augmentation practice |

### **📋 Deliverable**
**Image preprocessing + dataset setup** - Due Saturday

---

## 📁 **Folder Structure**

```
Week 4/
├── README.md                              # This file
├── Grapevine_Leaves_Image_Dataset/        # Main dataset
│   ├── Ak/                               # Class 1: Ak variety (100 images)
│   ├── Ala_Idris/                        # Class 2: Ala_Idris variety (100 images)
│   ├── Buzgulu/                          # Class 3: Buzgulu variety (100 images)
│   ├── Dimnit/                           # Class 4: Dimnit variety (100 images)
│   ├── Nazli/                            # Class 5: Nazli variety (100 images)
│   └── Grapevine_Leaves_Image_Dataset_Citation_Request.txt
├── src/                                   # Source code and notebooks
│   ├── Week4_Deep_Learning_Computer_Vision.ipynb  # Main learning notebook
│   └── Week4_Exercises.py                # Practice exercises
└── presentations/                         # (Create this for your deliverables)
```

---

## 🎯 **Learning Objectives**

By the end of this week, you will be able to:

1. **Understand Neural Networks**: Grasp the fundamentals of neural networks and backpropagation
2. **Master OpenCV**: Use OpenCV for image preprocessing and manipulation
3. **Build CNNs**: Design and implement Convolutional Neural Networks
4. **Handle Image Data**: Load, preprocess, and augment image datasets
5. **Apply Transfer Learning**: Use pre-trained models for computer vision tasks
6. **Evaluate Models**: Assess model performance using appropriate metrics

---

## 📊 **Dataset Overview**

### **Grapevine Leaves Classification Dataset**
- **Total Images**: 500
- **Classes**: 5 (Ak, Ala_Idris, Buzgulu, Dimnit, Nazli)
- **Images per Class**: 100
- **Format**: PNG files
- **Task**: Multi-class classification

### **Research Context**
This dataset was used in the research paper: *"A CNN-SVM study based on selected deep features for grapevine leaves classification"* which achieved **97.60% classification accuracy** using MobileNetv2 CNN with SVMs.

---

## 🚀 **Getting Started**

### **Prerequisites**
Make sure you have the following installed:
```bash
pip install tensorflow opencv-python matplotlib seaborn pandas numpy scikit-learn pillow pathlib
```

### **Quick Start**
1. Open `src/Week4_Deep_Learning_Computer_Vision.ipynb`
2. Follow the notebook step by step
3. Complete the exercises in `src/Week4_Exercises.py`
4. Prepare your presentation for Saturday

---

## 📚 **Key Topics Covered**

### **1. Neural Networks Fundamentals**
- Perceptrons and Multi-layer Perceptrons
- Activation functions (ReLU, Sigmoid, Softmax)
- Backpropagation algorithm
- Gradient descent optimization

### **2. Computer Vision Basics**
- Image representation and color spaces
- Image preprocessing techniques
- Feature extraction and edge detection
- Histogram analysis

### **3. Convolutional Neural Networks (CNNs)**
- Convolution and pooling operations
- CNN architectures and design principles
- Overfitting prevention (Dropout, Regularization)
- Batch normalization

### **4. OpenCV for Image Processing**
- Image loading and manipulation
- Filtering and noise reduction
- Geometric transformations
- Morphological operations

### **5. Transfer Learning**
- Pre-trained models (VGG, ResNet, MobileNet)
- Feature extraction vs fine-tuning
- Model adaptation for new tasks

### **6. Model Evaluation**
- Classification metrics (Accuracy, Precision, Recall, F1)
- Confusion matrices
- Cross-validation for computer vision

---

## 💻 **Hands-on Exercises**

The `Week4_Exercises.py` file contains 6 comprehensive exercises:

1. **Image Preprocessing Techniques** - Compare different preprocessing methods
2. **CNN Architecture Experimentation** - Design and compare different CNN architectures
3. **Data Augmentation Impact Analysis** - Analyze the effect of data augmentation
4. **Transfer Learning Comparison** - Compare different pre-trained models
5. **Model Interpretability** - Understand what your model is learning
6. **Custom Loss Functions and Metrics** - Implement advanced training techniques

---

## 📈 **Saturday Presentation Guidelines**

### **Theme: "Dataset Insights + Preprocessing Plan"**

Your presentation should include:

#### **1. Dataset Analysis (5-7 minutes)**
- Dataset overview and statistics
- Class distribution analysis
- Sample image visualization
- Data quality assessment
- Challenges identified

#### **2. Preprocessing Strategy (5-7 minutes)**
- Chosen preprocessing techniques and rationale
- Data augmentation strategy
- Train/validation/test split approach
- Pipeline design and implementation

#### **3. Technical Implementation (3-5 minutes)**
- Code walkthrough of key functions
- Preprocessing pipeline demonstration
- Performance considerations

#### **4. Next Steps (2-3 minutes)**
- Planned model architectures
- Expected challenges and solutions
- Timeline for model development

### **Presentation Tips**
- Use visual aids (charts, sample images, code snippets)
- Keep technical explanations clear and concise
- Practice your timing
- Prepare for Q&A session

---

## 🔗 **Additional Resources**

### **Essential Reading**
- [Deep Learning Book - Chapter 9: Convolutional Networks](http://www.deeplearningbook.org/)
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

### **Video Tutorials**
- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Andrew Ng: Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)

### **Practical Resources**
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Keras Documentation](https://keras.io/)
- [OpenCV Python Documentation](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

---

## 🏆 **Success Criteria**

To successfully complete Week 4, you should:

- [ ] Complete the main Jupyter notebook
- [ ] Finish at least 3 out of 6 exercises
- [ ] Prepare and deliver your Saturday presentation
- [ ] Submit your preprocessing pipeline code
- [ ] Demonstrate understanding of CNN concepts
- [ ] Show proficiency with OpenCV operations

---

## 🤝 **Getting Help**

### **During the Week**
- Ask questions during live sessions
- Use the cohort Slack/Discord channel
- Form study groups with fellow interns
- Reach out to mentors during office hours

### **Common Issues and Solutions**
- **Memory errors**: Reduce batch size or image resolution
- **Slow training**: Use GPU acceleration or smaller models
- **Import errors**: Check your environment and package versions
- **Dataset loading issues**: Verify file paths and permissions

---

## 📝 **Assessment Rubric**

| Criteria | Excellent (4) | Good (3) | Satisfactory (2) | Needs Improvement (1) |
|----------|---------------|----------|------------------|----------------------|
| **Technical Understanding** | Deep understanding of CNNs and CV concepts | Good grasp of key concepts | Basic understanding | Limited understanding |
| **Code Quality** | Clean, well-documented, efficient | Good structure and documentation | Functional code | Basic functionality |
| **Presentation** | Clear, engaging, well-structured | Good presentation skills | Adequate delivery | Needs improvement |
| **Problem Solving** | Creative solutions, handles edge cases | Good problem-solving approach | Solves basic problems | Struggles with problems |

---

## 🌟 **Week 4 Motivation**

> *"Computer vision is one of the most exciting fields in AI today. The ability to teach machines to 'see' and understand visual content opens up incredible possibilities for solving real-world problems, especially in agriculture and healthcare across Africa."*

Remember: This week is about building a strong foundation in deep learning and computer vision. Take your time to understand the concepts, experiment with the code, and don't hesitate to ask questions. The skills you learn this week will be crucial for the advanced topics in Weeks 5 and 6!

---

**Good luck, and let's build something amazing! 🚀**

---

*For questions or clarifications, reach out to your instructors or use the cohort communication channels.*
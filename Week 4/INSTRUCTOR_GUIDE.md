# Week 4 Instructor Guide
## Deep Learning & Computer Vision - Crop Disease Detection System

### 📋 **Quick Overview**

**Theme**: Vision & Foundation  
**Project**: Crop Disease Detection System using Grapevine Leaves Dataset  
**Deliverable**: Image preprocessing + dataset setup (Due Saturday)  
**Presentation**: "Dataset Insights + Preprocessing Plan"

---

## 📅 **Daily Breakdown**

### **Monday (Strategist Session)**
**Topic**: How AI & CV Improve African Food Security, Healthcare and Finance

**Suggested Activities**:
- Case studies of AI in African agriculture
- Discussion on computer vision applications in healthcare
- Overview of financial technology using image recognition
- Introduction to the week's project context

**Key Points to Cover**:
- Real-world impact of computer vision in Africa
- Crop disease detection importance for food security
- Economic benefits of early disease detection
- Technology accessibility and implementation challenges

---

### **Tuesday**
**Topic**: Neural networks basics, backpropagation, OpenCV image preprocessing

**Materials to Use**:
- `Week4_Deep_Learning_Computer_Vision.ipynb` (Sections 1-4)
- Focus on neural network fundamentals and OpenCV basics

**Hands-on Activities**:
- Neural network visualization exercises
- Basic OpenCV operations (resize, blur, edge detection)
- Image loading and manipulation practice

**Learning Checkpoints**:
- Students can explain backpropagation conceptually
- Students can perform basic image preprocessing with OpenCV
- Understanding of activation functions and neural network layers

---

### **Thursday (Advanced Workshop)**
**Topic**: Image dataset exploration, preprocessing pipeline design

**Materials to Use**:
- `Week4_Deep_Learning_Computer_Vision.ipynb` (Sections 5-8)
- `Week4_Exercises.py` (Exercises 1-3)
- `presentation_template.py` for analysis

**Workshop Structure**:
1. **Dataset Exploration** (45 mins)
   - Run dataset analysis using `presentation_template.py`
   - Discuss findings and insights
   - Identify preprocessing needs

2. **Pipeline Design** (45 mins)
   - Design preprocessing pipeline together
   - Implement data loading and augmentation
   - Test pipeline with sample data

3. **Preparation for Saturday** (30 mins)
   - Review presentation requirements
   - Help students organize their findings
   - Practice presentation structure

---

### **Saturday (Presentation Day)**
**Format**: "Dataset Insights + Preprocessing Plan"

**Evaluation Criteria**:
- Dataset analysis depth and accuracy
- Preprocessing strategy rationale
- Technical implementation quality
- Presentation clarity and structure

**Time Allocation**:
- 15-20 minutes per presentation
- 5 minutes Q&A
- Immediate feedback and suggestions

---

## 🎯 **Learning Objectives Assessment**

### **By End of Week, Students Should**:
- [ ] Understand neural network fundamentals
- [ ] Use OpenCV for image preprocessing
- [ ] Analyze image datasets systematically
- [ ] Design preprocessing pipelines
- [ ] Present technical findings clearly

### **Assessment Methods**:
- Code review of preprocessing pipeline
- Saturday presentation evaluation
- Completion of exercises
- Participation in discussions

---

## 📁 **File Structure Guide**

```
Week 4/
├── README.md                              # Student guide
├── INSTRUCTOR_GUIDE.md                    # This file
├── Grapevine_Leaves_Image_Dataset/        # Dataset (500 images, 5 classes)
├── src/
│   ├── Week4_Deep_Learning_Computer_Vision.ipynb  # Main teaching material
│   ├── Week4_Exercises.py                # Practice exercises
│   └── presentation_template.py          # Presentation helper
└── presentations/                         # Student deliverables
```

---

## 🛠️ **Technical Setup**

### **Required Packages**:
```bash
pip install tensorflow opencv-python matplotlib seaborn pandas numpy scikit-learn pillow pathlib
```

### **Hardware Requirements**:
- Minimum 8GB RAM
- GPU recommended but not required
- 2GB free disk space

### **Common Issues & Solutions**:
- **Memory errors**: Reduce batch size or image resolution
- **OpenCV installation**: Use `pip install opencv-python-headless` if GUI issues
- **TensorFlow GPU**: Ensure CUDA compatibility if using GPU

---

## 📊 **Dataset Information**

### **Grapevine Leaves Dataset**:
- **Classes**: 5 (Ak, Ala_Idris, Buzgulu, Dimnit, Nazli)
- **Images per class**: 100
- **Total images**: 500
- **Format**: PNG
- **Research context**: 97.60% accuracy achieved in original paper

### **Teaching Points**:
- Balanced dataset (good for learning)
- Sufficient size for educational purposes
- Real-world agricultural application
- Clear visual differences between classes

---

## 🎓 **Pedagogical Notes**

### **Key Concepts to Emphasize**:
1. **Visual Learning**: Use lots of image examples
2. **Practical Application**: Connect to real-world problems
3. **Iterative Process**: Preprocessing is experimental
4. **Documentation**: Importance of explaining choices

### **Common Student Challenges**:
- Understanding convolution operations
- Choosing appropriate preprocessing techniques
- Balancing model complexity
- Interpreting model performance

### **Teaching Strategies**:
- Use visual analogies for convolution
- Show before/after preprocessing examples
- Encourage experimentation
- Provide immediate feedback

---

## 📈 **Assessment Rubric**

| Criteria | Weight | Excellent (4) | Good (3) | Satisfactory (2) | Needs Work (1) |
|----------|--------|---------------|----------|------------------|----------------|
| **Dataset Analysis** | 30% | Comprehensive insights, identifies patterns | Good analysis, some insights | Basic analysis completed | Superficial analysis |
| **Preprocessing Strategy** | 30% | Well-reasoned choices, considers trade-offs | Good strategy with rationale | Adequate preprocessing plan | Basic or unclear strategy |
| **Technical Implementation** | 25% | Clean, efficient, well-documented code | Good implementation, minor issues | Functional code, some problems | Basic functionality only |
| **Presentation** | 15% | Clear, engaging, well-structured | Good presentation skills | Adequate delivery | Unclear or disorganized |

---

## 🔄 **Feedback Guidelines**

### **During the Week**:
- Provide immediate feedback on code
- Encourage experimentation
- Help debug technical issues
- Guide conceptual understanding

### **For Saturday Presentations**:
- Focus on learning process, not just results
- Highlight good analytical thinking
- Suggest improvements for next week
- Connect to upcoming transfer learning topics

---

## 📚 **Additional Resources for Instructors**

### **Background Reading**:
- [CS231n Course Notes](http://cs231n.github.io/)
- [Deep Learning Book - Chapter 9](http://www.deeplearningbook.org/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

### **Supplementary Materials**:
- Neural network visualization tools
- Additional datasets for comparison
- Industry case studies
- Research papers on agricultural AI

---

## 🚀 **Preparation for Week 5**

### **Bridge to Next Week**:
- Review transfer learning concepts
- Prepare pre-trained model examples
- Set up advanced CNN architectures
- Plan model evaluation strategies

### **Student Preparation**:
- Complete any unfinished exercises
- Review CNN fundamentals
- Prepare questions about transfer learning
- Ensure technical setup is working

---

## 💡 **Tips for Success**

1. **Keep it Visual**: Use lots of images and diagrams
2. **Encourage Questions**: Create safe space for learning
3. **Hands-on Focus**: More coding, less theory
4. **Real-world Context**: Always connect to applications
5. **Celebrate Progress**: Acknowledge improvements and insights

---

**Remember**: This week is about building confidence with computer vision concepts. Focus on understanding over perfection, and ensure students feel prepared for the more advanced topics in Weeks 5 and 6.

---

*For questions or suggestions about this guide, please reach out to the curriculum team.*
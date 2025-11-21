# Week 02 — Vision Transformer (ViT) Image Classifier  
### Hugging Face Weekly Model Series – by Tomer Biton

This week's model: **google/vit-base-patch16-224**  
A Vision Transformer (ViT) trained for image classification on ImageNet-1k.

## 🔍 What is ViT?

**ViT (Vision Transformer)** applies the same transformer architecture used in NLP to image patches.  
Instead of CNN filters, the image is split into **16×16 patches**, embedded, and processed like tokens.

### Why it matters
- State-of-the-art performance on image classification  
- Lighter and simpler than many CNNs  
- Easy to use for real-world computer vision tasks  
- Works out-of-the-box with Hugging Face pipelines and feature extractors

## 📦 What this mini-project includes
✔ A Python script to classify single images   
✔ Simple, reproducible setup  
✔ Ready for experimentation / extension


## 🧠 Model Used  
**Model:** `google/vit-base-patch16-224`  
**Task:** Image Classification  
**Dataset:** ImageNet-1k  
**Patch Size:** 16×16  
**Input Size:** 224×224  


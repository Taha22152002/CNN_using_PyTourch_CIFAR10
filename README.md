

# 📦 CIFAR-10 Image Classification with PyTorch

This project demonstrates how to build, train, evaluate, and visualize predictions of a **Convolutional Neural Network (CNN)** using **PyTorch** on the **CIFAR-10** dataset. The goal is to classify images into one of ten classes such as airplane, cat, ship, etc.

> ✅ This is a Jupyter Notebook-based project for ease of experimentation and visualization.

---

## 📌 Features Covered

* 📚 Dataset loading and normalization using `torchvision`
* 🧠 CNN model creation using `torch.nn`
* 🏋️‍♂️ Model training using `SGD` and `CrossEntropyLoss`
* 📈 Model evaluation and accuracy measurement
* 🖼️ Image prediction and visualization
* 💾 Saving and loading model weights using `torch.save()` and `torch.load()`

---

## 🧪 Requirements

Install the required dependencies before running the notebook:

```bash
pip install torch torchvision matplotlib
```

You can also use a Colab environment if PyTorch is not available locally.

---

## 🚀 How to Run

1. **Download or clone the repo** and open the notebook file:

   ```bash
   CNN_using_PyTorch_CIFAR10.ipynb
   ```

2. **Launch Jupyter Notebook:**

   ```bash
   jupyter notebook CNN_using_PyTorch_CIFAR10.ipynb
   ```

3. **Run the notebook cell by cell** to:

   * Load CIFAR-10 dataset
   * Define the CNN architecture
   * Train the model
   * Evaluate accuracy
   * Visualize predictions

---

## 🧠 Model Architecture

```text
Input: 3x32x32 (RGB image)
↓
Conv2D(3 → 6 filters) + ReLU → MaxPooling
↓
Conv2D(6 → 16 filters) + ReLU → MaxPooling
↓
Flatten
↓
FC (16*5*5 → 120) → ReLU
↓
FC (120 → 84) → ReLU
↓
FC (84 → 10)
↓
Softmax (handled internally in loss function)
```

---

## 📊 Sample Predictions

The notebook plots a batch of test images with their **predicted vs. actual** labels:

<img width="434" height="145" alt="image" src="https://github.com/user-attachments/assets/931dc1ce-6a89-47f9-86df-27258aa7350f" />

---

## 📁 Files

* `CNN_using_PyTorch_CIFAR10.ipynb` – Main notebook with complete code and explanation.
* `cifar10_predictions.png` – Visualization of model predictions (generated during runtime).
* `cifar_net.pth` – Saved model weights (generated during runtime).

---

## 📌 Notes

* The images may look **pixelated/blurry** due to CIFAR-10's low resolution (32×32).
* The notebook uses **random batches** for prediction display.
* GPU acceleration (if available) is recommended for faster training.

---

## 🤝 Credits

Built with ❤️ using [PyTorch](https://pytorch.org/) and [Torchvision](https://pytorch.org/vision/).



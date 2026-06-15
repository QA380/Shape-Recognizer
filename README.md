Time Capsule<br>
Project archived until reinstated. 01-06-15-26-1118<br>
Decode if necessary. once a measure becomes a target it ceases to be a good measure.<br>
for you, be safe.<br>


































# Shape-Recognizer
A simple 2D shape recignizer from a handful set of training data, using python.

# Interactive CNN Sandbox & Machine Learning Dashboard Documentation

This comprehensive technical guide breaks down the architecture, underlying machine learning concepts, and operational mechanics of the `CNN_Sandbox.py` application. This application serves as an interactive environment for building, training, and diagnosing a Convolutional Neural Network (CNN) directly on custom, hand-drawn data using PyTorch and Tkinter.

---

## Section 1: Conceptual Core (The Machine Learning Subject)

To understand how the application functions, it is essential to explore the core machine learning paradigms operating beneath the interface.

### 1. What is a Convolutional Neural Network (CNN)?

Traditional neural networks (Fully Connected layers) treat images as flat vectors of pixels, completely discarding spatial structures. CNNs, however, preserve spatial geometry by using a concept called **Convolutions**.

* **Filters/Kernels:** The model passes small sliding windows (in this code, $3 \times 3$ matrices) across the image. These filters compute dot products to detect specific edge details, lines, and textures.
* **Feature Maps:** The output of a convolution is a map highlighting where specific shapes or lines occurred in the drawing.
* **Downsampling (MaxPooling):** To make the model invariant to small translations (e.g., whether you drew a circle exactly in the center or slightly to the left), a $2 \times 2$ MaxPool filter slides across the feature maps, choosing only the maximum value in each quadrant. This reduces image size by half, keeping only the most dominant features and shrinking the computational load.

### 2. The Standardized Dimension Pipeline ($28 \times 28$)

When you draw on the canvas, your trackpad or mouse produces a jagged sequence of coordinate vectors on a $200 \times 200$ pixel canvas. A raw machine learning algorithm cannot easily ingest inputs of varying scales, speeds, or stroke thicknesses.
To resolve this, the code employs a rigid preprocessing canvas translation:

1. **Inversion & Bounding Box:** Grayscale drawings are inverted so that the background becomes mathematical $0$ (black) and strokes become values up to $255$ (white). It automatically crops tight boundaries around your drawing using a bounding box.
2. **Squaring & Padding:** It adds a uniform $15\%$ padding margin to prevent edge features from getting clipped by convolution filters.
3. **Resampling:** The image is downsampled to exactly $28 \times 28$ pixels using Lanczos filtering. This compresses your drawing into a compact, standardized tensor representation, matching classical computer vision benchmark formats like MNIST.

### 3. Data Augmentation (The 4x Multiplier Effect)

Deep learning models are notoriously data-hungry. If you provide a model with only 5 drawings of a square, it will memorize the exact pixel locations of those 5 lines. To mitigate this without forcing you to draw thousands of lines, the application performs **Synthetic Data Augmentation** at the moment of saving:

* **Original:** The exact drawing centered at $28 \times 28$.
* **Clockwise Rotation:** The image is rotated exactly $10^\circ$ clockwise.
* **Counter-Clockwise Rotation:** The image is rotated exactly $10^\circ$ counter-clockwise.
* **Gaussian Noise:** Random mathematical jitter between $[-30, 30]$ is injected into each pixel.

This means **1 canvas click = 4 unique images on disk**. This forces the model to look at the holistic geometry of your drawing instead of local pixel values.

### 4. Mathematical Engine: Optimization & Cross-Entropy Loss

* **Cross-Entropy Loss:** Since this is a classification problem (e.g., distinguishing Circle vs. Square), the network outputs an unnormalized score (logit) for each shape class. Cross-entropy calculates the difference between the model's soft probability distribution and the true, sharp representation (one-hot vector). A lower score means higher target confidence.
* **Stochastic Mini-Batch Gradient Descent:** Rather than looking at every image simultaneously or updating weights after every individual image, the engine segments the training data into random blocks of up to **16 images** (Batches). It computes the average error of that batch, calculates gradients via backpropagation, and steps toward a minimized error surface.
* **Adam Optimizer:** An advanced variant of Gradient Descent that maintains adaptive learning rates for each independent weight parameter based on moving averages of recent gradients.

---

## Section 2: Code Architecture Breakdown

The code is cleanly modularized into structural blocks. Below is an exhaustive breakdown of the functional logic.

### 1. The Convolutional Neural Network Structure

```python
class PyTorchCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Dropout(p=0.2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(8 * 13 * 13, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

```

* **`nn.Conv2d(1, 8, kernel_size=3)`**: Extracts basic lines and textures. Takes 1 grayscale channel and outputs 8 separate hidden feature maps. A $28 \times 28$ input through a $3 \times 3$ kernel yields a spatial size of $26 \times 26$.
* **`nn.ReLU()`**: Eliminates linearity. Replaces all negative values with absolute zeros ($0$), allowing the network to compute complex, non-linear geometric splits.
* **`nn.MaxPool2d(2, 2)`**: Condenses spatial regions. Shrinks the $26 \times 26$ feature map dimensions down to exactly $13 \times 13$.
* **`nn.Flatten()`**: Unrolls the spatial matrix into a flat 1D array. With 8 distinct maps at a size of $13 \times 13$, the flattened layer yields a vector size of exactly $8 \times 13 \times 13 = 1352$ features.
* **`nn.Dropout(p=0.2)`**: A vital regularization line. It randomly deactivates $20\%$ of neurons on every single forward pass during training. This prevents complex co-adaptations and forces the model to develop robust redundant feature detection pathways.
* **`nn.Linear(1352, 32)`**: Downsamples the high-dimensional features into a dense vector of 32 general concepts.
* **`nn.Linear(32, num_classes)`**: Map those final concepts into raw classification logits for your unique dataset labels.

### 2. The Training Framework Pipeline

When clicking the **START OPTIMIZATION LOOP** button, the pipeline executes the following operational state sequence:

```python
def start_training_pipeline(self):
    ...
    # 1. Read files and convert to normalized 1x28x28 numpy matrices
    # 2. Shuffle data randomly to ensure balanced mini-batch distributions
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

```

* **The 80/20 Holdout Split:** The model shuffles all synthetic image structures on disk and divides them. $80\%$ of the images are assigned to `X_train`—this is the material the model treats as an open-book assignment to optimize its parameters. $20\%$ is isolated as `X_val`—the validation test bed to score generalization.

```python
def run_training_epoch_step(self):
    ...
    self.model.train()
    # Execute Stochastic Mini-Batch loop across dataset
    for i in range(0, dataset_size, batch_size):
        ...
        self.optimizer.zero_grad() # Clear out gradients from previous step
        outputs = self.model(batch_x) # Forward Pass
        loss_tensor = self.criterion(outputs, batch_y) # Compute Loss Error
        loss_tensor.backward() # Backpropagation (Reverse derivative pass)
        self.optimizer.step() # Tweak weights via Adam formula

```

* **Asynchronous GUI Refresh Loop:** At the end of every epoch, the system executes validation profiling (`self.model.eval()`), tracks metrics inside historical arrays, updates the GUI matplotlib charts, and leverages `self.root.after(10, self.run_training_epoch_step)` to loop without freezing the Tkinter application interface window.

### 3. Early Stopping Mechanics

To prevent extreme over-memorization, the script maintains an automated performance monitor:

```python
if val_loss < (self.best_val_loss - 1e-4):
    self.best_val_loss = val_loss
    self.patience_counter = 0
else:
    self.patience_counter += 1
    if self.patience_counter >= 100:
        self.training_active = False
        self.btn_train.config(text="EARLY STOPPING TRIGGERED", bg=COLOR_BLUE)

```

If the Validation Loss curve (the test set score) completely stops improving for **100 consecutive epochs**, the framework halts the optimization loop to preserve the best generalization configuration.

---

## Section 3: App Operation & Dashboard Telemetry Manual

Operating this application efficiently requires understanding how to manage your data budget and interpret live charts.

### 1. Operation Walkthrough

```
[ Step 1: Data Entry ] ----> [ Step 2: Set Hyperparameters ] ----> [ Step 3: Run Loop ]
 Draw shape on canvas          Adjust Learning Rate (α) Slider      Click Green Opt Button
 Enter text tag label          (Recommended: 0.001 - 0.005)         Watch metrics descend
 Click "Save Data"

```

* **Step 1 (Data Collection):** Draw a shape on the left input canvas, type a class identifier string into the center entry field (e.g., `circle`), and click **Save Data**. Clear the canvas and repeat this process multiple times.
* **Step 2 (Hyperparameter Selection):** Locate the **Learning Rate ($\alpha$)** slider. This determines how large a stride the optimization engine takes down the mathematical error slope.
* **Step 3 (Execution):** Tap the green **START OPTIMIZATION LOOP** button to initialize training. Tap it again at any point to pause optimization, adjust parameters mid-flight, and resume.

---

### 2. Reading the Chart & Diagnostics

During training, look closely at the **Telemetry & Optimization Tuning** sub-graph area. This visualizer tells you everything you need to know about your model's stability.

#### A. A Good Fit Chart (Goal Configuration)

* **Visual Behavior:** Both the **solid blue line (Train Loss)** and the **dashed red line (Val Loss)** drop sharply in a tight, parallel trajectory down toward $0.1$ or lower.
* **Mathematical Meaning:** The model is learning general geometric invariants (angles, spatial density, curves) that apply evenly across both familiar training files and fresh validation files.
* **Action:** Let it run or save the weights immediately; the model will be robust on the live canvas.

#### B. An Overfitting Chart (The Data Scarcity Trap)

* **Visual Behavior:** The blue line plummets toward $0.0$, but the dashed red line decouples, flattens out early, or begins to **climb sharply back upward**.
* **Mathematical Meaning:** The model has run out of general patterns to learn and is now memorizing the unique pixel paths of your training drawings. It scores 100% on the training set but fails on validation checks.
* **Fix Strategy:** 1. Click **Purge Dataset**, draw a broader variety of samples, and aim for a larger data budget (e.g., a total dataset allocation reading of $>150$ total files).
2. Slide the **Learning Rate ($\alpha$)** down lower (to `0.001` or `0.002`) to prevent the network weights from aggressively optimizing for outliers.

#### C. An Underfitting / Volatile Chart

* **Visual Behavior:** The lines display extreme zigzag behaviors, spiking violently up and down across successive epochs without flattening.
* **Mathematical Meaning:** The learning rate is set too high. The optimizer is taking steps so massive that it keeps overshooting the optimal weight valley, bouncing erratically across the loss landscape.
* **Fix Strategy:** Drag the learning rate slider down to a smaller value immediately.

---

### 3. Deep Telemetry Analytics

Clicking the blue **Telemetry Analytics** button launches an evaluation dashboard containing two advanced model diagnostic panels:

```
           CONFUSION MATRIX                          PER-CLASS ACCURACY
       
         Predicted: Circle  Square                100% |------|
        +-------------------------+                    |      |
 True:  |   Circle   [12]     [1] |                    |      |
 Circle |                         |                    |      |   |------|
        |                         |                 50%|      |   |      |
 True:  |   Square   [0]      [14]|                    |      |   |      |
 Square |                         |                    +------------------
        +-------------------------+                      Circle  Square

```

1. **Validation Confusion Matrix:** This grid cross-references the true structural ground-truth labels against the network's actual mathematical choices.
* **The Main Diagonal:** The path cutting from top-left to bottom-right tracks correct classifications. You want high integers here.
* **Off-Diagonal Values:** If you see a high number in an off-diagonal cell, it tells you exactly where the model is confused (e.g., misclassifying hand-drawn triangles as squares because their lines look too blocky).


2. **Per-Class Accuracy Bar Chart:** Displays a normalized bar plot showing accuracy percentages for each class. If one bar is at 100% and another is at 20%, it means your dataset is highly imbalanced or one shape needs to be drawn more distinctly to help the model differentiate its geometric features.

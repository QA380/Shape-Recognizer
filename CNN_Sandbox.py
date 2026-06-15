import os
import time
import shutil
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageDraw, ImageOps

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import production library core modules
import torch
import torch.nn as nn
import torch.optim as optim

# =======================================================
# GUI DESIGN CONFIGURATION
# =======================================================
BG_MAIN = "#111111"
BG_PANEL = "#1a1a1a"
BG_INSIDE = "#222222"
TEXT_MAIN = "#ececec"
TEXT_MUTED = "#888888"
COLOR_BLUE = "#58c4dd"
COLOR_GREEN = "#26ceaa"
COLOR_RED = "#e07a5f"
COLOR_YELLOW = "#f4d35e"

FONT_TITLE = ("Helvetica", 11, "bold")
FONT_BODY = ("Helvetica", 10)


# =======================================================
# PRODUCTION PYTORCH CNN ARCHITECTURE
# =======================================================
class PyTorchCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Matches dimensions: Conv(3x3) -> ReLU -> MaxPool(2x2) -> Dropout -> Dense(32)
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

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =======================================================
# DASHBOARD APPLICATION CORE INTERFACE
# =======================================================
class CNNMonitorDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("V4 Production Framework (PyTorch Engine)")
        self.root.configure(bg=BG_MAIN)
        self.root.geometry("1420x860")

        self.train_dir = "training_data"
        if not os.path.exists(self.train_dir): os.makedirs(self.train_dir)

        self.is_trained = False
        self.training_active = False
        self.predict_changed = False
        self.img_size = (28, 28)
        self.classes_ = []

        # Extended Tracking
        self.history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

        self.canvas_size = 200
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=2)

        self._build_input_panel()
        self._build_evaluation_panel()
        self._build_telemetry_panel()

        self.update_data_counts()
        self.root.after(50, self.continuous_predict)

    def _build_input_panel(self):
        self.frame_train = tk.LabelFrame(self.root, text=" Neural Input Network Builder ", bg=BG_PANEL, fg=TEXT_MAIN,
                                         font=FONT_TITLE, padx=10, pady=10)
        self.frame_train.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.canvas_train = tk.Canvas(self.frame_train, width=self.canvas_size, height=self.canvas_size, bg='white',
                                      cursor='cross', highlightthickness=0)
        self.canvas_train.pack(pady=5)
        self.canvas_train.bind('<B1-Motion>', self.paint_train)
        self.image_train = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw_train = ImageDraw.Draw(self.image_train)

        self.label_entry = tk.Entry(self.frame_train, bg=BG_INSIDE, fg=TEXT_MAIN, insertbackground=TEXT_MAIN,
                                    font=FONT_BODY, justify="center")
        self.label_entry.pack(pady=5, ipady=3, fill="x")

        tk.Button(self.frame_train, text="Save Data", command=self.save_training_data, bg=BG_INSIDE,
                  fg=TEXT_MAIN, bd=0).pack(fill="x", pady=2)
        tk.Button(self.frame_train, text="Clear Canvas", command=self.clear_train, bg=BG_INSIDE, fg=TEXT_MAIN,
                  bd=0).pack(fill="x", pady=2)

        self.lbl_data_counts = tk.Label(self.frame_train, text="Dataset Allocation:\nNone", justify="left",
                                        bg=BG_INSIDE, fg=TEXT_MAIN, font=FONT_BODY)
        self.lbl_data_counts.pack(fill="x", pady=10, ipady=5)

        # File I/O Management
        btn_frame = tk.Frame(self.frame_train, bg=BG_PANEL)
        btn_frame.pack(fill="x", pady=2)
        tk.Button(btn_frame, text="Save Model", command=self.save_model, bg=BG_INSIDE, fg=COLOR_BLUE, bd=0).pack(
            side="left", fill="x", expand=True, padx=2)
        tk.Button(btn_frame, text="Load Model", command=self.load_model, bg=BG_INSIDE, fg=COLOR_YELLOW, bd=0).pack(
            side="right", fill="x", expand=True, padx=2)

        # FIX: Action layout containing interactive reset commands
        btn_frame2 = tk.Frame(self.frame_train, bg=BG_PANEL)
        btn_frame2.pack(fill="x", pady=2)
        tk.Button(btn_frame2, text="Reset Training", command=self.reset_training, bg=BG_INSIDE, fg=COLOR_RED,
                  bd=0).pack(
            side="left", fill="x", expand=True, padx=2)
        tk.Button(btn_frame2, text="Purge Dataset", command=self.reset_data, bg=BG_INSIDE, fg=TEXT_MUTED, bd=0).pack(
            side="right", fill="x", expand=True, padx=2)

        self.btn_train = tk.Button(self.frame_train, text="START OPTIMIZATION LOOP", command=self.toggle_training,
                                   bg=COLOR_GREEN, fg=BG_MAIN, font=FONT_TITLE, bd=0, pady=8)
        self.btn_train.pack(fill="x", pady=10)

    def _build_evaluation_panel(self):
        self.frame_predict = tk.LabelFrame(self.root, text=" Real-Time Signal Evaluation ", bg=BG_PANEL, fg=TEXT_MAIN,
                                           font=FONT_TITLE, padx=10, pady=10)
        self.frame_predict.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.canvas_predict = tk.Canvas(self.frame_predict, width=self.canvas_size, height=self.canvas_size, bg='white',
                                        cursor='cross', highlightthickness=0)
        self.canvas_predict.pack(pady=5)
        self.canvas_predict.bind('<B1-Motion>', self.paint_predict)
        self.image_predict = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw_predict = ImageDraw.Draw(self.image_predict)

        tk.Button(self.frame_predict, text="Clear Canvas", command=self.clear_predict, bg=BG_INSIDE, fg=TEXT_MAIN,
                  bd=0).pack(fill="x", pady=4)

        self.lbl_shape = tk.Label(self.frame_predict, text="Final output: None", bg=BG_PANEL, fg=COLOR_BLUE,
                                  font=FONT_TITLE, anchor="w")
        self.lbl_shape.pack(anchor="w", pady=6)

        self.fig_eval = Figure(figsize=(3.8, 2.3), dpi=95, facecolor=BG_PANEL)
        self.ax_img = self.fig_eval.add_subplot(121, facecolor=BG_MAIN)
        self.ax_img.axis('off')
        self.im_display = self.ax_img.imshow(np.zeros((28, 28)), cmap='gray', vmin=0, vmax=1)
        self.ax_bar = self.fig_eval.add_subplot(122, facecolor=BG_MAIN)
        for spine in self.ax_bar.spines.values(): spine.set_color("#2d2d2d")

        self.canvas_eval = FigureCanvasTkAgg(self.fig_eval, master=self.frame_predict)
        self.canvas_eval.get_tk_widget().pack(fill="x", expand=True)

        tk.Button(self.frame_predict, text="Telemetry Analytics", command=self.open_analytics_dashboard,
                  bg=COLOR_BLUE, fg=BG_MAIN, font=FONT_TITLE, bd=0, pady=6).pack(fill="x", pady=8)

    def _build_telemetry_panel(self):
        self.frame_monitor = tk.LabelFrame(self.root, text=" Telemetry & Optimization Tuning ", bg=BG_PANEL,
                                           fg=TEXT_MAIN, font=FONT_TITLE, padx=10, pady=10)
        self.frame_monitor.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

        self.tuner_panel = tk.Frame(self.frame_monitor, bg=BG_INSIDE, padx=5, pady=5)
        self.tuner_panel.pack(fill="x")

        tk.Label(self.tuner_panel, text="Learning Rate (α)", bg=BG_INSIDE, fg=TEXT_MAIN).grid(row=0, column=0,
                                                                                              sticky="w")
        self.slider_lr = tk.Scale(self.tuner_panel, from_=0.001, to=0.05, resolution=0.001, orient="horizontal",
                                  bg=BG_INSIDE, fg=TEXT_MAIN, highlightthickness=0)
        self.slider_lr.set(0.005)
        self.slider_lr.grid(row=0, column=1, sticky="we")

        self.lbl_telemetry = tk.Label(self.frame_monitor, text="Engine Diagnostics:\nWaiting...", bg="#0a0a0a",
                                      fg=TEXT_MAIN, justify="left", font=("Consolas", 10))
        self.lbl_telemetry.pack(fill="x", pady=10)

        self.fig_loss = Figure(figsize=(4, 3), dpi=90, facecolor=BG_PANEL)
        self.ax_loss = self.fig_loss.add_subplot(111, facecolor=BG_MAIN)
        self.ax_loss.set_title("Training vs Validation Loss", color=TEXT_MAIN)
        self.line_loss, = self.ax_loss.plot([], [], color=COLOR_BLUE, label='Train Loss')
        self.line_val_loss, = self.ax_loss.plot([], [], color=COLOR_RED, linestyle="--", label='Val Loss')
        self.ax_loss.legend(facecolor=BG_PANEL, edgecolor=BG_MAIN, labelcolor=TEXT_MAIN)
        self.canvas_loss = FigureCanvasTkAgg(self.fig_loss, master=self.frame_monitor)
        self.canvas_loss.get_tk_widget().pack(fill="both", expand=True)

    def paint_train(self, event):
        self._paint(event, self.canvas_train, self.draw_train)

    def paint_predict(self, event):
        self._paint(event, self.canvas_predict, self.draw_predict)
        self.predict_changed = True

    def _paint(self, event, canvas, draw_obj):
        r = 7
        canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill='black', outline='black')
        draw_obj.ellipse([event.x - r, event.y - r, event.x + r, event.y + r], fill='black')

    def clear_train(self):
        self.canvas_train.delete("all")
        self.image_train = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw_train = ImageDraw.Draw(self.image_train)

    def clear_predict(self):
        self.canvas_predict.delete("all")
        self.image_predict = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw_predict = ImageDraw.Draw(self.image_predict)
        self.predict_changed = False
        self.lbl_shape.config(text="Final output: None")
        self.im_display.set_data(np.zeros((28, 28)))
        self.ax_bar.clear()
        self.canvas_eval.draw_idle()

    def update_data_counts(self):
        counts = {}
        total = 0
        if os.path.exists(self.train_dir):
            for label in sorted(os.listdir(self.train_dir)):
                ld = os.path.join(self.train_dir, label)
                if os.path.isdir(ld):
                    c = len([f for f in os.listdir(ld) if f.endswith('.png')])
                    if c > 0: counts[label] = c; total += c
        self.classes_ = list(counts.keys())
        text = "Dataset Allocation:\n" + "".join([f" - {k}: {v}\n" for k, v in counts.items()]) + f"Total: {total}"
        self.lbl_data_counts.config(text=text if total > 0 else "Dataset Allocation:\nEmpty")

    def reset_data(self):
        if messagebox.askyesno("Confirm Purge", "Completely erase local image dataset files from disk?"):
            self.training_active = False
            shutil.rmtree(self.train_dir)
            os.makedirs(self.train_dir)
            self.update_data_counts()
            self.clear_train()
            self.clear_predict()
            self.reset_training_state()

    # FIX: Expose structural pipeline resets without deleting collected data images
    def reset_training(self):
        if messagebox.askyesno("Confirm Reset", "Reset neural network weights and clear all graph histories?"):
            self.reset_training_state()
            messagebox.showinfo("Reset Complete", "Model re-initialized. Ready to train.")

    def reset_training_state(self):
        self.training_active = False
        if hasattr(self, 'model'):
            delattr(self, 'model')
        self.history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        self.is_trained = False
        self.lbl_telemetry.config(text="Engine Diagnostics:\nReset complete. Ready.")
        self.line_loss.set_data([], [])
        self.line_val_loss.set_data([], [])
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        self.canvas_loss.draw_idle()
        self.btn_train.config(text="START OPTIMIZATION LOOP", bg=COLOR_GREEN)

    # FIX: Processing raw canvas updates into a canonical centered 28x28 grayscale image before writing to disk
    def save_training_data(self):
        shape_label = self.label_entry.get().strip().lower()
        if not shape_label: return
        ld = os.path.join(self.train_dir, shape_label)
        if not os.path.exists(ld): os.makedirs(ld)

        ts = int(time.time() * 1000)

        img_inverted = ImageOps.invert(self.image_train.convert('L'))
        bbox = img_inverted.getbbox()
        if bbox:
            cropped = self.image_train.crop(bbox)
            w, h = cropped.size
            max_dim = max(w, h)
            margin = int(max_dim * 0.15) + 4
            square_size = max_dim + (2 * margin)
            square_canvas = Image.new('L', (square_size, square_size), 'white')
            square_canvas.paste(cropped, ((square_size - w) // 2, (square_size - h) // 2))
            img_28 = square_canvas.resize(self.img_size, Image.Resampling.LANCZOS)
        else:
            img_28 = self.image_train.resize(self.img_size, Image.Resampling.LANCZOS)

        # Write out standardized 28x28 pixel assets
        img_28.save(os.path.join(ld, f"{ts}_orig.png"))

        # FIX: Changed fill_color string 'white' to integer 255 for standard 'L' mode operations
        img_28.rotate(10, fill_color=255).save(os.path.join(ld, f"{ts}_rotcw.png"))
        img_28.rotate(-10, fill_color=255).save(os.path.join(ld, f"{ts}_rotccw.png"))

        # Fix noise implementation to scale values relative to background
        arr = np.array(img_28)
        noise = np.random.randint(-30, 30, arr.shape)
        arr_noisy = np.clip(arr + noise, 0, 255).astype('uint8')
        Image.fromarray(arr_noisy).save(os.path.join(ld, f"{ts}_noise.png"))

        self.update_data_counts()
        self.clear_train()

    def process_image_for_cnn(self, img):
        img_inverted = ImageOps.invert(img.convert('L'))
        bbox = img_inverted.getbbox()
        if bbox:
            cropped = img_inverted.crop(bbox)
            w, h = cropped.size
            max_dim = max(w, h)
            margin = int(max_dim * 0.15) + 4
            square_size = max_dim + (2 * margin)
            square_canvas = Image.new('L', (square_size, square_size), 0)
            square_canvas.paste(cropped, ((square_size - w) // 2, (square_size - h) // 2))
            img_final = square_canvas.resize(self.img_size, Image.Resampling.LANCZOS)
        else:
            img_final = img_inverted.resize(self.img_size, Image.Resampling.LANCZOS)
        return np.array(img_final, dtype=np.float32)[np.newaxis, :, :] / 255.0

    def save_model(self):
        if not hasattr(self, 'model'): return
        path = filedialog.asksaveasfilename(defaultextension=".pt",
                                            filetypes=[("PyTorch Weights", "*.pt"), ("Pickle Models", "*.pkl")])
        if path:
            torch.save({
                'classes': self.classes_,
                'model_state_dict': self.model.state_dict()
            }, path)
            messagebox.showinfo("Saved", "Model weights saved successfully.")

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Weights", "*.pt"), ("Pickle Models", "*.pkl")])
        if path:
            checkpoint = torch.load(path)
            self.classes_ = checkpoint['classes']
            self.model = PyTorchCNN(num_classes=len(self.classes_))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.slider_lr.get())
            self.is_trained = True
            messagebox.showinfo("Loaded", "PyTorch Model successfully loaded.")

    def toggle_training(self):
        if self.training_active:
            self.training_active = False
            self.btn_train.config(text="RESUME OPTIMIZATION", bg=COLOR_GREEN)
        else:
            self.start_training_pipeline()

    def start_training_pipeline(self):
        self.update_data_counts()
        if len(self.classes_) < 2: return

        X_raw, y_raw = [], []
        for label_idx, label_name in enumerate(self.classes_):
            ld = os.path.join(self.train_dir, label_name)
            for f in os.listdir(ld):
                if f.endswith(".png"):
                    img = Image.open(os.path.join(ld, f)).convert('L')
                    X_raw.append(self.process_image_for_cnn(img))
                    y_raw.append(label_idx)

        if len(X_raw) < 4: return

        indices = np.arange(len(X_raw))
        np.random.shuffle(indices)
        X = np.array(X_raw, dtype=np.float32)[indices]
        y = np.array(y_raw, dtype=np.int32)[indices]

        split = int(0.8 * len(X))
        if split == 0 or split == len(X):
            split = max(1, len(X) - 1)

        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        self.X_train_t = torch.tensor(X_train, dtype=torch.float32)
        self.y_train_t = torch.tensor(y_train, dtype=torch.long)
        self.X_val_t = torch.tensor(X_val, dtype=torch.float32)
        self.y_val_t = torch.tensor(y_val, dtype=torch.long)

        # Force a fresh initialization if history is empty (prevents leaking old weights across different runs)
        if not hasattr(self, 'model') or len(self.history['loss']) == 0 or self.model.classifier[2].out_features != len(
                self.classes_):
            self.model = PyTorchCNN(num_classes=len(self.classes_))
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.slider_lr.get())
            self.history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
            self.best_val_loss = float('inf')
            self.patience_counter = 0

        self.training_active = True
        self.btn_train.config(text="PAUSE OPTIMIZATION", bg=COLOR_RED)
        self.run_training_epoch_step()

    # FIX: Refactored execution code into a Stochastic Mini-Batch Gradient Descent loop
    def run_training_epoch_step(self):
        if not self.training_active: return

        current_lr = self.slider_lr.get()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

        self.model.train()
        dataset_size = len(self.X_train_t)
        batch_size = min(16, dataset_size)

        permutation = torch.randperm(dataset_size)
        epoch_loss = 0.0
        correct_preds = 0

        for i in range(0, dataset_size, batch_size):
            indices = permutation[i: i + batch_size]
            batch_x = self.X_train_t[indices]
            batch_y = self.y_train_t[indices]

            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss_tensor = self.criterion(outputs, batch_y)
            loss_tensor.backward()
            self.optimizer.step()

            epoch_loss += loss_tensor.item() * len(indices)
            preds = torch.argmax(outputs, dim=1)
            correct_preds += (preds == batch_y).sum().item()

        loss = epoch_loss / dataset_size
        acc = correct_preds / dataset_size

        # PyTorch Validation
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(self.X_val_t)
            val_loss_tensor = self.criterion(val_outputs, self.y_val_t)
            val_loss = val_loss_tensor.item()
            val_preds = torch.argmax(val_outputs, dim=1)
            val_acc = (val_preds == self.y_val_t).float().mean().item()

        self.history['loss'].append(loss)
        self.history['acc'].append(acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(current_lr)

        epoch = len(self.history['loss'])
        self._update_main_ui(epoch, loss, val_loss)

        if epoch >= 300:
            self.training_active = False
            self.is_trained = True
            self.btn_train.config(text="OPTIMIZATION COMPLETE", bg=COLOR_BLUE)
            return

        # Expanded early stopping patience constraint threshold to 100 epochs to accommodate batches smoothly
        if val_loss < (self.best_val_loss - 1e-4):
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= 100:
                self.training_active = False
                self.is_trained = True
                self.btn_train.config(text="EARLY STOPPING TRIGGERED", bg=COLOR_BLUE)
                return

        if self.training_active:
            self.root.after(10, self.run_training_epoch_step)

    def _update_main_ui(self, epoch, loss, val_loss):
        current_lr = self.optimizer.param_groups[0]['lr']
        self.lbl_telemetry.config(
            text=f"Epoch: {epoch}\nTrain Loss: {loss:.4f}\nVal Loss: {val_loss:.4f}\nLR: {current_lr:.5f}")
        rng = np.arange(len(self.history['loss']))
        self.line_loss.set_data(rng, self.history['loss'])
        self.line_val_loss.set_data(rng, self.history['val_loss'])
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        self.canvas_loss.draw_idle()

    def continuous_predict(self):
        img_features = self.process_image_for_cnn(self.image_predict)
        self.im_display.set_data(img_features[0])
        self.canvas_eval.draw_idle()

        if self.is_trained and hasattr(self, 'model') and self.predict_changed:
            self.predict_changed = False
            if np.sum(img_features) > 0.5:
                X_input = torch.tensor(img_features[np.newaxis, :, :, :], dtype=torch.float32)

                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(X_input)
                    probs = torch.softmax(outputs, dim=1).numpy()[0]

                conf_list = sorted(zip(self.classes_, probs), key=lambda x: x[1], reverse=True)
                self.lbl_shape.config(text=f"Final output: {conf_list[0][0].upper()}")

                self.ax_bar.clear()
                y_pos = np.arange(len(self.classes_))
                self.ax_bar.barh(y_pos, probs * 100, color=COLOR_BLUE)
                self.ax_bar.set_yticks(y_pos)
                self.ax_bar.set_yticklabels([c.capitalize() for c in self.classes_], color=TEXT_MAIN)
                self.ax_bar.set_xlim(0, 100)
                self.canvas_eval.draw_idle()

        self.root.after(40, self.continuous_predict)

    def open_analytics_dashboard(self):
        if not self.is_trained:
            messagebox.showwarning("Not Ready", "Train the model fully before analyzing.")
            return

        dash = tk.Toplevel(self.root)
        dash.title("Deep Telemetry & Confusion Matrix")
        dash.configure(bg=BG_MAIN)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_val_t)
            preds = torch.argmax(outputs, dim=1).numpy()

        cm = np.zeros((len(self.classes_), len(self.classes_)), dtype=int)
        for t, p in zip(self.y_val_t.numpy(), preds): cm[t, p] += 1

        fig = Figure(figsize=(8, 4), dpi=100, facecolor=BG_PANEL)

        ax1 = fig.add_subplot(121, facecolor=BG_MAIN)
        ax1.matshow(cm, cmap='Blues')
        ax1.set_title("Validation Confusion Matrix", color=TEXT_MAIN, pad=10)
        ax1.set_xticks(range(len(self.classes_)))
        ax1.set_yticks(range(len(self.classes_)))
        ax1.set_xticklabels(self.classes_, color=TEXT_MAIN, rotation=45)
        ax1.set_yticklabels(self.classes_, color=TEXT_MAIN)
        for i in range(len(self.classes_)):
            for j in range(len(self.classes_)):
                ax1.text(j, i, str(cm[i, j]), va='center', ha='center', color='red' if cm[i, j] == 0 else 'black')

        ax2 = fig.add_subplot(122, facecolor=BG_MAIN)
        class_accs = cm.diagonal() / np.where(cm.sum(axis=1) == 0, 1, cm.sum(axis=1))
        ax2.bar(self.classes_, class_accs * 100, color=COLOR_GREEN)
        ax2.set_title("Per-Class Accuracy (%)", color=TEXT_MAIN)
        ax2.set_ylim(0, 100)
        ax2.tick_params(colors=TEXT_MAIN)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=dash)
        canvas.get_tk_widget().pack(padx=10, pady=10, fill="both", expand=True)
        canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = CNNMonitorDashboard(root)
    root.mainloop()
import os
import time
import shutil
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageDraw, ImageOps

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
COLOR_PURPLE = "#b266ff"

FONT_TITLE = ("Helvetica", 11, "bold")
FONT_BODY = ("Helvetica", 10)
FONT_MONO = ("Consolas", 10, "bold")


# =======================================================
# NEURAL NETWORK ARCHITECTURE COMPONENT BLOCKS
# =======================================================
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss_CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = y_pred_clipped[range(samples), y_true]
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)


class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.forward(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


class Optimizer_Adam:
    def __init__(self, learning_rate=0.01, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * (layer.dweights ** 2)
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * (layer.dbiases ** 2)

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (
                np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (
                np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


# =======================================================
# DASHBOARD APPLICATION CORE INTERFACE
# =======================================================
class CNNMonitorDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Deep Learning Sandbox: Multi-Layer Architecture Diagnostics")
        self.root.configure(bg=BG_MAIN)
        self.root.geometry("1420x860")

        self.train_dir = "training_data"
        if not os.path.exists(self.train_dir): os.makedirs(self.train_dir)

        self.is_trained = False
        self.training_active = False
        self.predict_changed = False
        self.img_size = (28, 28)
        self.classes_ = []

        self.loss_history = []
        self.accuracy_history = []
        self.grad_l1_history = []
        self.grad_l2_history = []
        self.grad_l3_history = []

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
                                         font=FONT_TITLE, bd=1, relief="solid", padx=10, pady=10)
        self.frame_train.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.canvas_train = tk.Canvas(self.frame_train, width=self.canvas_size, height=self.canvas_size, bg='white',
                                      cursor='cross', highlightthickness=0)
        self.canvas_train.pack(pady=5)
        self.canvas_train.bind('<B1-Motion>', self.paint_train)
        self.image_train = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw_train = ImageDraw.Draw(self.image_train)

        tk.Label(self.frame_train, text="Target Vector Label:", bg=BG_PANEL, fg=TEXT_MUTED, font=FONT_BODY).pack()
        self.label_entry = tk.Entry(self.frame_train, bg=BG_INSIDE, fg=TEXT_MAIN, insertbackground=TEXT_MAIN,
                                    font=FONT_BODY, bd=1, relief="solid", justify="center")
        self.label_entry.pack(pady=5, ipady=3, fill="x")

        tk.Button(self.frame_train, text="Append Sample to Dataset", command=self.save_training_data, bg=BG_INSIDE,
                  fg=TEXT_MAIN, bd=0, font=FONT_BODY, pady=4).pack(fill="x", pady=2)
        tk.Button(self.frame_train, text="Reset Input Matrix", command=self.clear_train, bg=BG_INSIDE, fg=TEXT_MAIN,
                  bd=0, font=FONT_BODY, pady=4).pack(fill="x", pady=2)

        self.lbl_data_counts = tk.Label(self.frame_train, text="Dataset Allocation:\nNone", justify="left",
                                        bg=BG_INSIDE, fg=TEXT_MAIN, font=FONT_BODY)
        self.lbl_data_counts.pack(fill="x", pady=10, ipady=5)

        tk.Button(self.frame_train, text="Wipe Local Disk Storage", command=self.reset_data, bg="#2a1a1a", fg=COLOR_RED,
                  bd=0, font=FONT_BODY, pady=2).pack(fill="x", pady=2)
        self.btn_train = tk.Button(self.frame_train, text="START OPTIMIZATION LOOP", command=self.toggle_training,
                                   bg=COLOR_GREEN, fg=BG_MAIN, font=("Helvetica", 10, "bold"), bd=0, pady=8)
        self.btn_train.pack(fill="x", pady=10)

    def _build_evaluation_panel(self):
        self.frame_predict = tk.LabelFrame(self.root, text=" Real-Time Signal Evaluation ", bg=BG_PANEL, fg=TEXT_MAIN,
                                           font=FONT_TITLE, bd=1, relief="solid", padx=10, pady=10)
        self.frame_predict.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.canvas_predict = tk.Canvas(self.frame_predict, width=self.canvas_size, height=self.canvas_size, bg='white',
                                        cursor='cross', highlightthickness=0)
        self.canvas_predict.pack(pady=5)
        self.canvas_predict.bind('<B1-Motion>', self.paint_predict)
        self.image_predict = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw_predict = ImageDraw.Draw(self.image_predict)

        tk.Button(self.frame_predict, text="Reset Predict Matrix", command=self.clear_predict, bg=BG_INSIDE,
                  fg=TEXT_MAIN, bd=0, font=FONT_BODY, pady=4).pack(fill="x", pady=4)

        self.info_board = tk.Frame(self.frame_predict, bg="#0a0a0a", bd=0, relief="flat")
        self.info_board.pack(fill="x", pady=5)

        self.lbl_shape = tk.Label(self.info_board, text="ArgMax Output: None", bg="#0a0a0a", fg=COLOR_BLUE,
                                  font=("Consolas", 12, "bold"), width=35, anchor="w")
        self.lbl_shape.pack(anchor="w", padx=10, pady=6)

        # DIAGNOSTIC ADDTION: Subplot matrix layout for Matrix Input Viewer & Dynamic Softmax Bars
        self.fig_eval = Figure(figsize=(3.8, 2.3), dpi=95, facecolor=BG_PANEL)

        self.ax_img = self.fig_eval.add_subplot(121, facecolor=BG_MAIN)
        self.ax_img.set_title("Network Input (28x28)", fontsize=8, color=TEXT_MAIN)
        self.ax_img.axis('off')
        self.im_display = self.ax_img.imshow(np.zeros((28, 28)), cmap='gray', vmin=0, vmax=1)

        self.ax_bar = self.fig_eval.add_subplot(122, facecolor=BG_MAIN)
        self.ax_bar.set_title("Class Confidence", fontsize=8, color=TEXT_MAIN)
        self.ax_bar.tick_params(colors=TEXT_MUTED, labelsize=7)
        for spine in self.ax_bar.spines.values(): spine.set_color("#2d2d2d")

        self.canvas_eval = FigureCanvasTkAgg(self.fig_eval, master=self.frame_predict)
        self.canvas_eval.get_tk_widget().pack(fill="x", expand=True, pady=5)

        self.btn_inspect = tk.Button(self.frame_predict, text="Isolate Layer-1 Weights Map",
                                     command=self.open_feature_inspector, state="disabled", bg=COLOR_BLUE, fg=BG_MAIN,
                                     font=("Helvetica", 10, "bold"), bd=0, pady=6)
        self.btn_inspect.pack(fill="x", pady=8)

    def _build_telemetry_panel(self):
        self.frame_monitor = tk.LabelFrame(self.root, text=" Extended Parameter Tuner & Telemetry ", bg=BG_PANEL,
                                           fg=TEXT_MAIN, font=FONT_TITLE, bd=1, relief="solid", padx=10, pady=10)
        self.frame_monitor.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        self.frame_monitor.grid_columnconfigure(0, weight=1)
        self.frame_monitor.grid_columnconfigure(1, weight=1)

        self.top_monitor_panel = tk.Frame(self.frame_monitor, bg=BG_PANEL)
        self.top_monitor_panel.grid(row=0, column=0, columnspan=2, sticky="ew")

        self.tuner_panel = tk.Frame(self.top_monitor_panel, bg=BG_INSIDE, padx=5, pady=5)
        self.tuner_panel.pack(side="left", fill="x", expand=True, padx=(0, 5))
        tk.Label(self.tuner_panel, text="Learning Rate (α)", bg=BG_INSIDE, fg=TEXT_MAIN, font=FONT_BODY).grid(row=0,
                                                                                                              column=0,
                                                                                                              sticky="w")
        self.slider_lr = tk.Scale(self.tuner_panel, from_=0.001, to=0.2, resolution=0.001, orient="horizontal",
                                  bg=BG_INSIDE, fg=TEXT_MAIN, highlightthickness=0, troughcolor=BG_PANEL)
        self.slider_lr.set(0.01)
        self.slider_lr.grid(row=0, column=1, sticky="we")
        tk.Label(self.tuner_panel, text="Target Epoch Limits", bg=BG_INSIDE, fg=TEXT_MAIN, font=FONT_BODY).grid(row=1,
                                                                                                                column=0,
                                                                                                                sticky="w")
        self.entry_epochs = tk.Entry(self.tuner_panel, width=6, bg=BG_MAIN, fg=TEXT_MAIN, bd=1, justify="center")
        self.entry_epochs.insert(0, "300")
        self.entry_epochs.grid(row=1, column=1, sticky="w", pady=5)

        self.lbl_telemetry = tk.Label(self.top_monitor_panel, text="Engine Diagnostics:\nWaiting...", bg="#0a0a0a",
                                      fg=TEXT_MAIN, justify="left", font=FONT_MONO, padx=10)
        self.lbl_telemetry.pack(side="right", fill="both", expand=True)

        self._init_charts()

    def _init_charts(self):
        self.fig_loss = Figure(figsize=(3.5, 2.2), dpi=90, facecolor=BG_PANEL)
        self.ax_loss = self.fig_loss.add_subplot(111, facecolor=BG_MAIN)
        self._format_ax(self.ax_loss, "Convergence Curve", "Epochs", "Magnitude")
        self.line_loss, = self.ax_loss.plot([], [], color=COLOR_BLUE, label='Loss')
        self.line_acc, = self.ax_loss.plot([], [], color=COLOR_YELLOW, linestyle="--", label='Accuracy')
        self.ax_loss.legend(fontsize=6, loc='upper right', facecolor=BG_PANEL, edgecolor=BG_MAIN)
        self.canvas_loss = FigureCanvasTkAgg(self.fig_loss, master=self.frame_monitor)
        self.canvas_loss.get_tk_widget().grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.fig_grad = Figure(figsize=(3.5, 2.2), dpi=90, facecolor=BG_PANEL)
        self.ax_grad = self.fig_grad.add_subplot(111, facecolor=BG_MAIN)
        self._format_ax(self.ax_grad, "Gradient Flow (L2 Norm)", "Epochs", "||dW||")
        self.line_g1, = self.ax_grad.plot([], [], color=COLOR_RED, alpha=0.8, label='L1')
        self.line_g2, = self.ax_grad.plot([], [], color=COLOR_PURPLE, alpha=0.8, label='L2')
        self.line_g3, = self.ax_grad.plot([], [], color=COLOR_GREEN, alpha=0.8, label='L3')
        self.ax_grad.legend(fontsize=6, loc='upper right', facecolor=BG_PANEL, edgecolor=BG_MAIN)
        self.canvas_grad = FigureCanvasTkAgg(self.fig_grad, master=self.frame_monitor)
        self.canvas_grad.get_tk_widget().grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        self.fig_hist = Figure(figsize=(3.5, 2.2), dpi=90, facecolor=BG_PANEL)
        self.ax_hist = self.fig_hist.add_subplot(111, facecolor=BG_MAIN)
        self._format_ax(self.ax_hist, "Weight Distribution (Layer 1)", "Value", "Density")
        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, master=self.frame_monitor)
        self.canvas_hist.get_tk_widget().grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

    def _format_ax(self, ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=10, color=TEXT_MAIN, pad=5)
        ax.set_xlabel(xlabel, fontsize=8, color=TEXT_MUTED)
        ax.set_ylabel(ylabel, fontsize=8, color=TEXT_MUTED)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        for spine in ax.spines.values(): spine.set_color("#2d2d2d")
        ax.grid(True, linestyle='--', color="#2d2d2d", alpha=0.4)

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
        self.lbl_shape.config(text="ArgMax Output: None")
        self.im_display.set_data(np.zeros((28, 28)))
        self.ax_bar.clear()
        self.ax_bar.set_title("Class Confidence", fontsize=8, color=TEXT_MAIN)
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
        if total == 0:
            self.lbl_data_counts.config(text="Dataset Allocation:\nEmpty")
            self.classes_ = []
        else:
            text = "Dataset Allocation:\n"
            self.classes_ = list(counts.keys())
            for label, c in counts.items(): text += f" - {label.capitalize()}: {c}\n"
            self.lbl_data_counts.config(text=text + f"Total: {total}")

    def reset_data(self):
        if messagebox.askyesno("Confirm", "Purge database?"):
            self.training_active = False
            shutil.rmtree(self.train_dir)
            os.makedirs(self.train_dir)
            self._reset_engine_state()
            self.update_data_counts()
            self.clear_train()
            self.clear_predict()

    def save_training_data(self):
        shape_label = self.label_entry.get().strip().lower()
        if not shape_label: return
        ld = os.path.join(self.train_dir, shape_label)
        if not os.path.exists(ld): os.makedirs(ld)
        fp = os.path.join(ld, f"{int(time.time() * 1000)}.png")
        self.image_train.resize(self.img_size).save(fp)
        self.update_data_counts()
        self.clear_train()

    def process_image_for_cnn(self, img):
        img = ImageOps.invert(img).resize(self.img_size)
        return np.array(img, dtype=np.float32)[np.newaxis, :, :] / 255.0

    def _reset_engine_state(self):
        self.is_trained = False
        self.loss_history.clear()
        self.accuracy_history.clear()
        self.grad_l1_history.clear()
        self.grad_l2_history.clear()
        self.grad_l3_history.clear()

        self.ax_loss.clear()
        self._format_ax(self.ax_loss, "Convergence Curve", "Epochs", "Magnitude")
        self.line_loss, = self.ax_loss.plot([], [], color=COLOR_BLUE, label='Loss')
        self.line_acc, = self.ax_loss.plot([], [], color=COLOR_YELLOW, linestyle="--", label='Accuracy')
        self.ax_loss.legend(fontsize=6, loc='upper right', facecolor=BG_PANEL, edgecolor=BG_MAIN)

        self.ax_grad.clear()
        self._format_ax(self.ax_grad, "Gradient Flow (L2 Norm)", "Epochs", "||dW||")
        self.line_g1, = self.ax_grad.plot([], [], color=COLOR_RED, alpha=0.8, label='L1')
        self.line_g2, = self.ax_grad.plot([], [], color=COLOR_PURPLE, alpha=0.8, label='L2')
        self.line_g3, = self.ax_grad.plot([], [], color=COLOR_GREEN, alpha=0.8, label='L3')
        self.ax_grad.legend(fontsize=6, loc='upper right', facecolor=BG_PANEL, edgecolor=BG_MAIN)

        self.ax_hist.clear()
        self._format_ax(self.ax_hist, "Weight Distribution", "Value", "Density")

        self.canvas_loss.draw()
        self.canvas_grad.draw()
        self.canvas_hist.draw()
        self.btn_inspect.config(state="disabled")
        self.btn_train.config(text="START OPTIMIZATION LOOP", bg=COLOR_GREEN, state="normal")
        if hasattr(self, 'dense1'):
            del self.dense1, self.dense2, self.dense3

    def toggle_training(self):
        if self.training_active:
            self.training_active = False
            self.btn_train.config(text="RESUME OPTIMIZATION", bg=COLOR_GREEN)
        else:
            self.start_training_pipeline()

    def start_training_pipeline(self):
        self.update_data_counts()
        if len(self.classes_) < 2:
            messagebox.showerror("Error", "Need >= 2 classes.")
            return

        try:
            self.target_epochs = int(self.entry_epochs.get().strip())
        except ValueError:
            return

        if len(self.loss_history) >= self.target_epochs:
            self._reset_engine_state()

        X_raw, y_raw = [], []
        for label_idx, label_name in enumerate(self.classes_):
            ld = os.path.join(self.train_dir, label_name)
            for f in os.listdir(ld):
                if f.endswith(".png"):
                    img = Image.open(os.path.join(ld, f)).convert('L')
                    X_raw.append(self.process_image_for_cnn(img))
                    y_raw.append(label_idx)

        indices = np.arange(len(X_raw))
        np.random.shuffle(indices)

        self.X_train = np.array(X_raw, dtype=np.float32)[indices]
        self.y_train = np.array(y_raw, dtype=np.int32)[indices]
        self.X_train = self.X_train.reshape(len(X_raw), -1)

        if not hasattr(self, 'dense1'):
            self.dense1 = Layer_Dense(784, 32)
            self.activation1 = Activation_ReLU()
            self.dense2 = Layer_Dense(32, 16)
            self.activation2 = Activation_ReLU()
            self.dense3 = Layer_Dense(16, len(self.classes_))

            self.loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
            self.optimizer = Optimizer_Adam(learning_rate=self.slider_lr.get())

        self.training_active = True
        self.btn_train.config(text="PAUSE OPTIMIZATION", bg=COLOR_RED)
        self.run_training_epoch_step()

    def run_training_epoch_step(self):
        if not self.training_active: return

        epochs_chunk = 10
        if len(self.loss_history) >= self.target_epochs:
            self.training_active = False
            self.is_trained = True
            self.btn_train.config(text="OPTIMIZATION COMPLETE", bg=COLOR_BLUE)
            self.btn_inspect.config(state="normal")
            return

        for _ in range(epochs_chunk):
            current_epoch_idx = len(self.loss_history)
            if current_epoch_idx >= self.target_epochs:
                break

            self.optimizer.learning_rate = self.slider_lr.get()

            # Forward pass
            self.dense1.forward(self.X_train)
            self.activation1.forward(self.dense1.output)
            self.dense2.forward(self.activation1.output)
            self.activation2.forward(self.dense2.output)
            self.dense3.forward(self.activation2.output)

            loss = self.loss_activation.forward(self.dense3.output, self.y_train)
            acc = np.mean(np.argmax(self.loss_activation.output, axis=1) == self.y_train)

            # Backward pass
            self.loss_activation.backward(self.loss_activation.output, self.y_train)
            self.dense3.backward(self.loss_activation.dinputs)
            self.activation2.backward(self.dense3.dinputs)
            self.dense2.backward(self.activation2.dinputs)
            self.activation1.backward(self.dense2.dinputs)
            self.dense1.backward(self.activation1.dinputs)

            # Optimize
            self.optimizer.pre_update_params()
            self.optimizer.update_params(self.dense1)
            self.optimizer.update_params(self.dense2)
            self.optimizer.update_params(self.dense3)
            self.optimizer.post_update_params()

            self.loss_history.append(loss)
            self.accuracy_history.append(acc)
            self.grad_l1_history.append(np.linalg.norm(self.dense1.dweights))
            self.grad_l2_history.append(np.linalg.norm(self.dense2.dweights))
            self.grad_l3_history.append(np.linalg.norm(self.dense3.dweights))

        if len(self.loss_history) >= self.target_epochs:
            self.training_active = False
            self.is_trained = True
            self.btn_train.config(text="OPTIMIZATION COMPLETE", bg=COLOR_BLUE)
            self.btn_inspect.config(state="normal")

        last_recorded_idx = len(self.loss_history) - 1
        if last_recorded_idx >= 0:
            self._update_telemetry_ui(last_recorded_idx, self.loss_history[-1], self.accuracy_history[-1])

        if self.training_active:
            self.root.after(10, self.run_training_epoch_step)

    def _update_telemetry_ui(self, epoch, loss, acc):
        self.lbl_telemetry.config(
            text=f"Epoch: {epoch + 1}/{self.target_epochs}\nLoss: {loss:.6f}\nAccuracy : {acc * 100:.1f}%\nLearning Rate: {self.optimizer.current_learning_rate:.4f}")

        epochs_range = np.arange(len(self.loss_history))

        self.line_loss.set_data(epochs_range, self.loss_history)
        self.line_acc.set_data(epochs_range, self.accuracy_history)
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        self.canvas_loss.draw_idle()

        self.line_g1.set_data(epochs_range, self.grad_l1_history)
        self.line_g2.set_data(epochs_range, self.grad_l2_history)
        self.line_g3.set_data(epochs_range, self.grad_l3_history)
        self.ax_grad.relim()
        self.ax_grad.autoscale_view()
        self.canvas_grad.draw_idle()

        self.ax_hist.clear()
        self._format_ax(self.ax_hist, "Weight Distribution (Layer 1)", "Value", "Density")
        if hasattr(self, 'dense1'):
            self.ax_hist.hist(self.dense1.weights.flatten(), bins=50, color=COLOR_BLUE, alpha=0.7, density=True)
        self.canvas_hist.draw_idle()

    def continuous_predict(self):
        img_features = self.process_image_for_cnn(self.image_predict)

        # DIAGNOSTIC UPDATE: Live update the 28x28 grayscale heatmap so the user sees what the model receives
        self.im_display.set_data(img_features[0])
        self.canvas_eval.draw_idle()

        if self.is_trained and hasattr(self, 'dense1') and self.predict_changed:
            self.predict_changed = False

            if np.sum(img_features) > 0.5:
                X_input = img_features.reshape(1, -1)

                self.dense1.forward(X_input)
                self.activation1.forward(self.dense1.output)
                self.dense2.forward(self.activation1.output)
                self.activation2.forward(self.dense2.output)
                self.dense3.forward(self.activation2.output)
                self.loss_activation.activation.forward(self.dense3.output)

                probs = self.loss_activation.activation.output[0]
                conf_list = sorted(zip(self.classes_, probs), key=lambda x: x[1], reverse=True)

                self.lbl_shape.config(text=f"ArgMax Output: {conf_list[0][0].upper()}")

                # DIAGNOSTIC UPDATE: Replot horizontal bars matching current probability states
                self.ax_bar.clear()
                self.ax_bar.set_title("Class Confidence", fontsize=8, color=TEXT_MAIN)
                self.ax_bar.tick_params(colors=TEXT_MUTED, labelsize=7)
                for spine in self.ax_bar.spines.values(): spine.set_color("#2d2d2d")

                y_pos = np.arange(len(self.classes_))
                self.ax_bar.barh(y_pos, probs * 100, color=COLOR_BLUE, height=0.5)
                self.ax_bar.set_yticks(y_pos)
                self.ax_bar.set_yticklabels([c.capitalize() for c in self.classes_], color=TEXT_MAIN)
                self.ax_bar.set_xlim(0, 100)
                self.ax_bar.grid(True, linestyle='--', color="#2d2d2d", alpha=0.3)
                self.canvas_eval.draw_idle()

        self.root.after(40, self.continuous_predict)

    def open_feature_inspector(self):
        inspect_window = tk.Toplevel(self.root)
        inspect_window.title("Layer 1 Weight Grid Matrix Maps")
        inspect_window.configure(bg=BG_MAIN)

        fig = Figure(figsize=(6, 6), dpi=100, facecolor=BG_PANEL)
        for i in range(16):
            ax = fig.add_subplot(4, 4, i + 1)
            neuron_weights = self.dense1.weights[:, i].reshape(28, 28)
            ax.imshow(neuron_weights, cmap='plasma')
            ax.axis('off')

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=inspect_window)
        canvas.get_tk_widget().pack(padx=10, pady=10)
        canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = CNNMonitorDashboard(root)
    root.mainloop()
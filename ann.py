import os
import time
import shutil
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps

# Import PyTorch components
import torch
import torch.nn as nn
import torch.optim as optim

# Import pure Figure objects for crash-proof embedded charts
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =======================================================
# 3BLUE1BROWN DESIGN SYSTEM CONSTANTS
# =======================================================
BG_MAIN = "#111111"        # Deep matte black canvas background
BG_PANEL = "#1a1a1a"       # Sleek dark-charcoal component frame
BG_INSIDE = "#222222"      # Inset field tone
TEXT_MAIN = "#ececec"      # Crisp glowing white text
TEXT_MUTED = "#888888"     # Subdued math coordinate grey
COLOR_BLUE = "#58c4dd"     # Signature 3B1B vector cyan
COLOR_GREEN = "#26ceaa"    # Signature 3B1B validation emerald
COLOR_RED = "#e07a5f"      # Soft error-trace crimson

FONT_TITLE = ("Helvetica", 11, "bold")
FONT_BODY = ("Helvetica", 10)
FONT_MONO = ("Consolas", 10, "bold")


# ==========================================
# 1. CNN ARCHITECTURE (Functions Untouched)
# ==========================================
class ShapeCNN(nn.Module):
    def __init__(self, num_classes):
        super(ShapeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_layer1_activations(self, x):
        return self.relu1(self.conv1(x))


# ==========================================
# 2. DASHBOARD APPLICATION
# ==========================================
class CNNMonitorDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("3Blue1Brown Modern Deep Learning Dashboard")
        self.root.configure(bg=BG_MAIN)
        
        # Hardware Engine Detection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # --- ML Properties ---
        self.train_dir = "training_data"
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
            
        self.model = None
        self.is_trained = False
        self.img_size = (28, 28)
        self.classes_ = []
        self.loss_history = []
        
        self.canvas_size = 230
        
        # =======================================================
        # COLUMN 0: TRAINING BOARD (3B1B Skin)
        # =======================================================
        self.frame_train = tk.LabelFrame(
            self.root, text=" Neural Input Network Builder ", 
            bg=BG_PANEL, fg=TEXT_MAIN, font=FONT_TITLE, bd=1, relief="solid", padx=12, pady=12
        )
        self.frame_train.grid(row=0, column=0, padx=15, pady=15, sticky="n")
        
        self.canvas_train = tk.Canvas(
            self.frame_train, width=self.canvas_size, height=self.canvas_size, 
            bg='white', cursor='cross', highlightbackground=BG_PANEL, highlightthickness=0
        )
        self.canvas_train.pack(pady=8)
        self.canvas_train.bind('<B1-Motion>', self.paint_train)
        
        self.image_train = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw_train = ImageDraw.Draw(self.image_train)
        
        tk.Label(self.frame_train, text="Target Vector Label:", bg=BG_PANEL, fg=TEXT_MUTED, font=FONT_BODY).pack()
        self.label_entry = tk.Entry(
            self.frame_train, bg=BG_INSIDE, fg=TEXT_MAIN, insertbackground=TEXT_MAIN,
            font=FONT_BODY, bd=1, relief="solid", justify="center"
        )
        self.label_entry.pack(pady=5, ipady=3, fill="x")
        
        tk.Button(
            self.frame_train, text="Append Sample to Dataset", command=self.save_training_data,
            bg=BG_INSIDE, fg=TEXT_MAIN, activebackground="#333333", activeforeground=TEXT_MAIN,
            bd=0, cursor="hand2", font=FONT_BODY, pady=4
        ).pack(fill="x", pady=2)
        
        tk.Button(
            self.frame_train, text="Reset Input Matrix", command=self.clear_train,
            bg=BG_INSIDE, fg=TEXT_MAIN, activebackground="#333333", activeforeground=TEXT_MAIN,
            bd=0, cursor="hand2", font=FONT_BODY, pady=4
        ).pack(fill="x", pady=2)
        
        self.data_panel = tk.Frame(self.frame_train, bg=BG_INSIDE, bd=0, relief="flat")
        self.data_panel.pack(fill="x", pady=10)
        self.lbl_data_counts = tk.Label(
            self.data_panel, text="Dataset Allocation:\nNone", 
            justify="left", bg=BG_INSIDE, fg=TEXT_MAIN, font=FONT_BODY
        )
        self.lbl_data_counts.pack(anchor="w", padx=8, pady=8)
        
        tk.Button(
            self.frame_train, text="Wipe Local Disk Storage", command=self.reset_data, 
            bg="#2a1a1a", fg=COLOR_RED, activebackground="#3a1a1a", activeforeground=COLOR_RED,
            bd=0, cursor="hand2", font=FONT_BODY, pady=2
        ).pack(fill="x", pady=2)
        
        tk.Button(
            self.frame_train, text="OPTIMIZE & TRAIN MODEL", command=self.train_model, 
            bg=COLOR_GREEN, fg=BG_MAIN, activebackground="#1eb495", activeforeground=BG_MAIN,
            font=("Helvetica", 10, "bold"), bd=0, cursor="hand2", pady=8
        ).pack(fill="x", pady=12)

        # =======================================================
        # COLUMN 1: LIVE EVALUATION ENGINE (3B1B Skin)
        # =======================================================
        self.frame_predict = tk.LabelFrame(
            self.root, text=" Real-Time Signal Evaluation ", 
            bg=BG_PANEL, fg=TEXT_MAIN, font=FONT_TITLE, bd=1, relief="solid", padx=12, pady=12
        )
        self.frame_predict.grid(row=0, column=1, padx=15, pady=15, sticky="n")
        
        self.canvas_predict = tk.Canvas(
            self.frame_predict, width=self.canvas_size, height=self.canvas_size, 
            bg='white', cursor='cross', highlightbackground=BG_PANEL, highlightthickness=0
        )
        self.canvas_predict.pack(pady=8)
        self.canvas_predict.bind('<B1-Motion>', self.paint_predict)
        
        self.image_predict = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw_predict = ImageDraw.Draw(self.image_predict)
        
        tk.Button(
            self.frame_predict, text="Reset Predict Matrix", command=self.clear_predict,
            bg=BG_INSIDE, fg=TEXT_MAIN, activebackground="#333333", activeforeground=TEXT_MAIN,
            bd=0, cursor="hand2", font=FONT_BODY, pady=4
        ).pack(fill="x", pady=4)
        
        self.info_board = tk.Frame(self.frame_predict, bg="#0a0a0a", bd=0, relief="flat")
        self.info_board.pack(fill="x", pady=5)
        
        self.lbl_shape = tk.Label(
            self.info_board, text="ArgMax Output: None", 
            bg="#0a0a0a", fg=COLOR_BLUE, font=("Consolas", 12, "bold")
        )
        self.lbl_shape.pack(anchor="w", padx=12, pady=6)
        
        self.lbl_conf = tk.Label(
            self.info_board, text="Softmax Distributions:\nUntrained Matrix", 
            bg="#0a0a0a", fg=TEXT_MAIN, font=FONT_MONO, justify="left"
        )
        self.lbl_conf.pack(anchor="w", padx=12, pady=6)
        
        self.btn_inspect = tk.Button(
            self.frame_predict, text="Isolate Conv2d Filter Activations", command=self.open_feature_inspector, 
            state="disabled", bg=COLOR_BLUE, fg=BG_MAIN, activebackground="#47afc7", activeforeground=BG_MAIN,
            font=("Helvetica", 10, "bold"), bd=0, cursor="hand2", pady=6
        )
        self.btn_inspect.pack(fill="x", pady=8)

        # =======================================================
        # COLUMN 2: TELEMETRY & DARK MATH CHART (3B1B Skin)
        # =======================================================
        self.frame_monitor = tk.LabelFrame(
            self.root, text=" Mathematical Graph Analysis ", 
            bg=BG_PANEL, fg=TEXT_MAIN, font=FONT_TITLE, bd=1, relief="solid", padx=12, pady=12
        )
        self.frame_monitor.grid(row=0, column=2, padx=15, pady=15, sticky="n")
        
        self.telemetry_panel = tk.Frame(self.frame_monitor, bg=BG_INSIDE, bd=0, relief="flat")
        self.telemetry_panel.pack(fill="x", pady=5)
        
        param_text = f"Compute Target: {str(self.device).upper()}\nTotal Param Count: 0\nLast Optimization Delta: N/A\nExecution Time: N/A"
        self.lbl_telemetry = tk.Label(
            self.telemetry_panel, text=param_text, bg=BG_INSIDE, fg=TEXT_MAIN, 
            justify="left", font=FONT_MONO, padx=10, pady=10
        )
        self.lbl_telemetry.pack(anchor="w")
        
        # Custom 3B1B Themed Dark Graph Config
        self.fig = Figure(figsize=(3.4, 2.7), dpi=100, facecolor=BG_PANEL)
        self.ax = self.fig.add_subplot(111, facecolor=BG_MAIN)
        self.ax.set_title("Loss Convergence Curve", fontsize=10, color=TEXT_MAIN, pad=10)
        self.ax.set_xlabel("Epochs", fontsize=8, color=TEXT_MUTED)
        self.ax.set_ylabel("CrossEntropy", fontsize=8, color=TEXT_MUTED)
        self.ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color("#2d2d2d")
        self.fig.tight_layout()
        
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=self.frame_monitor)
        self.chart_canvas.get_tk_widget().pack(pady=8)

        # Bottom Global Status Strip
        self.status_label = tk.Label(
            root, text="System Core initialized successfully.", 
            bd=0, relief="flat", anchor="w", bg="#0a0a0a", fg=TEXT_MUTED, font=FONT_BODY, padx=10, pady=5
        )
        self.status_label.grid(row=1, column=0, columnspan=3, sticky="we")

        self.update_data_counts()
        self.root.after(10, self.continuous_predict)

    # --- Drawing Routines ---
    def paint_train(self, event): self._paint(event, self.canvas_train, self.draw_train)
    def paint_predict(self, event): self._paint(event, self.canvas_predict, self.draw_predict)
    def _paint(self, event, canvas, draw_obj):
        r = 7
        canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r, fill='black', outline='black')
        draw_obj.ellipse([event.x-r, event.y-r, event.x+r, event.y+r], fill='black')

    def clear_train(self):
        self.canvas_train.delete("all")
        self.image_train = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw_train = ImageDraw.Draw(self.image_train)

    def clear_predict(self):
        self.canvas_predict.delete("all")
        self.image_predict = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw_predict = ImageDraw.Draw(self.image_predict)

    def update_data_counts(self):
        counts = {}
        total = 0
        if os.path.exists(self.train_dir):
            for label in sorted(os.listdir(self.train_dir)):
                label_dir = os.path.join(self.train_dir, label)
                if os.path.isdir(label_dir):
                    c = len([f for f in os.listdir(label_dir) if f.endswith('.png')])
                    if c > 0: counts[label] = c; total += c
        if total == 0:
            self.lbl_data_counts.config(text="Dataset Allocation:\nEmpty Folder Topology")
            self.classes_ = []
        else:
            text = "Dataset Allocation:\n"
            self.classes_ = list(counts.keys())
            for label, c in counts.items(): text += f" - {label.capitalize()}: {c}\n"
            text += f"Total Components: {total}"
            self.lbl_data_counts.config(text=text)

    def reset_data(self):
        if messagebox.askyesno("Destructive Step", "Purge database directories?"):
            shutil.rmtree(self.train_dir); os.makedirs(self.train_dir)
            self.is_trained = False; self.model = None; self.loss_history = []
            self.update_data_counts()
            self.clear_train(); self.clear_predict()
            
            self.ax.clear()
            self.ax.set_facecolor(BG_MAIN)
            self.ax.set_title("Loss Convergence Curve", fontsize=10, color=TEXT_MAIN, pad=10)
            self.ax.set_xlabel("Epochs", fontsize=8, color=TEXT_MUTED)
            self.ax.set_ylabel("CrossEntropy", fontsize=8, color=TEXT_MUTED)
            self.ax.tick_params(colors=TEXT_MUTED, labelsize=8)
            for spine in self.ax.spines.values(): spine.set_color("#2d2d2d")
            self.chart_canvas.draw()
            
            self.btn_inspect.config(state="disabled")
            self.status_label.config(text="Local dataset wiped completely.")

    def save_training_data(self):
        shape_label = self.label_entry.get().strip().lower()
        if not shape_label:
            messagebox.showwarning("Incomplete Data", "Input a label directory name first!")
            return
        ld = os.path.join(self.train_dir, shape_label)
        if not os.path.exists(ld): os.makedirs(ld)
        fp = os.path.join(ld, f"{int(time.time() * 1000)}.png")
        self.image_train.resize(self.img_size).save(fp)
        self.update_data_counts()
        self.clear_train()

    def process_image_for_cnn(self, img):
        img = ImageOps.invert(img).resize(self.img_size)
        return np.array(img, dtype=np.float32)[np.newaxis, :, :] / 255.0

    # =======================================================
    # TRAINING CONTROLLER & METRICS
    # =======================================================
    def train_model(self):
        X_raw, y_raw = [], []
        self.update_data_counts()
        if len(self.classes_) < 2:
            messagebox.showerror("Graph Error", "Categorical dimensions must be >= 2 classes.")
            return
            
        for label_idx, label_name in enumerate(self.classes_):
            ld = os.path.join(self.train_dir, label_name)
            for f in os.listdir(ld):
                if f.endswith(".png"):
                    img = Image.open(os.path.join(ld, f)).convert('L')
                    X_raw.append(self.process_image_for_cnn(img))
                    y_raw.append(label_idx)
                    
        X_train = torch.tensor(np.array(X_raw), dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_raw, dtype=torch.long).to(self.device)
        
        self.model = ShapeCNN(num_classes=len(self.classes_)).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        
        self.loss_history = []
        start_time = time.time()
        
        self.model.train()
        for epoch in range(120):
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())
            
        elapsed_time = time.time() - start_time
        self.is_trained = True
        self.btn_inspect.config(state="normal")
        
        telemetry_text = (
            f"Compute Target: {str(self.device).upper()}\n"
            f"Total Param Count: {total_params}\n"
            f"Last Optimization Delta: {self.loss_history[-1]:.5f}\n"
            f"Execution Time: {elapsed_time:.3f}s"
        )
        self.lbl_telemetry.config(text=telemetry_text)
        
        # Re-render Graph into 3B1B color scheme
        self.ax.clear()
        self.ax.set_facecolor(BG_MAIN)
        self.ax.plot(self.loss_history, color=COLOR_BLUE, linewidth=2, label='Loss Trace')
        self.ax.set_title("Loss Convergence Curve", fontsize=10, color=TEXT_MAIN, pad=10)
        self.ax.set_xlabel("Epochs", fontsize=8, color=TEXT_MUTED)
        self.ax.set_ylabel("CrossEntropy", fontsize=8, color=TEXT_MUTED)
        self.ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        self.ax.grid(True, linestyle='--', color="#2d2d2d", alpha=0.6)
        for spine in self.ax.spines.values(): spine.set_color("#2d2d2d")
        self.fig.tight_layout()
        self.chart_canvas.draw()
        
        self.status_label.config(text="CNN parameters converged successfully.")

    # =======================================================
    # REAL-TIME EVALUATION LOOP & INSPECTION PORT
    # =======================================================
    def continuous_predict(self):
        if self.is_trained and self.model is not None:
            img_features = self.process_image_for_cnn(self.image_predict)
            if np.sum(img_features) > 0.5:
                self.model.eval()
                with torch.no_grad():
                    input_tensor = torch.tensor(img_features, dtype=torch.float32).unsqueeze(0).to(self.device)
                    logits = self.model(input_tensor)
                    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                conf_list = list(zip(self.classes_, probabilities))
                conf_list.sort(key=lambda x: x[1], reverse=True)
                
                self.lbl_shape.config(text=f"ArgMax Output: {conf_list[0][0].upper()}")
                conf_text = "Softmax Distributions:\n"
                for s_name, prob in conf_list:
                    conf_text += f" - {s_name.capitalize()}: {prob*100:.1f}%\n"
                self.lbl_conf.config(text=conf_text)
            else:
                self.lbl_shape.config(text="ArgMax Output: None")
                self.lbl_conf.config(text="Softmax Distributions:\nWaiting for input signals...")
                
        self.root.after(10, self.continuous_predict)

    def open_feature_inspector(self):
        img_features = self.process_image_for_cnn(self.image_predict)
        if np.sum(img_features) <= 0.5:
            messagebox.showinfo("Inspector Info", "Draw something on the Prediction Board first!")
            return
            
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(img_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            activations = self.model.get_layer1_activations(input_tensor).cpu().squeeze(0).numpy()
            
        inspect_window = tk.Toplevel(self.root)
        inspect_window.title("Layer 1 Conv2d Active Map Topology")
        inspect_window.configure(bg=BG_MAIN)
        
        # Clean 3B1B Multi-subplot grid representation
        fig = Figure(figsize=(5, 5), dpi=100, facecolor=BG_PANEL)
        
        for i in range(16):
            ax = fig.add_subplot(4, 4, i + 1)
            # 'plasma' closely mirrors the beautiful neon-gradient style of weight mapping matrices
            ax.imshow(activations[i], cmap='plasma')
            ax.axis('off')
            
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=inspect_window)
        canvas.get_tk_widget().pack(padx=10, pady=10)
        canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = CNNMonitorDashboard(root)
    root.mainloop()
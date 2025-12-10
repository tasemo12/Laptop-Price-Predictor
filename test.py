import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Constants
GPU_SCORES = {"RTX4090":100,"RTX4080":90,"RTX4070TI":82,"RTX4070":75,"RTX4060":65,"RTX3080":70,"RTX3070":62,"RTX3060TI":58,"RTX3060":52,"RTX3050TI":45,"RTX3050":40,"RTX2070":55,"GTX1660TI":38,"GTX1660":35,"GTX1650":28,"GTX1050TI":22,"INTEGRATED":10}
CPU_SCORES = {"I9":100,"RYZEN 9":95,"I7":80,"RYZEN 7":75,"I5":60,"RYZEN 5":58,"I3":40,"RYZEN 3":38}
BRAND_SCORES = {"ALIENWARE":10,"RAZER":9,"MSI":7,"ASUS":7,"HP":6,"DELL":6,"LENOVO":6,"SAMSUNG":5,"GIGABYTE":5,"ACER":4}
SCREEN_MAP = {"720P":1,"1080P":2,"1440P":3,"4K":4,"8K":5}
RAM_MAP = {"DDR3":1,"DDR4":2,"DDR5":3}
COLORS = {"bg":"#ffffff","card":"#f8f9fa","accent":"#2c3e50","text":"#212529","border":"#dee2e6"}

# Scoring functions
score_gpu = lambda g: next((v for k,v in GPU_SCORES.items() if k in str(g).upper().replace(" ","")), 15)
score_cpu = lambda c: (lambda s: next((v for k,v in CPU_SCORES.items() if k in s), 30) + (next((i-5)*2 for i in range(13,5,-1) if str(i) in s) if any(str(i) in s for i in range(13,5,-1)) else 0))(str(c).upper())
extract_num = lambda v: (lambda n: n * (1024 if "TB" in str(v).upper() else 1))(float(str(v).upper().replace("GB","").replace("TB","").strip())) if v else np.nan

# Load & prepare data
df = pd.read_csv("laptop_dataset_5000.csv")
df = df[pd.to_numeric(df["Price"], errors="coerce").between(300, 5000)].dropna(subset=["Price"])
df[["RAM","Storage"]] = df[["RAM","Storage"]].applymap(extract_num)
df = df.dropna(subset=["RAM","Storage"])
df["Screen_Quality"] = df["Screen"].str.upper().map(SCREEN_MAP).dropna()
df = df.dropna(subset=["Screen_Quality"])
df[["GPU_Score","CPU_Score"]] = df[["GPU","CPU"]].apply(lambda col: col.map(score_gpu if col.name=="GPU" else score_cpu))
df["RAM_Type_Score"] = df["RAM_Type"].map(RAM_MAP).dropna()
df = df.dropna(subset=["RAM_Type_Score"])
df["Brand_Score"] = df["Brand"].str.upper().map(BRAND_SCORES).fillna(5)
df["RAM_GPU"] = df["RAM"] * df["GPU_Score"] / 100
df["CPU_GPU"] = df["CPU_Score"] * df["GPU_Score"] / 100
df["Total_Perf"] = (df["CPU_Score"] + df["GPU_Score"]) / 2

# Train BOOST model ONLY
X = df[["CPU_Score","RAM","RAM_Type_Score","GPU_Score","Storage","Screen_Quality","Brand_Score","RAM_GPU","CPU_GPU","Total_Perf"]].values
y = df["Price"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Boost model
model = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=42
)

model.fit(X_train_s, y_train)
score = model.score(X_test_s, y_test)
mae = np.mean(np.abs(y_test - model.predict(X_test_s)))

print(f"GradientBoosting: R²={score*100:.1f}%, MAE=${mae:.2f}")

unique_vals = {k: sorted(df[k].unique().tolist()) for k in ["CPU","GPU","Brand"]}
cache = {}

# GUI
root = tk.Tk()
root.title("Laptop Price Predictor")
root.geometry("650x750")
root.configure(bg=COLORS["bg"])
entries = {}

style = ttk.Style()
style.theme_use('clam')
style.configure("C.TCombobox", fieldbackground="white", background=COLORS["card"], foreground="black", borderwidth=1)

# Header
header = tk.Frame(root, bg=COLORS["bg"], height=100)
header.pack(fill=tk.X)
header.pack_propagate(False)

tk.Label(header, text="LAPTOP PRICE PREDICTOR", font=("Segoe UI",20,"bold"), bg=COLORS["bg"], fg=COLORS["text"]).pack(pady=(25,5))
tk.Label(header, text="Machine Learning Powered Price Estimation", font=("Segoe UI",9), bg=COLORS["bg"], fg="#6c757d").pack()

sep = tk.Frame(root, height=1, bg=COLORS["border"])
sep.pack(fill=tk.X, padx=30)

main = tk.Frame(root, bg=COLORS["bg"])
main.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

# Input fields
for lbl, key, vals in [
    ("CPU","CPU",unique_vals["CPU"]),
    ("RAM","RAM",["4GB","8GB","16GB","32GB","64GB"]),
    ("RAM Type","Type",["DDR3","DDR4","DDR5"]),
    ("GPU","GPU",unique_vals["GPU"]),
    ("Storage","Storage",["128GB","256GB","512GB","1TB","2TB","3TB"]),
    ("Screen","Screen",["720p","1080p","1440p","4K","8K"]),
    ("Brand","Brand",unique_vals["Brand"])
]:
    card = tk.Frame(main, bg=COLORS["card"], relief=tk.SOLID, bd=1, highlightbackground=COLORS["border"])
    card.pack(fill=tk.X, pady=6)
    inner = tk.Frame(card, bg=COLORS["card"])
    inner.pack(fill=tk.X, padx=15, pady=10)
    tk.Label(inner, text=lbl+":", font=("Segoe UI",10), bg=COLORS["card"], fg=COLORS["text"], width=12).pack(side=tk.LEFT)
    combo = ttk.Combobox(inner, values=vals, width=35, font=("Segoe UI",9), style="C.TCombobox", state="readonly")
    combo.pack(side=tk.RIGHT)
    combo.set(vals[len(vals)//2])
    entries[key] = combo

def predict():
    try:
        key = tuple(e.get() for e in entries.values())
        if key in cache:
            show_result(cache[key])
            return

        vals = [
            score_cpu(entries["CPU"].get()),
            float(entries["RAM"].get().replace("GB","")),
            RAM_MAP[entries["Type"].get()],
            score_gpu(entries["GPU"].get()),
            extract_num(entries["Storage"].get()),
            SCREEN_MAP[entries["Screen"].get().upper()],
            BRAND_SCORES.get(entries["Brand"].get().upper(),5)
        ]

        vals.extend([
            vals[1]*vals[3]/100,
            vals[0]*vals[3]/100,
            (vals[0]+vals[3])/2
        ])

        pred = model.predict(scaler.transform([vals]))[0]
        cache[key] = pred
        show_result(pred)
    except Exception as e:
        messagebox.showerror("Error", str(e))

def show_result(pred):
    w = tk.Toplevel(root)
    w.title("Price Prediction")
    w.geometry("400x300")
    w.configure(bg=COLORS["bg"])
    w.resizable(False, False)
    w.transient(root)
    w.grab_set()
    
    tk.Label(w, text="ESTIMATED PRICE", font=("Segoe UI",12,"bold"), bg=COLORS["bg"], fg=COLORS["text"]).pack(pady=(30,10))
    
    pf = tk.Frame(w, bg=COLORS["card"], relief=tk.SOLID, bd=2)
    pf.pack(pady=10, padx=40, fill=tk.X)
    tk.Label(pf, text=f"${pred:,.2f}", font=("Segoe UI",32,"bold"), bg=COLORS["card"], fg=COLORS["text"]).pack(pady=20)
    
    tk.Label(w, text=f"Accuracy: {score*100:.1f}%  |  Avg Error: ±${mae:.0f}", font=("Segoe UI",9), bg=COLORS["bg"], fg="#6c757d").pack(pady=10)
    tk.Label(w, text=f"Dataset: {len(df)} laptops", font=("Segoe UI",8), bg=COLORS["bg"], fg="#6c757d").pack()
    
    tk.Button(w, text="Close", command=w.destroy, bg=COLORS["accent"], fg="white", font=("Segoe UI",10,"bold"), relief=tk.FLAT, padx=30, pady=8).pack(pady=20)

btn = tk.Button(main, text="PREDICT PRICE", command=predict, bg=COLORS["accent"], fg="white", font=("Segoe UI",12,"bold"), relief=tk.FLAT, padx=40, pady=12)
btn.pack(pady=20)

root.mainloop()

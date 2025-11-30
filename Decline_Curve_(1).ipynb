import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from openpyxl import load_workbook
from pandas import ExcelWriter
from openpyxl.drawing.image import Image as XLImage

# -------------------- Load Production Data --------------------
df = pd.read_excel("production_data.xlsx")
t = df["Time (months)"].values
q = df["Rate (STB/month)"].values

# -------------------- Decline Curve Models --------------------
def exp_decline(t, qi, D):
    return qi * np.exp(-D * t)

def harmonic_decline(t, qi, D):
    return qi / (1 + D * t)

def hyperbolic_decline(t, qi, D, b):
    denominator = 1 + b * D * t
    denominator = np.where(denominator <= 0, np.nan, denominator)
    return qi / (denominator ** (1 / b))

# -------------------- Fit Models --------------------
popt_exp, _ = curve_fit(exp_decline, t, q, p0=[1240, 0.05])
popt_har, _ = curve_fit(harmonic_decline, t, q, p0=[1240, 0.05])
try:
    popt_hyp, _ = curve_fit(
        hyperbolic_decline, t, q,
        p0=[1240, 0.05, 0.5],
        bounds=([800, 0.001, 0.2], [2000, 1.0, 1.5])
    )
except RuntimeError:
    popt_hyp = [0, 0, 1]

# -------------------- Forecast Rates --------------------
t_future = np.linspace(0, 60, 61)  # 0 to 60 months
exp_rate = exp_decline(t_future, *popt_exp)
har_rate = harmonic_decline(t_future, *popt_har)
hyp_rate = hyperbolic_decline(t_future, *popt_hyp)
hyp_rate = np.nan_to_num(hyp_rate, nan=0.0, posinf=0.0)

# -------------------- Create Reservoir DCA Table --------------------
forecast_df = pd.DataFrame({
    "Time (months)": t_future,
    "Exponential Rate": exp_rate,
    "Harmonic Rate": har_rate,
    "Hyperbolic Rate": hyp_rate
})

# -------------------- Plot Reservoir DCA --------------------
plt.figure(figsize=(12, 6))
plt.plot(t, q, 'ko', label="Actual Data")
plt.plot(t_future, exp_rate, 'b--', label="Exponential")
plt.plot(t_future, har_rate, 'r--', label="Harmonic")
plt.plot(t_future, hyp_rate, 'g--', label="Hyperbolic")
plt.xlabel("Time (months)")
plt.ylabel("Oil Rate (STB/month)")
plt.title("Decline Curve Analysis Forecast")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("dca_plot.png")
plt.show()

# -------------------- Calculate Cumulative Production --------------------
qi, D = popt_exp[0], popt_exp[1]
t_months = np.arange(0, 61, 1)
rate_exp = exp_decline(t_months, qi, D)
cumulative_exp = (qi - rate_exp) / D

# -------------------- Exponential Forecast Table --------------------
exp_df = pd.DataFrame({
    "Time (months)": t_months,
    "Rate (STB/month)": rate_exp,
    "Cumulative (STB)": cumulative_exp
})

# -------------------- Plot Graph 1: Rate & Cumulative vs Time --------------------
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(t_months, rate_exp, 'k-', label='Rate')
ax1.set_xlabel('Time (months)')
ax1.set_ylabel('Rate (STB/month)', color='black')
ax1.tick_params(axis='y', labelcolor='black')

ax2 = ax1.twinx()
ax2.plot(t_months, cumulative_exp / 1000, 'b--', label='Cumulative')
ax2.set_ylabel('Cumulative (kSTB)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title('Rate and Cumulative vs Time')
plt.grid(True)
plt.tight_layout()
plt.savefig("graph1.png")
plt.show()

# -------------------- Plot Graph 2: Rate vs Cumulative --------------------
plt.figure(figsize=(8, 5))
plt.plot(cumulative_exp, rate_exp, 'go-')
plt.xlabel('Cumulative (STB)')
plt.ylabel('Rate (STB/month)')
plt.title('Rate vs Cumulative Production')
plt.grid(True)
plt.tight_layout()
plt.savefig("graph2.png")
plt.show()

# -------------------- Save Everything to Excel --------------------
file_path = "forecast_output.xlsx"

# Step 1: Create main file with "Reservoir DCA" sheet
with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
    forecast_df.to_excel(writer, sheet_name="Reservoir DCA", index=False)

# Step 2: Append "Exponential Forecast" sheet
with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    exp_df.to_excel(writer, sheet_name="Exponential Forecast", index=False)


# Step 3: Insert Graphs into the "Exponential Forecast" sheet
book = load_workbook(file_path)
ws = book["Exponential Forecast"]

img1 = XLImage("graph1.png")
img1.width = 600
img1.height = 350
ws.add_image(img1, "E2")

img2 = XLImage("graph2.png")
img2.width = 600
img2.height = 350
ws.add_image(img2, "E30")

# Save Excel
book.save(file_path)

print("✅ Project completed. Excel file with both sheets and graphs saved successfully.")



# -------------------- Model Extensions --------------------

# Reuse t_future (0 to 60 months)
dt = 1  # month

# Cumulative Production Calculation via Summation
cum_exp = np.cumsum(exp_rate * dt)
cum_har = np.cumsum(har_rate * dt)
cum_hyp = np.cumsum(hyp_rate * dt)

cumulative_df = pd.DataFrame({
    "Time (months)": t_future,
    "Cumulative Exp": cum_exp,
    "Cumulative Harmonic": cum_har,
    "Cumulative Hyperbolic": cum_hyp
})

# -------------------- Abandonment Time --------------------
econ_limit = 100  # STB/month

# Get the time index when rate drops below economic limit
def get_abandonment_time(rate_array, time_array):
    for r, t in zip(rate_array, time_array):
        if r < econ_limit:
            return t
    return None

t_exp_ab = get_abandonment_time(exp_rate, t_future)
t_har_ab = get_abandonment_time(har_rate, t_future)
t_hyp_ab = get_abandonment_time(hyp_rate, t_future)

# -------------------- R² and Best-Fit Model --------------------
from sklearn.metrics import r2_score
exp_fit = exp_decline(t, *popt_exp)
har_fit = harmonic_decline(t, *popt_har)
hyp_fit = hyperbolic_decline(t, *popt_hyp)

r2_exp = r2_score(q, exp_fit)
r2_har = r2_score(q, har_fit)
r2_hyp = r2_score(q, hyp_fit)

r2_values = {
    "Exponential": r2_exp,
    "Harmonic": r2_har,
    "Hyperbolic": r2_hyp
}
best_fit_model = max(r2_values, key=r2_values.get)

# -------------------- Save Model Extension Sheet --------------------
with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    cumulative_df.to_excel(writer, sheet_name="Model Extensions", index=False)

# -------------------- Add Notes as New Rows --------------------
book = load_workbook(file_path)
ws = book["Model Extensions"]

ws.append([])
ws.append(["Economic Limit (STB/month):", econ_limit])
ws.append(["Abandonment Time (months) - Exponential:", t_exp_ab if t_exp_ab is not None else "Not reached within forecast period"])
ws.append(["Abandonment Time (months) - Harmonic:", t_har_ab if t_har_ab is not None else "Not reached within forecast period"])
ws.append(["Abandonment Time (months) - Hyperbolic:", t_hyp_ab if t_hyp_ab is not None else "Not reached within forecast period"])
ws.append([])
ws.append(["R² - Exponential", r2_exp])
ws.append(["R² - Harmonic", r2_har])
ws.append(["R² - Hyperbolic", r2_hyp])
ws.append(["Best Fit Model Based on R²:", best_fit_model])

book.save(file_path)

# ------add graph comparing cumulative production ------
# -------------------- Plot: Cumulative vs Time (All Models) --------------------
plt.figure(figsize=(10, 6))
plt.plot(t_future, cum_exp, label="Exponential", color='blue')
plt.plot(t_future, cum_har, label="Harmonic", color='red')
plt.plot(t_future, cum_hyp, label="Hyperbolic", color='green')
plt.xlabel("Time (months)")
plt.ylabel("Cumulative Production (STB)")
plt.title("Cumulative Production vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cumulative_comparison.png")
plt.show()

# -------------------- Insert Plot into Model Extensions Sheet --------------------
from openpyxl.drawing.image import Image as XLImage

# Reopen workbook again
book = load_workbook(file_path)
ws = book["Model Extensions"]

# Use direct file path, not open()
img = XLImage("cumulative_comparison.png")
img.width = 700
img.height = 400
ws.add_image(img, "F2")  # Position it

# Save workbook
book.save(file_path)

print("✅ Model Extensions sheet added with cumulative production, abandonment times, and R² analysis.")

# -------------------- Finally Done --------------------

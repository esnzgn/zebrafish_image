# pip install aicsimageio[all] aicspylibczi matplotlib napari

from aicsimageio import AICSImage
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Path to your .czi file on the external drive
czi_path = Path("/MOLM-16-16.7.2024/A-1-16.7.24.czi")

# %%
# Load the image
img = AICSImage(czi_path)
data = img.get_image_data("CYX", T=0, Z=0)  # First timepoint, first z-plane, all channels
print(f"Image shape: {data.shape}")

# %%
# Preview the first channel using matplotlib
plt.imshow(data[0], cmap='gray')
plt.title(czi_path.name)
plt.axis('off')
plt.show()


#

# %%
# Define CZI directory and conditions
czi_dir = Path("/Volumes/Seagate Expansion Drive/ZebrafishPaper/Nikole/MOLM-16-16.7.2024")
conditions = ["VA", "RU", "LS", "ctrl"]

# Helper: identify condition from filename
def identify_condition(name):
    name = name.upper()
    if "VA" in name:
        return "VA"
    elif "RU" in name:
        return "RU"
    elif "LS" in name and "CTRL" in name:
        return "LS_CTRL"
    elif "LS" in name:
        return "LS"
    else:
        return "Unknown"

# %%
# Collect signals
results = []

for czi_file in czi_dir.glob("*.czi"):
    try:
        img = AICSImage(czi_file)
        data = img.get_image_data("CYX", T=0, Z=0)
        signal = np.mean(data[0])  # mean of first channel

        condition = identify_condition(czi_file.name)
        results.append({
            "filename": czi_file.name,
            "condition": condition,
            "signal": signal
        })
    except Exception as e:
        print(f"Failed to process {czi_file.name}: {e}")

# %%
# Create dataframe
df = pd.DataFrame(results)
print(df.head())

# Save results if needed
df.to_csv("signal_summary.csv", index=False)

# %%
# Boxplot of signal intensities
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="condition", y="signal", palette="Set2")
sns.stripplot(data=df, x="condition", y="signal", color='black', jitter=True, alpha=0.5)
plt.title("Signal Intensities by Condition")
plt.ylabel("Mean Intensity")
plt.xlabel("Condition")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Statistical comparison: VA vs others
va_signals = df[df["condition"] == "VA"]["signal"]

for group in ["RU", "LS", "LS_CTRL"]:
    group_signals = df[df["condition"] == group]["signal"]
    t_stat, p_val = ttest_ind(va_signals, group_signals, equal_var=False)
    print(f"VA vs {group}: p = {p_val:.4e}")

# pip install aicsimageio[all] aicspylibczi matplotlib napari

# %%
from aicsimageio import AICSImage
from pathlib import Path
import matplotlib.pyplot as plt
from aicsimageio import AICSImage
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# %%
# Path to your .czi file on the external drive
czi_path = Path("/Volumes/Seagate Expansion Drive/ZebrafishPaper/Nikole/MOLM-16-16.7.2024/A-1-16.7.24.czi")

# %%
# Load the image
img = AICSImage(czi_path)
data = img.get_image_data("CYX", T=0, Z=0)  # First timepoint, first z-plane, all channels
print(f"Image shape: {data.shape}")

# %%
# Preview the first channel using matplotlib
plt.imshow(data[0], cmap='gray')
plt.title(czi_path.name)
plt.axis('off')
plt.show()


#

# %%
# Define CZI directory and conditions
czi_dir = Path("/Volumes/Seagate Expansion Drive/ZebrafishPaper/Nikole/MOLM-16-16.7.2024")
conditions = ["VA", "RU", "LS", "ctrl"]

# Helper: identify condition from filename
def identify_condition(name):
    name = name.upper()
    if "VA" in name:
        return "VA"
    elif "RU" in name:
        return "RU"
    elif "LS" in name and "CTRL" in name:
        return "LS_CTRL"
    elif "LS" in name:
        return "LS"
    else:
        return "Unknown"

# %%
# Collect signals
results = []

for czi_file in czi_dir.glob("*.czi"):
    try:
        img = AICSImage(czi_file)
        data = img.get_image_data("CYX", T=0, Z=0)
        signal = np.mean(data[0])  # mean of first channel

        condition = identify_condition(czi_file.name)
        results.append({
            "filename": czi_file.name,
            "condition": condition,
            "signal": signal
        })
    except Exception as e:
        print(f"Failed to process {czi_file.name}: {e}")

# %%
# Create dataframe
df = pd.DataFrame(results)
print(df.head())

# Save results if needed
df.to_csv("signal_summary.csv", index=False)

# %%
# Boxplot of signal intensities
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="condition", y="signal", palette="Set2")
sns.stripplot(data=df, x="condition", y="signal", color='black', jitter=True, alpha=0.5)
plt.title("Signal Intensities by Condition")
plt.ylabel("Mean Intensity")
plt.xlabel("Condition")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Statistical comparison: VA vs others
va_signals = df[df["condition"] == "VA"]["signal"]

for group in ["RU", "LS", "LS_CTRL"]:
    group_signals = df[df["condition"] == group]["signal"]
    t_stat, p_val = ttest_ind(va_signals, group_signals, equal_var=False)
    print(f"VA vs {group}: p = {p_val:.4e}")


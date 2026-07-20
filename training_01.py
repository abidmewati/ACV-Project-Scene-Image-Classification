import os
import zipfile
import glob
from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Configuration & Local Paths
# ==========================================
ZIP_FILE_PATH = "archive.zip"         # Path to your downloaded dataset zip (if applicable)
EXTRACT_TO_DIR = "./Dataset"          # Local extraction target directory

# Dataset splits (adjust according to your extracted folder layout)
TRAIN_DIR = os.path.join(EXTRACT_TO_DIR, "seg_train/seg_train")
TEST_DIR = os.path.join(EXTRACT_TO_DIR, "seg_test/seg_test")
PRED_DIR = os.path.join(EXTRACT_TO_DIR, "seg_pred/seg_pred")


def extract_dataset(zip_path, extract_path):
    """Extracts the dataset zip file if it exists and hasn't been extracted yet."""
    if os.path.exists(extract_path):
        print(f"[*] Target directory '{extract_path}' already exists. Skipping extraction.")
        return

    if os.path.exists(zip_path):
        print(f"[*] Extracting {zip_path} to {extract_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("[+] Extraction complete!")
    else:
        print(f"[!] Warning: {zip_path} not found. Please ensure your dataset is placed in: {extract_path}")


def analyze_class_distribution(base_path):
    """Traverses the dataset directory to count images per class and plots the distribution."""
    if not os.path.exists(base_path):
        print(f"[!] Path does not exist: {base_path}")
        return None

    class_counts = {}
    classes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for cls in classes:
        class_dir = os.path.join(base_path, cls)
        # Match standard image formats
        images = glob.glob(os.path.join(class_dir, "*.[jJ][pP]*[gG]")) + \
                 glob.glob(os.path.join(class_dir, "*.[pP][nN][gG]"))
        class_counts[cls] = len(images)

    df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count']).sort_values(by='Count', ascending=False)
    
    print(f"\n[*] Class distribution in: {base_path}")
    print(df.to_string(index=False))
    return df


def plot_distribution(df, split_name):
    """Generates and displays a bar plot of the dataset distribution."""
    if df is None or df.empty:
        return
        
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Class', y='Count', data=df, palette='viridis')
    plt.title(f"Class Distribution - {split_name} Split")
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def verify_and_clean_images(base_path):
    """
    Scans the dataset to identify, report, and optionally clean up corrupt, 
    unreadable, or zero-byte files that can cause PyTorch Dataloader crashes.
    """
    print(f"\n[*] Verifying images in: {base_path}...")
    corrupt_files = []
    total_scanned = 0

    image_paths = glob.glob(os.path.join(base_path, "**/*.[jJ][pP]*[gG]"), recursive=True) + \
                  glob.glob(os.path.join(base_path, "**/*.[pP][nN][gG]"), recursive=True)

    for img_path in image_paths:
        total_scanned += 1
        try:
            with Image.open(img_path) as img:
                img.verify()  # Verifies image integrity without fully loading it
        except (IOError, SyntaxError) as e:
            print(f"[!] Corrupt or unreadable image found: {img_path}")
            corrupt_files.append(img_path)

    print(f"[+] Scan Complete. Scanned: {total_scanned} files. Corrupt files found: {len(corrupt_files)}")
    
    if corrupt_files:
        choice = input("Do you want to delete/quarantine these corrupt images? (y/n): ").strip().lower()
        if choice == 'y':
            for f in corrupt_files:
                try:
                    os.remove(f)
                    print(f"[-] Removed: {f}")
                except Exception as e:
                    print(f"[!] Failed to remove {f}: {e}")
    return corrupt_files


def inspect_image_properties(base_path):
    """Inspects and displays resolution metrics and color channel statistics of the dataset."""
    print(f"\n[*] Inspecting image dimensions in: {base_path}...")
    image_paths = glob.glob(os.path.join(base_path, "**/*.[jJ][pP]*[gG]"), recursive=True)[:100] # Check first 100 samples
    
    if not image_paths:
        print("[!] No sample images found to inspect.")
        return

    sizes = []
    modes = []
    
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)  # (width, height)
                modes.append(img.mode)  # RGB, L, RGBA etc.
        except Exception:
            continue

    df_sizes = pd.DataFrame(sizes, columns=['Width', 'Height'])
    print("\nSample Image Dimensions (First 100 images):")
    print(df_sizes.describe())
    
    print("\nImage Modes Found:")
    print(pd.Series(modes).value_counts())


def visualize_samples(base_path, num_samples=5):
    """Visualizes sample images from each class directory."""
    if not os.path.exists(base_path):
        return
        
    classes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for cls in classes:
        class_dir = os.path.join(base_path, cls)
        images = glob.glob(os.path.join(class_dir, "*.[jJ][pP]*[gG]"))[:num_samples]
        
        if not images:
            continue
            
        fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
        fig.suptitle(f"Sample images from class: {cls}", fontsize=14)
        
        # Handle cases where only 1 image is present
        if len(images) == 1:
            axes = [axes]
            
        for ax, img_path in zip(axes, images):
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(f"{img.size[0]}x{img.size[1]}")
                ax.axis('off')
            except Exception as e:
                ax.axis('off')
                
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("=== DATASET PREPROCESSING PIPELINE ===")
    
    # Step 1: Handle Dataset Extraction
    extract_dataset(ZIP_FILE_PATH, EXTRACT_TO_DIR)
    
    # Step 2: Quality Control (Verify file integrity before starting training)
    if os.path.exists(TRAIN_DIR):
        verify_and_clean_images(TRAIN_DIR)
    if os.path.exists(TEST_DIR):
        verify_and_clean_images(TEST_DIR)

    # Step 3: Class Distribution Analysis
    print("\n--- Analyzing Split Distributions ---")
    train_dist = analyze_class_distribution(TRAIN_DIR)
    test_dist = analyze_class_distribution(TEST_DIR)
    
    # Step 4: Visualize Distributions
    if train_dist is not None:
        plot_distribution(train_dist, "Training")
    if test_dist is not None:
        plot_distribution(test_dist, "Testing")

    # Step 5: Check Dimensions and Channels
    if os.path.exists(TRAIN_DIR):
        inspect_image_properties(TRAIN_DIR)
        
    # Step 6: Display Sample Images
    print("\n--- Displaying Class Samples ---")
    visualize_samples(TRAIN_DIR, num_samples=5)
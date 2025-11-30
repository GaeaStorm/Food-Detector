# Imports
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# Define the Inventory Detection Function
# Code stitched from https://www.linkedin.com/pulse/smart-fridge-ai-object-detection-system-jigar-joshi-p6zse/

def detect_inventory(model, image_path, confidence_threshold=0.3, iou_threshold=0.5):
    print(f"\nProcessing image for inventory detection: {image_path}")

    # Verify that the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at path: {image_path}")
        return pd.DataFrame(), None

    try:
        results = model.predict(
            source=image_path,
            conf=confidence_threshold,
            iou=iou_threshold,
            save=False,
            verbose=False
        )
    except Exception as e:
        print(f"Error during prediction: {e}")
        return pd.DataFrame(), None

    first_result = results[0] if results else None
    detections = first_result.boxes if first_result else None

    if detections is None or len(detections) == 0:
        print("No items detected in the image.")
        return pd.DataFrame(), results

    # Initialize an inventory list to store detected items
    inventory = []
    for box in detections:
        cls_id = int(box.cls)  # Class ID
        confidence = box.conf.item()  # Confidence score
        class_name = model.names[cls_id]  # Get class name from model's class names
        if confidence < confidence_threshold:
            continue
        inventory.append([class_name, confidence])

    # Create a DataFrame to store the inventory
    inventory_df = pd.DataFrame(inventory, columns=['Item', 'Confidence'])
    inventory_df['Count'] = 1
    inventory_df = inventory_df.groupby('Item').agg({'Count': 'sum', 'Confidence': 'mean'}).reset_index()

    print(f"Detected Inventory:\n{inventory_df}")
    return inventory_df, results

# Define the Function to Display Results with Bounding Boxes

def display_results(model, image_path, results):
    
    print(f"\nDisplaying results for image: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image with OpenCV: {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide axis

    first_result = results[0] if results else None
    detections = first_result.boxes if first_result else None

    if detections is None or len(detections) == 0:
        print("No detections to display.")
        plt.show()
        return

    for box in detections:
        cls_id = int(box.cls)
        confidence = box.conf.item()
        class_name = model.names[cls_id]

        xyxy = box.xyxy.tolist()

        if isinstance(xyxy[0], list):
            xyxy = xyxy[0]

        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{class_name} {confidence:.2f}"

        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         fill=False, edgecolor='red', linewidth=2))
        plt.text(x1, y1 - 10, label, color='red', fontsize=12,
                 bbox=dict(facecolor='yellow', alpha=0.5))

    plt.show()

# UI Functions for Interacting with Inventory

def save_inventory(inventory_df, filename="inventory.csv"):
    """Save the detected inventory to a CSV file."""
    if not inventory_df.empty:
        inventory_df.to_csv(filename, index=False)
        print(f"Inventory saved to {filename}")
    else:
        print("No inventory to save.")

def load_inventory(filename="inventory.csv"):
    """Load inventory from a CSV file."""
    if os.path.exists(filename):
        inventory_df = pd.read_csv(filename)
        print(f"Inventory loaded from {filename}")
        return inventory_df
    else:
        print(f"File {filename} not found.")
        return pd.DataFrame()

# UI Widgets for Saving/Loading/Displaying Inventory
def inventory_ui(inventory_df):
    """Create an interactive UI for inventory management."""
    output = widgets.Output()

    def on_save_clicked(b):
        with output:
            clear_output()
            save_inventory(inventory_df)

    def on_load_clicked(b):
        with output:
            clear_output()
            inventory_df = load_inventory()
            display(inventory_df)

    def on_display_clicked(b):
        with output:
            clear_output()
            if not inventory_df.empty:
                display(inventory_df)
            else:
                print("No inventory to display.")

    save_button = widgets.Button(description="Save Inventory")
    load_button = widgets.Button(description="Load Inventory")
    display_button = widgets.Button(description="Display Inventory")

    save_button.on_click(on_save_clicked)
    load_button.on_click(on_load_clicked)
    display_button.on_click(on_display_clicked)

    display(widgets.VBox([save_button, load_button, display_button, output]))

def expand_ingredients(user_ingredients):
    """
    Expand a list of your 26 canonical ingredient categories
    into a list of all synonyms used in Recipe1M+.

    Example:
        ["brinjal", "basil"] →
        ["brinjal", "eggplant", "aubergine", "basil", "fresh basil"]
    """
    
    INGREDIENT_SYNONYMS = {
        "apple": ["apple"],
        "basil": ["basil", "fresh basil"],
        "bitter gourd": ["bitter gourd", "bitter melon"],
        "bread": ["bread"],
        "brinjal": ["brinjal", "eggplant", "aubergine"],
        "butter": ["butter", "unsalted butter"],
        "cabbage": ["cabbage"],
        "capsicum": ["capsicum", "bell pepper", "green pepper", "red pepper"],
        "carrots": ["carrot", "carrots"],
        "cauliflower": ["cauliflower"],
        "chillies": ["chillies", "chili", "chilies", "chili pepper", "red chili"],
        "coriander": ["coriander", "cilantro"],
        "cucumber": ["cucumber"],
        "egg": ["egg", "eggs"],
        "garlic": ["garlic"],
        "ginger": ["ginger"],
        "lemon": ["lemon", "lemon juice", "lemon zest"],
        "lettuce": ["lettuce"],
        "milk": ["milk", "whole milk", "skim milk"],
        "okra": ["okra"],
        "onion": ["onion", "yellow onion", "red onion"],
        "potato": ["potato", "potatoes"],
        "pumpkin": ["pumpkin"],
        "tomato": ["tomato", "tomatoes"],
        "watermelon": ["watermelon"],
    }

    expanded = []

    for ing in user_ingredients:
        ing_norm = ing.strip().lower()
        if ing_norm in INGREDIENT_SYNONYMS:
            expanded.extend(INGREDIENT_SYNONYMS[ing_norm])
        else:
            # in case you ever pass unknown ingredients
            expanded.append(ing_norm)

    # Remove duplicates while preserving order
    seen = set()
    unique_expanded = []
    for item in expanded:
        if item not in seen:
            seen.add(item)
            unique_expanded.append(item)

    return unique_expanded


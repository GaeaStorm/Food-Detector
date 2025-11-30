# Imports

import os
import cv2
import torch
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("runs/detect/train2/weights/best.pt")

from helper import *

# Detect Inventory and Use UI

image_path = 'dataset/test/images/7df06d79-021c-47c8-9541-e16dc63f3622_JPG.rf.71c0bb9c97749e464a30c6be78a2c70c.jpg'  # Update with your image path
print(f"Checking image path: {image_path}")

inventory_df, results = detect_inventory(model, image_path)

# Show the image with detected bounding boxes
if results:
    display_results(model, image_path, results)
else:
    print("No results to display.")

ingredients_list = expand_ingredients(inventory_df['Item'].to_list())

from query_recipes import suggest_recipes_raw, _print_results

results = suggest_recipes_raw(ingredients=ingredients_list, 
                              max_kcal_per_100g=None,  # e.g. 150 to filter by calories 
                              cuisine=None,            # e.g. "italian" if you add cuisine labels later 
                              n_results=50, 
                              max_return=5, )

_print_results(results)
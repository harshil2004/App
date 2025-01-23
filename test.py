import os

filepath = "D:\\App\\pr_2\\trained_model.keras"
if os.path.exists(filepath):
    print("File found.")
else:
    print("File not found.")
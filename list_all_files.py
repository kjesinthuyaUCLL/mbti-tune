import os

TARGET_FOLDERS = ["app", "data", "models", "notebooks", "scripts", "src"]

print("\n📁 PROJECT STRUCTURE (filtered)\n")

# 1. Print root-level files
print("ROOT FOLDER FILES:")
for f in os.listdir("."):
    if os.path.isfile(f):
        print("  -", f)

print("\n--------------------------------------\n")

# 2. Print only the relevant folders and their contents
for folder in TARGET_FOLDERS:
    if os.path.exists(folder):
        print(f"{folder}/")
        for root, dirs, files in os.walk(folder):
            # Only show up to 2 levels deep for readability
            level = root.count(os.sep) - folder.count(os.sep)
            if level > 1:
                continue

            indent = "    " * level
            print(f"{indent}{os.path.basename(root)}/")

            subindent = "    " * (level + 1)
            for f in files:
                print(f"{subindent}{f}")

        print("\n--------------------------------------\n")
    else:
        print(f"{folder}/  (not found)\n--------------------------------------\n")

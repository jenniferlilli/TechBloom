#need to test with files
import os
import zipfile

UPLOAD_FOLDER = 'uploads'

def extractIDS(filepath):
    badge_ids = set()
    with open(filepath, 'r') as f:
        for line in f:
            for item in line.strip().split(','):
                badge_ids.add(item.strip())
    return list(badge_ids)

def extractImages(zipPath, extractTo):
    imagePaths = []
    with zipfile.ZipFile(zipPath, 'r') as zipRef:
        zipRef.extractall(extractTo)
        for root, _, files in os.walk(extractTo):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    imagePaths.append(os.path.join(root, file))
    return imagePaths

# Process the latest badge file and zip file
badge_file = None
zip_file = None

for fname in os.listdir(UPLOAD_FOLDER):
    if fname.lower().endswith(('.csv', '.txt')):
        badge_file = os.path.join(UPLOAD_FOLDER, fname)
    elif fname.lower().endswith('.zip'):
        zip_file = os.path.join(UPLOAD_FOLDER, fname)

if badge_file and zip_file:
    print(f'Found badge file: {badge_file}')
    print(f'Found zip file: {zip_file}')

    badge_ids = extractIDS(badge_file)
    print(f'\nExtracted Badge IDs:\n{badge_ids}')

    image_folder = os.path.join(UPLOAD_FOLDER, 'extracted_images')
    os.makedirs(image_folder, exist_ok=True)
    image_paths = extractImages(zip_file, image_folder)
    print(f'\nExtracted image paths:\n{image_paths}')
else:
    print("Required files not found in uploads/")

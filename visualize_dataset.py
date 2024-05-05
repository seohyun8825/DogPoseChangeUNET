import json
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def visualize_image_with_keypoints(image_path, keypoints, bbox):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # 바운딩 박스 그리기
    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    draw.rectangle([(bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h)], outline="green", width=2)

    # 키포인트 그리기
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
        if v != 0:  # 키포인트가 보이는 경우에만 그림
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill='red', outline='red')

    plt.imshow(image)
    plt.axis('off')
    plt.show()

def main():
    json_file_path = r"C:\Users\user\Desktop\24-1\학회\APT-36k\AP-36k-patr1\apt36k_annotations.json"
    data = load_json_data(json_file_path)
    
    annotations = data['annotations']
    images = {img['id']: img for img in data['images']}

    for annotation in annotations:
        if annotation['category_id'] == 2:  
            image_id = annotation['image_id']
            keypoints = annotation['keypoints']
            bbox = annotation['bbox']  
            
            if image_id in images:
                image_data = images[image_id]
                image_path = os.path.join(r"C:\Users\user\Desktop\24-1\학회\APT-36k\AP-36k-patr1", image_data['file_name'].replace("D:\\Animal_pose\\AP-36k-patr1\\", ""))
                visualize_image_with_keypoints(image_path, keypoints, bbox)

if __name__ == "__main__":
    main()

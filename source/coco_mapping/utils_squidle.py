import os

def get_image_name(url):
    img_name = os.path.split(url)
    if len(img_name[1]) == 0:
        img_name = os.path.basename(img_name[0])
    else:
        img_name = img_name[1]
    return img_name
import pandas as pd
import os
from PIL import Image
import PIL
import glob


class hello():
    
    def resize_images(self):
        rootdir = '/Users/dq/Documents/aicore_project/Airbnb_Project/images/'
        file_list=[]
        for paths, dirs, files in os.walk(rootdir):
            for file in files:
                if file.endswith(".png"):
                    file_list.append(os.path.join(paths, file))

        resized = []
        for f in file_list:
            image = Image.open(f)
            width = image.size[0]
            height = image.size[1]
            aspect_ratio = height/width
            new_width=720
            new_height=int(new_width*aspect_ratio)
            # print(image.size)
            resized_image = image.resize((new_width,new_height))
            resized.append(resized_image)

        for f in resized:
            if resized_image.mode!="RGB":
                resized.remove(f)
        index=0
        image_path = "/Users/dq/Documents/aicore_project/Airbnb_Project/processed_images/"
        os.mkdir(image_path)
        for i in resized:
            index+=1
            image.save(f"{image_path}/{index}.png")

        return resized

        # index=0
        # image_path = "/Users/dq/Documents/aicore_project/Airbnb_Project/processed_images/"
        # os.mkdir(image_path)
        # for i in resized:
        #     with open(os.path.join(f"{image_path}", f"{index}.png"), "wb") as img:
        #         image.save(img)
        #         index+=1

if __name__ == "__main__":
    s = hello()
    clean_image_data = s.resize_images()
    print(clean_image_data)
        
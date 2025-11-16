# from datasets import load_dataset

# ds = load_dataset("yuntian-deng/im2latex-100k")
# print(ds)
# example = ds["train"][0]
# print(example["formula"])
# print(type(example["image"]))

import kagglehub

# Download latest version
path = kagglehub.dataset_download("shahrukhkhan/im2latex100k")

print("Path to dataset files:", path)
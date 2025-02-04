# Training an RCNN Model Using a Custom Dataset and Detecting Objects Using DroidCam  

I did this as part of my learning for my final year project, which is based on a V2X Traffic Management System. As part of my project, I need to be able to detect vehicles using a camera and track their path. So, I did this to learn the basics of machine learning and familiarize myself with PyTorch. Since I made some progress, I thought, why not share it publicly so that it might be useful to someone out there? Most of the things in this tutorial are already available publicly, and possibly in a better way. I’ll link all the sources I’ve used. If you truly want to learn, I recommend checking out the sources directly.  

### Full Disclosure  
I don’t claim that all the information in this series is entirely correct, nor do I claim that all of it is my own work. The majority of the content is taken from other sources, and I have simply consolidated it for my particular case. I sincerely apologize if any information is incorrect. I’ve tried to credit the sources wherever possible; if I’ve missed anyone by mistake, I truly apologize.  

## Project Overview  
In this project, we’ll train an RCNN model on a custom dataset of images featuring a power bank and my spectacle case. After training, we’ll use DroidCam to wirelessly connect to our phone camera and detect objects. I’ll be using PyTorch and assuming that you have PyTorch and Jupyter Notebook set up and running.  

---

## Step 1: Downloading and extracting the Images  
The first step is to download the dataset from this [link](https://drive.google.com/file/d/1OIvEIFwkrTDZkjfE-0tlxPGKdaYfgp_W/view?usp=drive_link). Next extract the `custom_sample_dataset.zip` file. The folder is structured as follows:  

**_Insert Folder Structure Here_**  

You can use your own dataset if you want. I used `labelimg` to label 131 images of the power bank and 130 images of my spectacle case. There are plenty of tutorials available on labeling using `labelimg`; you can check them out if needed.  

Each image category is numbered from 1 to 130 or 131, with their class name and a corresponding XML file that contains the coordinates of the bounding boxes for our objects of interest.  

---

## Step 2: Running the Notebook File  
I’ve provided the `.ipynb` file in this repository. You can check it out. I’ll briefly explain what each cell does. I’ve extensively commented on the notebook, so you can check that out as well. Most of the code in this tutorial is from the official PyTorch tutorial series, specifically:  

🔗 **[PyTorch Object Detection Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)**  

If you want more in-depth knowledge, I highly recommend referring to the official tutorial.  

### 2.1 Creating a Custom Dataset Class  
The first step is to create a custom class for our dataset. All the specifics required by our model (`fasterrcnn_resnet50_fpn`) are provided in the notebook.  

Our dataset class should have two methods:  
- `__len__` → Returns the length of the dataset.  
- `__getitem__` → Returns a tuple `(image, target)`.  

The `parse_annotation` method extracts the bounding box locations and class names. One mistake I made while capturing images was leaving my phone’s auto-rotate feature on. As a result, I need to rotate the images before sending them for training. I don’t recommend using other transforms like random horizontal flips, as they can heavily offset the bounding box locations.  

---

### 2.2 Testing the Dataset _(Optional)_  
This section is entirely optional. We print some images along with their bounding boxes using our custom dataset class. Make sure to correctly set the dataset directory.  

Example:  
```python
dataset = CustomDataset(root_dir="/home/adarshh/ml/tf_gpu/custom_sample_dataset/")
```
---
### 2.3 Fine-Tuning from a Pretrained Model
We use a fine-tuning approach here. We take a pre-trained model and replace the last fully connected (fc) layer with our custom trained layer using:
```
FastRCNNPredictor(in_features, num_classes)
```
- `in_features` : Number of output features going into the last fc layer.
- `num_classes` : Number of output classes required.

We have three classes:
- `spec_case`
- `power_bank`
- `background` (required for the model)
---
### 2.4 Installing Helper Functions
In this section, we install some helper functions for training and displaying statistics.

⚠ **Note:** Run this only once and then comment out the entire section. Otherwise, every time this block runs, new files will be downloaded, consuming unnecessary disk space.

---
### 2.5 Data Augmentation and Transformations
This function applies transformations to our images, such as converting them to tensors or applying horizontal flips (if needed).

### 2.6 Testing the `forward()` Method
Before iterating over the dataset, it’s good to check what the model expects during training and inference using sample data.

Make sure to correctly set the dataset directory(example):
```
dataset = CustomDataset(root_dir="/home/adarshh/ml/tf_gpu/custom_sample_dataset/")

```
---
### 2.7 Main Training Function
This is the main function that trains the model. The code will automatically use an NVIDIA GPU if available and configured correctly.
- The dataset is split into train and test sets within the code.
- Set the number of epochs using the `num_epochs` variable.
- After training, the model is saved at the location specified in `best_model_params_path`. This prevents the need to re-train the model every time you run the script.

⚠ **Note**: Ensure that the path ends with model_name.pt. 

---

## Step 3: Testing on DroidCam
In this step, we test the trained model using DroidCam.

### Steps:
1. Install DroidCam on your phone and open it.
2. Find the IP address and enter it in the `droidcam_url` variable.
3. Run the code and show an object in front of the camera to check detection.
4. Press `q` to exit the video feed.

Example:
```
droidcam_url = "http://XXX.XXX.X.XX:4747/video"
```

## Footnote
That’s it! I hope this guide helps you.

In my case, I get around 12 FPS, even with my GPU enabled, which is quite slow for real-time applications. As my next step, I’m planning to try the YOLOv8 model instead of RCNN. Hopefully, that will give better performance.

Thank you for reading!

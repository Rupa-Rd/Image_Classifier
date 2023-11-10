import argparse
import json
import PIL
import torch
import numpy as np
from math import ceil
from torchvision import models

def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image', type=str, help='Path to image file for prediction.', required=True)
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file as str.', required=True)
    parser.add_argument('--top_k', type=int, help='Choose top K matches as int.')
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

    args = parser.parse_args()
    
    return args

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    
    for param in model.parameters(): 
        param.requires_grad = False
    
    # Load from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    img = PIL.Image.open(image)

    original_width, original_height = img.size
    
    if original_width < original_height:
        size = [256, 256**600]
    else: 
        size = [256**600, 256]
        
    img.thumbnail(size)
   
    center = original_width / 4, original_height / 4
    left, top, right, bottom = center[0] - (244 / 2), center[1] - (244 / 2), center[0] + (244 / 2), center[1] + (244 / 2)
    img = img.crop((left, top, right, bottom))

    numpy_img = np.array(img) / 255 

    # Normalize each color channel
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    numpy_img = (numpy_img - mean) / std
        
    # Set the color to the first channel
    numpy_img = numpy_img.transpose(2, 0, 1)
    
    return numpy_img

def predict(image, model, device, cat_to_name, topk=5):
    model.to(device)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        log_ps = model.forward(image)
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(topk, dim=1)
    
    class_to_idx = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [class_to_idx[lab] for lab in top_class.cpu().numpy()[0]]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_p[0].cpu().numpy(), top_labels, top_flowers

def print_probability(probs, flowers):
    for i, (flower, prob) in enumerate(zip(flowers, probs)):
        print(f"Rank {i+1}: Flower: {flower}, Likelihood: {prob*100:.2f}%")

def main():
    args = arg_parser()
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)
    
    image_tensor = process_image(args.image)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu == "gpu" else "cpu")
    
    top_probs, top_labels, top_flowers = predict(image_tensor, model, device, cat_to_name, args.top_k)
    
    print_probability(top_probs, top_flowers)

if __name__ == '__main__':
    main()

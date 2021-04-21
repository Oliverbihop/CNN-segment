import argparse
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import torch
from segment_pt import Net
import numpy as np
import cv2
import time

def main():
    use_gpu=True
    device='cuda:0'
    cap = cv2.VideoCapture('output_01_ahead.avi')
    model = Net().to(device)
    #summary(model,(3,80,160))
    loaded_state_dict = torch.load('./lane.pt')
    model.load_state_dict(loaded_state_dict)

    if use_gpu:
        torch.cuda.set_device(device=0)
        model.cuda()
    model.eval()
    while(1):

        ret, img= cap.read()
        img=img[70:230,:,:]
        cv2.imshow('ssa',img)
        #cv2.waitKey(0)
        t=time.time()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        loader = transforms.Compose([
            transforms.Resize((80, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        img_tensor = loader(img).unsqueeze(0)
        input_image = Variable(img_tensor, requires_grad=False)
        if use_gpu:
            input_image = input_image.cuda()

        with torch.no_grad():
            outputs = model.forward(input_image)
            print(outputs.size())
            pred = outputs.detach().cpu().numpy()
        pred=pred[0,:,:,:]*255
        pred=pred.transpose(1,2,0)
        pred=np.where(pred < 125, 0, 255)
        pred=pred[:,:,-1]
        pred=pred.astype(np.uint8)
        print(int(1/(time.time()-t)))
        print(pred.shape)

        print(pred.shape)
        cv2.imshow('pre',pred)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
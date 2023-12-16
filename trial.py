# %%
import numpy as np
import cv2
 
def show_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

#%%
'''
    read video
'''
# file_name = "54v_20230803_151926"
file_name = "55v_20230803_152257"
vid_name = "{0}.avi".format(file_name)
cap = cv2.VideoCapture(vid_name)
# cap = cv2.VideoCapture(vid_name)
frames = []
 
while(cap.isOpened()):
    ret, frame = cap.read()
    frames.append(frame)
 
    if frame is None:
        break
 
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
target = frames[100]
cap.release()
cv2.destroyAllWindows()

# %%
def find_line(gray):
    edges = cv2.Canny(gray,350, 400,apertureSize = 5)
    minLineLength=100
    maxLineGap=80
    lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=maxLineGap)
    a,b,c = lines.shape
    line_image = np.copy(gray) * 0
    for i in range(a):
        cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.line(line_image, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 255), 3, cv2.LINE_AA)
        # cv2.imwrite('houghlines5.jpg',gray)
    return gray, line_image


def find_contour(image, ori_image, rescale = 0.95):
    image = image.copy()
    # Find contours and filter using threshold area
    cnts, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = 180000
    max_area = 250000
    wh_ratio_threshold = 1.25
    rect = None
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            x,y,w,h = cv2.boundingRect(c)
            if w / h < wh_ratio_threshold:
                cx = x + w // 2
                cy = y + h // 2
                w = int(w * rescale)
                h = int(h * rescale)
                x = cx - w // 2
                y = cy - h // 2
                ROI = image[y:y+h, x:x+w]
                cv2.rectangle(ori_image, (x, y), (x + w, y + h), (36,255,12), 2)
                rect = [x,y,w,h]
 
    return image.copy(), ori_image.copy(), rect

#%%
'''
    detect and output
'''
kernel = np.ones((25, 1), np.uint8)
square_rescale = 0.9

dryout_min_px, dryout_max_px = 95, 200
base_ratio = 0.03

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_name = "{0}_DryoutMasked.avi".format(file_name)
out_vid = cv2.VideoWriter(out_name, fourcc, 20.0, (2688, 920))

ratio = []

for frame in frames:
    # gray= cv2.cvtColor(verti, cv2.COLOR_BGR2GRAY)

    if frame is None:
        break

    verti = cv2.erode(frame.copy(), kernel, iterations = 10)
    verti = cv2.cvtColor(verti, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(verti, 21)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    blurred = cv2.GaussianBlur(sharpen, (3, 3), 3)
    line_img, edge_img = find_line(sharpen.copy())
    contour_img, ori_with_edge_img, rect = find_contour(edge_img.copy(), frame.copy(), square_rescale)

    # demo
    high_contrast = frame.copy()
    gradient_img = frame.copy()
    if rect != None:
        x,y,w,h = rect
        cropped = frame[y:y+h, x:x+w].copy()
        
        alpha = 2
        beta = 0
        target = cv2.convertScaleAbs(cropped, alpha=alpha, beta=beta)
        # target = cv2.addWeighted(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), alpha, target, 0, beta)
        gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        ada = cv2.adaptiveThreshold(gray, maxValue=250, 
                                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                thresholdType=cv2.THRESH_BINARY,
                                blockSize = 551, C = -4)
        ret,thresh = cv2.threshold(ada,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        # get each contours area
        min_area = w * h / 50
        detected_dryout = cropped.copy()
        mask = cropped.copy() * 0
        
        # gradient mask
        gradient_mask = range(1, w + 1)
        base = int(w * base_ratio)
        gradient_mask = np.minimum(np.emath.logn(base, gradient_mask), 1)
        
        for i in range(3):
            cropped[:,:,i] = cropped[:,:,i] * gradient_mask
        
        for c in contours:
            if cv2.contourArea(c) > min_area:
                temp = cropped.copy() * 0
                cv2.drawContours(temp, [c], -1, (255,255,255), cv2.FILLED)
                # show_image(temp)
                temp[temp == 255] = 1
                
                temp_crop = cropped.copy()
                temp_crop *= temp
                
                non_zero = temp_crop[np.where(temp_crop!=0)]
                # mean = np.sum(non_zero) / len(non_zero)
                quantile = np.quantile(non_zero, 0.8)
                if dryout_min_px <= quantile <= dryout_max_px:
                    # print(quantile)
                    mask += temp
        
        mask[mask > 0] = 1
        total_pixels = np.shape(mask)[0] * np.shape(mask)[1]
        covered = np.sum(mask[:, :, 0])
        ratio.append(covered / total_pixels)
        
        mask[:, :, 0] *= 255
        detected_dryout[:, :, 0] = mask[:, :, 0]
        # detected_dryout *= mask
        
        ada = cv2.cvtColor(ada,cv2.COLOR_GRAY2RGB)
        high_contrast[y:y+h, x:x+w] = ada
        gradient_img[y:y+h, x:x+w] = cropped

        img = frame.copy()
        img[y:y+h, x:x+w] = detected_dryout
    else:
        img = frame.copy()
    
    out_img = cv2.hconcat([frame, img, high_contrast])
    out_vid.write(out_img)

    cv2.imshow('frame', out_img)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
out_vid.release()
cv2.destroyAllWindows()
# %%

import matplotlib.pyplot as plt
plt.plot(range(len(ratio)), ratio)
plt.title(file_name)
# %%

import pandas as pd

out = pd.DataFrame(
    {
        'frame': range(1, len(ratio)+1),
        'ratio': ratio
    }
)
out.to_csv(f"{file_name}_DryoutMasked_records.csv", index = False)
# %%

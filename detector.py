import cv2
import numpy as np
from tqdm import tqdm

def read_video(path):
    frames = []
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        frames.append(frame)
    
        if frame is None:
            break
    
    cap.release()
    
    return width, height, frames

# Enhence the edges of the slides
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

# find the contour of the dryout area and slides
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

# get the rect (x,y,w,h) from coordinates
def get_rect(coords):
    tl, tr = coords['top-left'], coords['top-right']
    bl, br = coords['bottom-left'], coords['bottom-right']
    x = (tl[0] + bl[0]) // 2
    y = (tl[1] + tr[1]) // 2
    w = ((tr[0] - tl[0]) + (br[0] - bl[0])) // 2
    h = ((bl[1] - tl[1]) + (br[1] - tr[1])) // 2
    
    return x,y,w,h

# main function
def dryout_detection(frame,
                    #  kernel,
                    #  square_rescale,
                     coords,
                     dryout_min_px,
                     dryout_max_px,
                     dryout_area_ratio,
                     dryout_area_quantile,
                     left_darken_gradient_mask_rate,
                     top_brighten_gradient_mask_rate,
                     bottom_brighten_gradient_mask_rate):
    
    if frame is None:
        return -1, None, None

    # Depreciated, switched to user defined coordinates
    # verti = cv2.erode(frame.copy(), kernel, iterations = 10)
    # verti = cv2.cvtColor(verti, cv2.COLOR_BGR2GRAY)
    # blur = cv2.medianBlur(verti, 21)
    # sharpen_kernel = np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]])
    # sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    # # blurred = cv2.GaussianBlur(sharpen, (3, 3), 3)
    # line_img, edge_img = find_line(sharpen.copy())
    # contour_img, ori_with_edge_img, rect = find_contour(edge_img.copy(), frame.copy(), square_rescale)

    # demo
    high_contrast = frame.copy()
    gradient_img = frame.copy()
    blurred = frame.copy()
    # if rect != None:
        # x,y,w,h = rect
    
    x,y,w,h = get_rect(coords)
    cropped = frame[y:y+h, x:x+w].copy()
    cropped_contours = cropped.copy()
    
    alpha = 2
    beta = 0
    target = cv2.convertScaleAbs(cropped, alpha=alpha, beta=beta)
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    
    # turn reflection to darker pixels
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    gray[thresh == 255] = np.mean(gray) - 30

    # blur, with ada
    blur = cv2.bilateralFilter(gray, 31, 30, 20)
    ada = cv2.adaptiveThreshold(blur, maxValue=255, 
                            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                            thresholdType=cv2.THRESH_BINARY,
                            blockSize = 1501, C = -4)
    ret,thresh = cv2.threshold(ada,127,255,0)
    
    # Otsu threhsold, too loose
    # blur = cv2.GaussianBlur(gray,(101, 101), 3)
    # _,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(cropped_contours, contours, -1, (0, 255, 0), 3)
    show_contours = frame.copy()
    show_contours[y:y+h, x:x+w] = cropped_contours
    
    # get each contours area
    dryout_min_area = w * h / dryout_area_ratio
    detected_dryout = cropped.copy()
    
    # gradient mask (from left, darken)
    gradient_mask = range(1, w + 1)
    base = int(w * left_darken_gradient_mask_rate)
    gradient_mask = np.minimum(np.emath.logn(base, gradient_mask), 1)
    for i in range(3):
        cropped[:,:,i] = cropped[:,:,i] * gradient_mask
    
    # gradient mask (from top, lighten)
    if top_brighten_gradient_mask_rate > 0:
        gradient_mask = np.array(range(1, h + 1))
        gradient_mask = np.float_power(gradient_mask, -1)
        base = int(h * top_brighten_gradient_mask_rate)
        gradient_mask = np.maximum(np.emath.logn(base, gradient_mask), -1) + 1

        mask = np.repeat(np.tile(gradient_mask, (w, 1))[:, :, np.newaxis], 3, axis=2)
        mask = np.rot90(mask, k = 3) * 255
        mask = mask.astype(np.uint8)
        cropped = cv2.addWeighted(cropped, 1, mask, 1, 0)

    # gradient mask (from bottom, lighten)
    if bottom_brighten_gradient_mask_rate > 0:
        gradient_mask = np.array(range(1, h + 1))
        gradient_mask = np.float_power(gradient_mask, -1)
        base = int(h * bottom_brighten_gradient_mask_rate)
        gradient_mask = np.maximum(np.emath.logn(base, gradient_mask), -1) + 1

        mask = np.repeat(np.tile(gradient_mask, (w, 1))[:, :, np.newaxis], 3, axis=2)
        mask = np.rot90(mask, k = 1) * 255
        mask = mask.astype(np.uint8)
        cropped = cv2.addWeighted(cropped, 1, mask, 1, 0)
    

    areas = 0
    mask = cropped.copy() * 0
    for c in contours:
        if cv2.contourArea(c) > dryout_min_area:
            temp = cropped.copy() * 0
            cv2.drawContours(temp, [c], -1, (255,255,255), cv2.FILLED)
            # show_image(temp)
            temp[temp == 255] = 1
            
            temp_crop = cropped.copy()
            temp_crop *= temp
            
            non_zero = temp_crop[np.where(temp_crop!=0)]
            # mean = np.sum(non_zero) / len(non_zero)
            quantile = np.quantile(non_zero, dryout_area_quantile / 100)
            if dryout_min_px <= quantile <= dryout_max_px:
                # print(quantile)
                mask += temp
                areas += 1
    
    mask[mask > 0] = 1
    total_pixels = np.shape(mask)[0] * np.shape(mask)[1]
    covered = np.sum(mask[:, :, 0])
    
    mask[:, :, 0] *= 255
    detected_dryout[:, :, 0] = mask[:, :, 0]
    # detected_dryout *= mask
    
    ada = cv2.cvtColor(ada,cv2.COLOR_GRAY2RGB)
    high_contrast[y:y+h, x:x+w] = ada
    
    blur = cv2.cvtColor(blur,cv2.COLOR_GRAY2RGB)
    blurred[y:y+h, x:x+w] = blur
    gradient_img[y:y+h, x:x+w] = cropped

    img = frame.copy()
    img[y:y+h, x:x+w] = detected_dryout
    
    row1 = cv2.hconcat([frame, blurred])
    row2 = cv2.hconcat([high_contrast, img])
    out_img = cv2.vconcat([row1, row2])
    
    return covered / total_pixels, areas, out_img.copy()
        
    
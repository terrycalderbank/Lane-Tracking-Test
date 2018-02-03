import numpy as np
import cv2

def get_line_params(x1,y1,x2,y2):
    #-----------------------
    #calculate slope and
    #intercept from two input
    #points
    #-----------------------
    if x1-x2 != 0:
        slope = (y1-y2)/(x1-x2)
        intercept = (y2-slope*x2)
        return slope, intercept
    else:
        return 0
    #y = mx + b, b = (y-mx)


#intitializing loop line variables for averaging  
Lslope = 0
Rslope = 0
Lintercept = 0
Rintercept = 0
Lslope_prev = 0
Lintercept_prev = 0
Rslope_prev = 0
Rintercept_prev = 0

#load image/video from file
#input_image = cv2.imread('highway.jpg',0)
cap = cv2.VideoCapture('highway3.mp4')

while(cap.isOpened()):
    #load one frame
    ret, input_image = cap.read()

    #scale image down using interpolation for best results
    input_image = cv2.resize(input_image,None, fx=0.60, fy=0.60, interpolation = cv2.INTER_AREA)
    grey = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
    
    #auto threshold detection for canny edge detection
    #get median value of input
    v = np.median(grey)
    
    sigma = 0.6
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    #canny edge detector with lower and upper tresholds
    edged = cv2.Canny(grey, lower, upper)
    #edged = cv2.Canny(input_image, 190, 200)
     
    #defining region of interest polygon
    #create blank mask
    mask = np.zeros_like(edged)

    #get image size parameters
    height, width = edged.shape[:2]

    #define polygon to use in mask
    #percentages from bottom
    top = 25
    side = 10
    top_width = 30
    
    topval = (1-(top/100))*height
    sideval = (1-(side/100))*height
    width_val = (top_width/100)
    
    vertices = np.array([[0,height],
                        [0, sideval],
                        [width*width_val,topval],
                        [width*(1-width_val),topval],
                        [width,sideval],
                        [width,height]],np.int32)


    #choosing mask colour based on image colour dimensions
    if len(edged.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #fill polygon in blank mask
    cv2.fillPoly(mask, [vertices], ignore_mask_color)

    #apply mask to edge detected image
    masked_image = cv2.bitwise_and(edged, mask)
    cv2.imshow('masked', masked_image)

    #blur image to reduce high frequency noise
    masked_image = cv2.GaussianBlur(masked_image,(3,3),0)
    #cv2.imshow('blurred', masked_image)
    

    #Hough line detection with variables
    minLineLength = 10
    maxLineGap = 60
    lines = cv2.HoughLinesP(masked_image,3,np.pi/180,180,minLineLength,maxLineGap)    
    
    Lslopes = []
    Lintercepts = []

    Rslopes = []
    Rintercepts = []

    #this variable filters out horizontal lines detected
    slope_threshold = 0.2
    
    try:
        #iterate through each detected line
        for line in lines:
            coords = line[0]
            x1 = coords[0]
            y1 = coords[1]
            x2 = coords[2]
            y2 = coords[3]
            slope, intercept = get_line_params(x1,y1,x2,y2)

            #sort lines into left and right
            if slope < -slope_threshold:
                Lslopes.append(slope)
                Lintercepts.append(intercept)
            if slope > slope_threshold:
                Rslopes.append(slope)
                Rintercepts.append(intercept)
    except:
        pass

    #take averages of detected slopes and draw new averaged left/right lines
    if len(Lslopes):
        Lslope = sum(Lslopes)/len(Lslopes)
        Lintercept = sum(Lintercepts)/len(Lintercepts)

        Lslope = (Lslope + Lslope_prev)/2
        Lintercept = (0.6*Lintercept + 0.4*Lintercept_prev)
        
        lx1 = int((height - Lintercept)/Lslope)
        ly1 = height
        lx2 = int(((topval)-Lintercept)/Lslope)
        ly2 = int((topval))
        cv2.line(input_image,(lx1,ly1),(lx2,ly2),(0,255,0),5)
        
    if len(Rslopes):
        Rslope = sum(Rslopes)/len(Rslopes)
        Rintercept = sum(Rintercepts)/len(Rintercepts)

        Rslope = (Rslope + Rslope_prev)/2
        Rintercept = (0.6*Rintercept + 0.4*Rintercept_prev)

        rx1 = int((height - Rintercept)/Rslope)
        ry1 = height
        rx2 = int(((topval)-Rintercept)/Rslope)
        ry2 = int((topval))
        cv2.line(input_image,(rx1,ry1),(rx2,ry2),(0,255,0),5)
        

    Lslope_prev = Lslope
    Lintercept_prev = Lintercept
    Rslope_prev = Rslope
    Rintercept_prev = Rintercept
    

    cv2.imshow('with lines',np.hstack([input_image]))
    #cv2.imshow('processing', np.hstack([grey,masked_image]))
    
    

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


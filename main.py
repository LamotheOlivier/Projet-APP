import numpy as np
import matplotlib 
import cv2 as cv
from PIL import Image, ImageDraw, ImageTk
from scipy.interpolate import BSpline, make_interp_spline

def interpolate(image, x_point, y_point):
    x_np = np.array(x_point)
    y_np = np.array(y_point)
    perm = np.argsort(x_np)
    
    x_np = x_np[perm]
    y_np = y_np[perm]
    f = make_interp_spline(x_np,y_np)
    x = np.linspace(x_np[0], x_np[-1], 100)
    y = f(x)
    for j in range(len(x)-1):
        cv.line(image, (x[j], y[j]) , (x[j+1], y[j+1]), (0,0,0))

    return f

def keep_close(canvas, spline):
    print(f"heigh {canvas.height}")

def draw_point(x,y, x_point, y_point, image):
    if len(x_point) <5:
        cv.circle(image,(x,y),5, (0,0,0))
        x_point.append(x)
        y_point.append(y)

        if len(x_point) == 5:
            spline = interpolate(image, x_point, y_point)
            keep_close(image, spline)

            

def init():
    x_point = []
    y_point = []

    # Define the callback function that will be called on mouse events
    def mouse_callback(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print(f"Left button clicked at ({x}, {y})")
        elif event == cv.EVENT_RBUTTONDOWN:
            print(f"Right button clicked at ({x}, {y})")

    # Create a blank image
    image = cv.imread("../../Textures_2024/Textures/texture3_masque.png")

    # Create a window
    cv.namedWindow('Window')

    # Set the mouse callback function for the window
    cv.setMouseCallback('Window', lambda event, x, y, flags, param : draw_point(x,y, x_point, y_point, image) )

    # Display the image in the window
    cv.imshow('Window', image)

    # im = Image.new("RGB", image_size, background_color)
    # for i in range(0, width, resize):
    #     for j in range(0, width, resize):
    #         canvas.create_image(i,j,image=im)

    cv.waitKey(0)
    cv.destroyAllWindows()



if __name__ == "__main__":
    init()
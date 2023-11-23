import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
cap = cv2.VideoCapture(0)
# To use video example
# cap = cv2.VideoCapture('Pictures/Test/Video4.mp4')

# for resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
upper_left = (200, 200)
bottom_right = (800, 500)
# Creating a loop to capture each frame of the video in the name of Img
while True:
    _, img = cap.read()  # Capture webcam footage
    rect_img = img[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]

    r = cv2.rectangle(img, upper_left, bottom_right, (100, 50, 200), 5)


    # Converting to grey scale
    gray_img = cv2.cvtColor(rect_img, cv2.COLOR_BGR2GRAY)

    # Allowing multiple face detection
    #//plate = cascade.
    faces = cascade.detectMultiScale(gray_img, 1.1, 20, 1)

    # Creating Rectangle around plates
    for (x, y, w, h) in faces:
        cv2.rectangle(rect_img, (x, y), (x + w, y + h), (0, 0, 250), 2)
        plate = img[(x, y), (x + w, y + h)]
        plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('Detected', plate)
        print(pytesseract.image_to_string(plate))

    #Add sliced part back in
    img[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]] = rect_img

    # Displaying the image
    cv2.imshow('Detected', img)

    # Waiting for escape key for image to close adding the break statement to end the face detection screen
    k = cv2.waitKey(30) & 0xff
    if k == 81 or k == 113:  # Press q to close the program
        break

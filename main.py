import cv2

cap = cv2.VideoCapture(0)
# To use video example
# cap = cv2.VideoCapture('Pictures/Test/Video4.mp4')

# for resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Creating a loop to capture each frame of the video in the name of Img
while True:
    _, img = cap.read()  # Capture webcam footage

    # Converting to grey scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Allowing multiple face detection
    faces = cascade.detectMultiScale(gray_img, 1.1, 6)

    # Creating Rectangle around face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 250), 2)

    # Displaying the image
    cv2.imshow('Detected', img)

    #number_plate = gray_img[y:y + h, x:x + w]
    cv2.imshow('Detected', img)
    # Waiting for escape key for image to close adding the break statement to end the face detection screen
    k = cv2.waitKey(30) & 0xff
    if k == 81 or k == 113:  # Press q to close the program
        break

import cv2
import pytesseract
from collections import Counter
import datetime


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
stringList = []
while True:
    rect_img = None
    start = datetime.datetime.now()

    _, img = cap.read()
    # img = cv2.flip(img, 1)
    # img = cv2.resize(img, (1024, 768))

    # Slice feed and put border around it
    sliced = img[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]
    r = cv2.rectangle(img, upper_left, bottom_right, (100, 50, 200), 5)

    # Converting the image to gray scale
    gray = cv2.cvtColor(sliced, cv2.COLOR_BGR2GRAY)
    Gaussian = cv2.GaussianBlur(gray, (7, 7), 0)

    # Detecting the plate number
    number_plates = cascade.detectMultiScale(Gaussian, 1.1, 4)


    # Drawing a rectangle around the plate number
    for (x, y, w, h) in number_plates:
        cv2.rectangle(sliced, (x, y), (x + w, y + h), (255, 0, 0), thickness=3)
        rect_img = sliced[y: y + h, x: x + w]
        gray = cv2.cvtColor(rect_img, cv2.COLOR_BGR2GRAY)
        #Gaussian2 = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Extract text via tesseract
        text = pytesseract.image_to_string(thresh, lang='eng', config="-c tessedit_char_whitelist=01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6")
        # 01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz All letters and numbers

        text = text.replace("|", "")
        text = text.replace("/", " ")
        text = text.replace("(", "")
        text = text.replace(")", "")

        # Remove whitespaces
        text = text.replace("\n", "")
        text = text.replace(" ", "")
        # Only take first 9 characters
        text = text[0:9]

        if text != "":
            # print("Test :", text)
            stringList.append(text)

            #wait till 10 samples are extracted
            if len(stringList) > 8:
                c = Counter(stringList)
                test = c.most_common(1)
                print(test[0][0])
                # Reset counter
                stringList = []
    # Set end time
    end = datetime.datetime.now()
    # Calculate the time it took to process one frame
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"

    # Reset list of nothing it present
    # if rect_img is None:
    #    stringList = []

    # Add fps counter in img
    cv2.putText(img, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
    cv2.imshow('Detected', img)

    # Waiting for escape key for image to close adding the break statement to end the face detection screen
    k = cv2.waitKey(30) & 0xff
    if k == 81 or k == 113:  # Press q to close the program
        break

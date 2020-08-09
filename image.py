import cv2

#our image
img_file = 'car.jpg'

#our pre-trained car classifier
classifier_file = 'car.xml'

#create opencv image
img = cv2.imread(img_file)

#create classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#convert to grayscale (needed for haar cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect cars
cars = car_tracker.detectMultiScale(black_n_white)

#draw rectangles around cars
for(x, y, w, h) in cars:
  cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#disply the image with faces spotted
cv2.imshow('Display image', img)

#don't autoclose (wait here in the code and listen for a key press)
cv2.waitKey()

print("Code Completed")
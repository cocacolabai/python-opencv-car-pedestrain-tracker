import cv2

#our video
video = cv2.VideoCapture('video.mp4')

#our pre-trained car classifier
classifier_file = 'car.xml'
classifier_file_2 = 'pedestrain.xml'

#create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)
pedestrain_tracker = cv2.CascadeClassifier(classifier_file_2)

#Run
while True:

  #Read the current frame
  (read_successful, frame) = video.read()

  #safe coding
  if read_successful:
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  else:
    break

  #detect cars
  cars = car_tracker.detectMultiScale(grayscaled_frame)
  #detect pedestrains
  pedestrains = pedestrain_tracker.detectMultiScale(grayscaled_frame)

  #draw rectangles around the cars
  for(x, y, w, h) in cars:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

  #draw rectangles around the pedestrains
  for(x, y, w, h) in pedestrains:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

  #display the frame
  cv2.imshow('Display', frame)

  #dont autoclose
  key = cv2.waitKey(1)

  #stop if Q KEY Pressed
  if key==81 or key==113:
    break


#release video capture object
video.release()

#end of code
print("Code Completed")
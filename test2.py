import face_recognition
import cv2
import numpy as np
import time
import requests 


URL_CONTROLLER = "http://127.0.0.1:5000/"

# your Serial port should be different!

video_capture = cv2.VideoCapture(0)

obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

gabriel_image = face_recognition.load_image_file("gabriel.jpg")
gabriel_face_encoding = face_recognition.face_encodings(gabriel_image)[0]

diogo_image = face_recognition.load_image_file("diogo.jpeg")
diogo_face_encoding = face_recognition.face_encodings(diogo_image)[0]

mozar_image = face_recognition.load_image_file("mozar.jpeg")
mozar_face_encoding = face_recognition.face_encodings(mozar_image)[0]


joao_image = face_recognition.load_image_file("joao.jpeg")
joao_face_encoding = face_recognition.face_encodings(joao_image)[0]

fabio_image = face_recognition.load_image_file("fabio.jpeg")
fabio_face_encoding = face_recognition.face_encodings(fabio_image)[0]

#yuri_image = face_recognition.load_image_file("yuri.jpg")
#yuri_face_encoding = face_recognition.face_encodings(yuri_image)[0]



#yoda_image = face_recognition.load_image_file("yoda.jpg")
#yoda_face_encoding = face_recognition.face_encodings(yoda_image)[0]

known_face_encodings = [
    obama_face_encoding,
    gabriel_face_encoding,
    diogo_face_encoding,
    mozar_face_encoding,
    joao_face_encoding,
    fabio_face_encoding,
   # yuri_face_encoding

 #   yoda_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Gabriel Henriques",
    "Diogo",
    "Mozar",
    "Joao",
    "Fábio Borges",
  #  "Yuri"

]

face_locations = []
face_encodings = []
face_names = []


 
while True:
    process_this_frame = True

    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                #TODO get request para API controller
                # r = requests.get(url = URL_CONTROLLER, params = {'command': "on"}) 
                # if(r.status_code == 200):
                #    return
            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
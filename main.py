import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time

class AttendanceSystem:
    def __init__(self, training_images_path='Training_images'):
        self.path = training_images_path
        self.images = []
        self.class_names = []
        self.encode_list_known = []
        self.last_detection_time = {}
        self.current_faces = {}
        self.face_id_counter = 0
        
        self.load_training_images()
        self.encode_known_faces()

    def load_training_images(self):
        my_list = os.listdir(self.path)
        for cl in my_list:
            cur_img = cv2.imread(f'{self.path}/{cl}')
            self.images.append(cur_img)
            self.class_names.append(os.path.splitext(cl)[0])
        print(f"Loaded {len(self.class_names)} images: {self.class_names}")

    def encode_known_faces(self):
        for img in self.images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            self.encode_list_known.append(encode)
        print("Encoding of known faces completed")

    def get_attendance_file(self):
        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"Attendance_{today}.csv"
        return filename

    def mark_attendance(self, name):
        filename = self.get_attendance_file()
        file_exists = os.path.isfile(filename)
        
        with open(filename, 'a+') as f:
            if not file_exists:
                f.write('Name,Time\n')
                print(f"Created new {filename} file with header.")
            
            f.seek(0)
            data_list = f.readlines()
            name_list = [line.split(',')[0] for line in data_list]
            
            if name not in name_list:
                now = datetime.now()
                dt_string = now.strftime('%H:%M:%S')
                f.write(f'{name},{dt_string}\n')
                print(f"Marked attendance for {name} at {dt_string}")
            else:
                print(f"{name} already marked for attendance today.")

    def run(self):
        cap = cv2.VideoCapture('http://10.0.0.190:8080/video')
        
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to grab frame")
                break
            
            img_s = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
            
            faces_cur_frame = face_recognition.face_locations(img_s)
            encodes_cur_frame = face_recognition.face_encodings(img_s, faces_cur_frame)
            
            # Reset current faces for this frame
            current_frame_faces = {}
            
            for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
                matches = face_recognition.compare_faces(self.encode_list_known, encode_face)
                face_dis = face_recognition.face_distance(self.encode_list_known, encode_face)
                match_index = np.argmin(face_dis)
                
                if matches[match_index]:
                    name = self.class_names[match_index].upper()
                    y1, x2, y2, x1 = [coord * 4 for coord in face_loc]
                    
                    # Check if this face was already detected
                    face_id = None
                    for id, (prev_encode, _, prev_name) in self.current_faces.items():
                        if face_recognition.compare_faces([prev_encode], encode_face)[0]:
                            face_id = id
                            break
                    
                    if face_id is None:
                        face_id = self.face_id_counter
                        self.face_id_counter += 1
                    
                    current_frame_faces[face_id] = (encode_face, (x1, y1, x2, y2), name)
                    
                    current_time = time.time()
                    if name not in self.last_detection_time or (current_time - self.last_detection_time[name]) > 5:
                        self.mark_attendance(name)
                        self.last_detection_time[name] = current_time
            
            # Update current_faces with the faces detected in this frame
            self.current_faces = current_frame_faces
            
            # Draw boxes for all faces in current_faces
            for face_id, (_, (x1, y1, x2, y2), name) in self.current_faces.items():
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Attendance System', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    attendance_system = AttendanceSystem()
attendance_system.run()                                                                                  
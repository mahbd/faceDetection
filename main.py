import json
import tkinter
from tkinter import filedialog

import face_recognition
import cv2
import os

info = {}
json_data_file_path = "12batchCSEFaceData.json"


def load_previous_data():
    global info, json_data_file_path
    print("Select an json file where data is saved")
    root = tkinter.Tk()
    root.withdraw()
    json_data_file_path = filedialog.askopenfilename()
    with open('12batchCSEFaceData.json', 'r') as file:
        info = json.loads(file.read())


def save_data():
    with open(json_data_file_path, 'w+') as face:
        face.write(json.dumps(info))


def find_best_match(face_encoding):
    result = ('none', [], 1)
    for name, face_encodings in info.items():
        compare_values = face_recognition.face_distance(face_encodings, face_encoding)
        for index, value in enumerate(compare_values):
            if value < result[2]:
                result = (name, index, value)
    return result


def recognize_mark_face(image_file):
    image = face_recognition.load_image_file(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image)
    for index, location in enumerate(locations):
        image_show = cv2.copyMakeBorder(image, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
        best_match = find_best_match(face_encodings[index])
        cv2.rectangle(image_show, (location[3], location[0]), (location[1], location[2]), (255, 0, 255), 3)
        cv2.imshow("Image", image_show)
        cv2.waitKey(100)
        if best_match[2] < 0.6:
            name = input(f'Is it {best_match[0]} || pos={round(best_match[2], 3)}?: ')
            if not name:  # Guess is correct if user doesn't change name
                name = best_match[0]
        else:
            name = input("Enter name: ")
        if name == 's':  # Skip current face
            continue
        elif name == 'b':  # Skip current image
            break
        elif name == 'q':  # Stop training
            return 1
        elif info.get(name):
            info[name].append(list(face_encodings[index]))
        else:
            info[name] = [list(face_encodings[index])]


def recognize_write_name(image_path):
    image = face_recognition.load_image_file(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image)
    for index, l in enumerate(locations):
        matched = find_best_match(face_encodings[index])
        txt_loc = (l[3] - 20, l[2] + 30)
        color = (255, 255, 255)
        cv2.rectangle(image, (l[3] - 20, l[0] - 20), (l[1] + 20, l[2] + 20), (255, 0, 255), 2)
        cv2.rectangle(image, (l[3] - 20, l[2] + 20), (l[1] + 20, l[2] + 40), (0, 0, 0), cv2.FILLED)
        # Write text
        cv2.putText(image, str(round(matched[2], 2)), (l[3] - 20, l[0] - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
        if matched[2] < 0.4:
            cv2.putText(image, matched[0], txt_loc, cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
        elif matched[2] < 0.6:
            cv2.putText(image, "Probably " + matched[0], txt_loc, cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
        else:
            cv2.putText(image, "Unknown", txt_loc, cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
    cv2.imshow('Image', image)
    cv2.waitKey(500)
    yn = input('q to quite. Enter to test more')
    if yn == 'q':
        return 1


def train_model():
    root = tkinter.Tk()
    root.withdraw()
    yn = input("Do you want to use all image of a folder?(y/n): ")
    if yn == 'y':
        folder_path = os.path.relpath(filedialog.askdirectory(), ".")
        images = os.listdir(folder_path)
        for im in images:
            if recognize_mark_face(os.path.join(folder_path, im)):
                break
    else:
        while True:
            image_path = os.path.relpath(filedialog.askopenfilename(), ".")
            if recognize_mark_face(image_path):
                break
    save_data()


def test_face():
    root = tkinter.Tk()
    root.withdraw()
    yn = input("Do you want to test all image of a folder?(y/n): ")
    if yn == 'y':
        folder_path = os.path.relpath(filedialog.askdirectory(), ".")
        images = os.listdir(folder_path)
        for im in images:
            if recognize_write_name(os.path.join(folder_path, im)):
                break
    else:
        while True:
            image_path = os.path.relpath(filedialog.askopenfilename(), ".")
            if recognize_write_name(image_path):
                break


def show_data_amount():
    load_previous_data()
    for name, data in info.items():
        print(f'{name}\t{len(data)}')


def main():
    yn = input("Do you want to load previous data?(y/n): ")
    if yn == 'y':
        load_previous_data()
        yn2 = input("Do you want to add more data?(y/n): ")
        if yn2 == 'y':
            train_model()
    else:
        train_model()
    show_data_amount()
    test_face()


main()

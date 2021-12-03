# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'app.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

class Ui_MainWindow():
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def setupUi(self, MainWindow):
        self.font = "UD デジタル 教科書体 NK-B"
        self.selectedProgram = 0

        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(QtCore.QSize(870, 540))
        font = QtGui.QFont()
        font.setFamily(self.font)
        font.setPointSize(12)
        MainWindow.setFont(font)
        icon = QtGui.QIcon()  # setting icon image
        icon.addPixmap(QtGui.QPixmap("ui/resources/logo.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("MainWindow{color: rgb(254, 254, 254); background-color: rgb(53, 53, 53);}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # self.setStyleSheet("background-color: rgb(0, 255, 127);color: rgb(61, 61, 61);")

        self.image_label = QLabel(self)
        self.image_label.setGeometry(QtCore.QRect(30, 20, 640, 480))

        self.vbox = QVBoxLayout()

        self.startBtn = QtWidgets.QPushButton(self.centralwidget)
        self.startBtn.setGeometry(QtCore.QRect(690, 230, 141, 51))
        font = QtGui.QFont()
        font.setFamily(self.font)
        self.startBtn.setFont(font)
        self.startBtn.setObjectName("startBtn")
        # self.startBtn.setStyleSheet(style.setBtnStyle())
        self.startBtn.setEnabled(False)
        self.startBtn.clicked.connect(self.appStart)

        self.exitBtn = QtWidgets.QPushButton(self.centralwidget)
        self.exitBtn.setGeometry(QtCore.QRect(690, 410, 141, 51))
        font = QtGui.QFont()
        font.setFamily(self.font)
        self.exitBtn.setFont(font)
        self.exitBtn.setObjectName("exitBtn")
        # self.exitBtn.setStyleSheet(style.setBtnStyle())
        self.exitBtn.clicked.connect(self.appExit)

        self.stopBtn = QtWidgets.QPushButton(self.centralwidget)
        self.stopBtn.setGeometry(QtCore.QRect(690, 290, 141, 51))
        font = QtGui.QFont()
        font.setFamily(self.font)
        self.stopBtn.setFont(font)
        self.stopBtn.setObjectName("stopBtn")
        # self.stopBtn.setStyleSheet(style.setBtnStyle())
        self.stopBtn.setEnabled(False)
        self.stopBtn.clicked.connect(self.appStop)

        self.settingBtn = QtWidgets.QPushButton(self.centralwidget)
        self.settingBtn.setGeometry(QtCore.QRect(690, 350, 141, 51))
        font = QtGui.QFont()
        font.setFamily(self.font)
        self.settingBtn.setFont(font)
        self.settingBtn.setObjectName("settingBtn")
        # self.settingBtn.setStyleSheet(style.setBtnStyle())
        self.settingBtn.clicked.connect(self.appSettings)

        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(670, 30, 181, 181))
        font = QtGui.QFont()
        font.setFamily(self.font)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        # self.groupBox.setStyleSheet("background-color: rgb(166, 166, 166);")

        self.fn03RadBtn = QtWidgets.QRadioButton(self.groupBox)
        self.fn03RadBtn.setGeometry(QtCore.QRect(20, 130, 135, 40))
        font = QtGui.QFont()
        font.setFamily(self.font)
        font.setPointSize(10)
        self.fn03RadBtn.setFont(font)
        self.fn03RadBtn.setObjectName("fn03RadBtn")
        self.fn03RadBtn.toggled.connect(self.selectProgram)

        self.fn02RadBtn = QtWidgets.QRadioButton(self.groupBox)
        self.fn02RadBtn.setGeometry(QtCore.QRect(20, 80, 135, 40))
        font = QtGui.QFont()
        font.setFamily(self.font)
        font.setPointSize(10)
        self.fn02RadBtn.setFont(font)
        self.fn02RadBtn.setObjectName("fn02RadBtn")
        self.fn02RadBtn.toggled.connect(self.selectProgram)

        self.fn01RadBtn = QtWidgets.QRadioButton(self.groupBox)
        self.fn01RadBtn.setGeometry(QtCore.QRect(20, 30, 135, 40))
        font = QtGui.QFont()
        font.setFamily(self.font)
        font.setPointSize(10)
        self.fn01RadBtn.setFont(font)
        self.fn01RadBtn.setObjectName("fn01RadBtn")
        self.fn01RadBtn.toggled.connect(self.selectProgram)


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 869, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # setting style module

    def setBtnStyle(self):
        return """
             QPushButton{font-size: 16px; font-weight: bold;\
             background-color: rgb(166, 166, 166);padding-top: 0px;margin 0px;}
             QPushButton:hover {font-size: 20px; font-weight: bold; \
             background-color: rgb(160, 166, 166);padding-top: 0px;margin 0px;
             }
             """

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "卒業制作「SOSEI」"))
        self.startBtn.setText(_translate("MainWindow", "START"))
        self.exitBtn.setText(_translate("MainWindow", "EXIT"))
        self.stopBtn.setText(_translate("MainWindow", "STOP"))
        self.groupBox.setTitle(_translate("MainWindow", "Program"))
        self.fn01RadBtn.setText(_translate("MainWindow", "RL Balance check"))
        self.fn02RadBtn.setText(_translate("MainWindow", "Doze detection"))
        self.fn03RadBtn.setText(_translate("MainWindow", "Other function"))
        self.settingBtn.setText(_translate("MainWindow", "SETTINGS"))

    def selectProgram(self, value):
        radBtn = self.sender()
        if radBtn.isChecked() == True:
            if (radBtn.text() == "RL Balance check"):
                self.selectedProgram = 1
            if (radBtn.text() == "Doze detection"):
                self.selectedProgram = 2
            if (radBtn.text() == "Other function"):
                self.selectedProgram = 3
        self.startBtn.setEnabled(True)

    def appStart(self):
        if (self.selectedProgram == 1):
            self.cap = cv2.VideoCapture(0)
            self.function = Funciton01(self.cap, self.change_pixmap_signal)
        if (self.selectedProgram == 2):
            self.cap = cv2.VideoCapture(0)
            self.function = Funciton02(self.cap, self.change_pixmap_signal)

        self.vbox.addWidget(self.image_label)
        self.setLayout(self.vbox)
        self.function.change_pixmap_signal.connect(self.update_image)
        self.function.start()
        self.stopBtn.setEnabled(True)

        print(str(self.selectedProgram) + "-START")
        self.startBtn.setEnabled(False)
        self.settingBtn.setEnabled(False)

    def appStop(self):
        self.image_label.hide()
        self.cap.release()
        # self.function.terminate()
        self.settingBtn.setEnabled(True)
        print("STOP")

    def appSettings(self):
        print("settings")

    def appExit(self):
        self.close()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class Funciton01(QThread):
    def __init__(self, cap, change_pixmap_signal):
        QThread.__init__(self)
        self.cap = cap
        self.change_pixmap_signal = change_pixmap_signal

    def run(self):
        # capture from web cam
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.cap.isOpened():
                ret, frame = self.cap.read()

                # Recolor to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                landmarks = results.pose_landmarks.landmark

                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]

                shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                left_distance = np.linalg.norm(np.array(nose) - np.array(shoulder_left))
                right_distance = np.linalg.norm(np.array(nose) - np.array(shoulder_right))

                if left_distance > right_distance:
                    status = "RIGHT"
                elif left_distance < right_distance:
                    status = "LEFT"
                else:
                    status = "OK"

                cv2.putText(image, "Status:", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, status, (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
                self.change_pixmap_signal.emit(image)

class Funciton02(QThread):
    def __init__(self, cap, change_pixmap_signal):
        QThread.__init__(self)
        self.cap = cap
        self.change_pixmap_signal = change_pixmap_signal

    def run(self):
        # capture from web cam
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.cap.isOpened():
                success, image = self.cap.read()

                # Flip the image horizontally for a later selfie-view display
                # Also convert the color space from BGR to RGB
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                # To improve performance
                image.flags.writeable = False

                # Get the result
                results = face_mesh.process(image)

                # To improve performance
                image.flags.writeable = True

                # Convert the color space from RGB to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                img_h, img_w, img_c = image.shape
                face_3d = []
                face_2d = []

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        for idx, lm in enumerate(face_landmarks.landmark):
                            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                                if idx == 1:
                                    nose_2d = (lm.x * img_w, lm.y * img_h)
                                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                                x, y = int(lm.x * img_w), int(lm.y * img_h)

                                # Get the 2D Coordinates
                                face_2d.append([x, y])

                                # Get the 3D Coordinates
                                face_3d.append([x, y, lm.z])

                                # Convert it to the NumPy array
                        face_2d = np.array(face_2d, dtype=np.float64)

                        # Convert it to the NumPy array
                        face_3d = np.array(face_3d, dtype=np.float64)

                        # The camera matrix
                        focal_length = 1 * img_w

                        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                               [0, focal_length, img_w / 2],
                                               [0, 0, 1]])

                        # The Distance Matrix
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)

                        # Solve PnP
                        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                        # Get rotational matrix
                        rmat, jac = cv2.Rodrigues(rot_vec)

                        # Get angles
                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                        # Get the y rotation degree
                        x = angles[0] * 360
                        y = angles[1] * 360
                        # print(y)

                        # See where the user's head tilting
                        if y < -10:
                            text = "LOOKING LEFT"
                        elif y > 10:
                            text = "LOOKING RIGHT"
                        elif x < -10:
                            text = "LOOKING DOWN"
                        else:
                            text = "LOOKING FORWARD"

                        # Display the nose direction
                        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix,
                                                                         dist_matrix)

                        p1 = (int(nose_2d[0]), int(nose_2d[1]))
                        p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

                        cv2.line(image, p1, p2, (255, 0, 0), 2)

                        # Add the text on the image
                        cv2.putText(image, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                self.change_pixmap_signal.emit(image)


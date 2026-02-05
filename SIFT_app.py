#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys

import numpy as np

class My_App(QtWidgets.QMainWindow):

	def __init__(self):
		super(My_App, self).__init__()
		loadUi("./SIFT_app_large.ui", self)

		self._cam_id = 0
		self._cam_fps = 10
		self._is_cam_enabled = False
		self._is_template_loaded = False

		self.browse_button.clicked.connect(self.SLOT_browse_button)
		self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

		self._camera_device = cv2.VideoCapture(self._cam_id)
		self._camera_device.set(3, 320)
		self._camera_device.set(4, 240)

		# Timer used to trigger the camera
		self._timer = QtCore.QTimer(self)
		self._timer.timeout.connect(self.SLOT_query_camera)
		self._timer.setInterval(1000 / self._cam_fps)

		self._template_frame = None

	def SLOT_browse_button(self):
		dlg = QtWidgets.QFileDialog()
		dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

		if dlg.exec_():
			self.template_path = dlg.selectedFiles()[0]

		pixmap = QtGui.QPixmap(self.template_path)
		self.template_label.setPixmap(pixmap)

		try:
			self._template_frame = cv2.imread(self.template_path)
		except Exception as e:
			print(e)

		print("Loaded template image file: " + self.template_path)

	# Source: stackoverflow.com/questions/34232632/
	def convert_cv_to_pixmap(self, cv_img):
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		height, width, channel = cv_img.shape
		bytesPerLine = channel * width
		q_img = QtGui.QImage(cv_img.data, width, height, 
					bytesPerLine, QtGui.QImage.Format_RGB888)
		return QtGui.QPixmap.fromImage(q_img)

	def SLOT_query_camera(self):
		ret, cam_frame = self._camera_device.read()

		# Default (no template) - show only camera feed
		pixmap = self.convert_cv_to_pixmap(cam_frame)

		if self._template_frame is not None:
			tem_frame = self._template_frame

			#-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
			detector = cv2.SIFT_create(nfeatures=1000)
			keypoints1, descriptors1 = detector.detectAndCompute(cam_frame, None)
			keypoints2, descriptors2 = detector.detectAndCompute(tem_frame, None)
			#-- Step 2: Matching descriptor vectors with a FLANN based matcher
			matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
			knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
			#-- Filter matches using the Lowe's ratio test
			ratio_thresh = 0.5
			good_matches = []
			for m,n in knn_matches:
				if m.distance < ratio_thresh * n.distance:
					good_matches.append(m)
			#-- Draw matches
			img_matches = np.empty((max(cam_frame.shape[0], tem_frame.shape[0]), cam_frame.shape[1]+tem_frame.shape[1], 3), dtype=np.uint8)
			cv2.drawMatches(cam_frame, keypoints1, tem_frame, keypoints2, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

			pixmap = self.convert_cv_to_pixmap(img_matches)
		else:
			print("Template image not selected!")

		self.live_image_label.setPixmap(pixmap)

	def SLOT_toggle_camera(self):
		if self._is_cam_enabled:
			self._timer.stop()
			self._is_cam_enabled = False
			self.toggle_cam_button.setText("&Enable camera")
		else:
			self._timer.start()
			self._is_cam_enabled = True
			self.toggle_cam_button.setText("&Disable camera")

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = My_App()
	myApp.show()
	sys.exit(app.exec_())

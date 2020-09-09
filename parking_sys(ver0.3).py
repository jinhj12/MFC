import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QDateTime, Qt
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtWidgets import QMainWindow, QAction
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt  # 그래프
import pytesseract
import glob
import re

###########################전역변수부################################
CLASSES = []
idx = 0
size_car = False
size_val = 0.65
InNum = []
OutNum = []
CarNum = []
InTime = []
TotalTime = ''
InTimeDB = []
TotalFee = 0
searchText = ''
numBlank = 0
availableCar = 10
filename = ''  # 입,출차파일 경로
sFilename = 'C:\\images\\mp4\\ca1.mp4'  # 주차장조회 파일 경로


class MyApp(QMainWindow):
    #################################함수 선언부##########################
    def CarPlate(self):
        global InNum, OutNum, CarNum, InTime, TotalTime, InTimeDB, TotalFee
        plt.style.use('dark_background')
        ############차 이미지 가져오기
        img_car = cv2.imread('car.jpg')
        height, width, channel = img_car.shape  # 사진 크기

        ############그레이스케일 변환
        gray = cv2.cvtColor(img_car, cv2.COLOR_BGR2GRAY)

        ############Maximize Contrast 대비
        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)  # Opeining과 원본 이미지의 차이
        imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)  # Closing과 원본 이미지의 차이

        imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
        gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

        ###########노이즈 제거 블러 > thereshold
        img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

        img_thresh = cv2.adaptiveThreshold(
            img_blurred,
            maxValue=255.0,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=19,
            C=9)

        # 윤곽선 검출 temp_result
        contours, _ = cv2.findContours(
            img_thresh,
            mode=cv2.RETR_LIST,
            method=cv2.CHAIN_APPROX_SIMPLE
        )

        #####윤곽선을 감싸는 사각형 그리기
        temp_result = np.zeros((height, width, channel), dtype=np.uint8)
        contours_dict = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2)

            # insert to dict
            contours_dict.append({
                'contour': contour,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'cx': x + (w / 2),
                'cy': y + (h / 2)
            })

        # 번호판 후보 찾기
        MIN_AREA = 80
        MIN_WIDTH, MIN_HEIGHT = 2, 8
        MIN_RATIO, MAX_RATIO = 0.25, 1.0

        possible_contours = []

        cnt = 0
        for d in contours_dict:
            area = d['w'] * d['h']
            ratio = d['w'] / d['h']

            if area > MIN_AREA \
                    and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
                    and MIN_RATIO < ratio < MAX_RATIO:
                d['idx'] = cnt
                cnt += 1
                possible_contours.append(d)

        # 윤곽선의 배열로 최종 후보 선정
        MAX_DIAG_MULTIPLYER = 5  # 번호사이 간격
        MAX_ANGLE_DIFF = 12.0  # 번호 사이 각도
        MAX_AREA_DIFF = 0.5  # 번호 면적 차이
        MAX_WIDTH_DIFF = 0.8  # 너비차이
        MAX_HEIGHT_DIFF = 0.2  # 높이차이
        MIN_N_MATCHED = 3  # 위의 5조건 만족하는 최소 개수

        # recursive 방식으로 찾기
        def find_chars(contour_list):
            matched_result_idx = []  # 최종적으로 남는 후보의 인덱스 저장

            for d1 in contour_list:
                matched_contours_idx = []
                for d2 in contour_list:
                    if d1['idx'] == d2['idx']:
                        continue

                    dx = abs(d1['cx'] - d2['cx'])
                    dy = abs(d1['cy'] - d2['cy'])
                    #
                    diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
                    # 윤곽선 중심과 중심 사이의 거리 구하기
                    distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                    if dx == 0:
                        angle_diff = 90  ##윤곽선 사이 각
                    else:
                        angle_diff = np.degrees(np.arctan(dy / dx))

                    area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                    width_diff = abs(d1['w'] - d2['w']) / d1['w']
                    height_diff = abs(d1['h'] - d2['h']) / d1['h']

                    if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                        matched_contours_idx.append(d2['idx'])

                # append this contour
                matched_contours_idx.append(d1['idx'])

                if len(matched_contours_idx) < MIN_N_MATCHED:
                    continue

                matched_result_idx.append(matched_contours_idx)

                unmatched_contour_idx = []
                for d4 in contour_list:
                    if d4['idx'] not in matched_contours_idx:
                        unmatched_contour_idx.append(d4['idx'])

                unmatched_contour = np.take(possible_contours, unmatched_contour_idx)  # 인덱스 같은 값만 추출

                # recursive
                recursive_contour_list = find_chars(unmatched_contour)

                for idx in recursive_contour_list:
                    matched_result_idx.append(idx)

                break

            return matched_result_idx

        result_idx = find_chars(possible_contours)

        matched_result = []
        for idx_list in result_idx:
            matched_result.append(np.take(possible_contours, idx_list))

        # 기울어진 이미지 회전
        PLATE_WIDTH_PADDING = 1.3  # 1.3
        PLATE_HEIGHT_PADDING = 1.5  # 1.5
        MIN_PLATE_RATIO = 3
        MAX_PLATE_RATIO = 10

        plate_imgs = []
        plate_infos = []

        for i, matched_chars in enumerate(matched_result):
            sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

            plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
            plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

            plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

            sum_height = 0
            for d in sorted_chars:
                sum_height += d['h']

            plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

            triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
            triangle_hypotenus = np.linalg.norm(
                np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
                np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
            )

            angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

            rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

            img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))

            img_cropped = cv2.getRectSubPix(
                img_rotated,
                patchSize=(int(plate_width), int(plate_height)),
                center=(int(plate_cx), int(plate_cy))
            )

            if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / \
                    img_cropped.shape[
                        0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
                continue

            plate_imgs.append(img_cropped)
            plate_infos.append({
                'x': int(plate_cx - plate_width / 2),
                'y': int(plate_cy - plate_height / 2),
                'w': int(plate_width),
                'h': int(plate_height)
            })

        ####한번더 threshold
        longest_idx, longest_text = -1, 0
        plate_chars = []

        for i, plate_img in enumerate(plate_imgs):
            plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
            _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

            plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
            plate_max_x, plate_max_y = 0, 0

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                area = w * h
                ratio = w / h
                if area > MIN_AREA \
                        and w > MIN_WIDTH and h > MIN_HEIGHT \
                        and MIN_RATIO < ratio < MAX_RATIO:
                    if x < plate_min_x:
                        plate_min_x = x
                    if y < plate_min_y:
                        plate_min_y = y
                    if x + w > plate_max_x:
                        plate_max_x = x + w
                    if y + h > plate_max_y:
                        plate_max_y = y + h

            img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
            img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
            _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0,
                                          type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10,
                                            borderType=cv2.BORDER_CONSTANT,
                                            value=(0, 0, 0))

            ####글자 인식
            text = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 0')
            result_chars = ''
            has_digit = False ##숫자포함문자열

            for c in text:
                if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                    if c.isdigit():
                        has_digit = True
                    result_chars += c


            plate_chars.append(result_chars)

            if has_digit and len(result_chars) > longest_text:
                longest_idx = i

        info = plate_infos[longest_idx]
        CarNum = plate_chars[longest_idx]

        print("차량 번호 ", CarNum)
        img_out = img_car.copy()

        cv2.rectangle(img_out, pt1=(info['x'], info['y']), pt2=(info['x'] + info['w'], info['y'] + info['h']),
                      color=(255, 0, 0), thickness=2)

        CarNUmber = plt.figure(figsize=(8, 6))
        plt.imshow(img_out)

        plt.show()
        reply = QMessageBox.question(self, '차량 번호 확인', '차량 번호는 %s가 맞습니까?' % CarNum,
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            plt.close()
            cv2.destroyWindow('Car Video')

        else:
            exit()

    def CarTime(self):
        global InNum, OutNum, CarNum, InTime, TotalTime, InTimeDB, TotalFee
        inT = datetime.datetime.now()  # 현재 시간

        InTime = inT.strftime('%y:%m:%d:%H:%M:%S')  # 현재 시간 문자열로 저장
        print("입차시간", InTime)
        self.statusBar().showMessage('입차시간 : %s' % inT)

    def SaveCar(self):
        global InNum, OutNum, CarNum, InTime, TotalTime, InTimeDB, TotalFee

        f = open("DB.txt", 'a')

        f.write(InNum)
        f.write(InTime)
        f.write("\n")
        print("입차정보 저장")
        f.close()

    def OpenCar(self):
        global InNum, OutNum, CarNum, InTime, TotalTime, InTimeDB, TotalFee
        p = re.compile(CarNum)

        for i in glob.glob(r'DB.txt'):
            with open(i, 'r') as f:
                for x, y in enumerate(f.readlines(), 1):
                    m = p.findall(y)
                    if m:
                        InTimeDB = y[7:]

    def calctime(self):
        global InNum, OutNum, CarNum, InTime, TotalTime, InTimeDB, TotalFee
        inTime_split = []
        inTime_split = InTimeDB.split(':')  # y m d h m s 쪼개기.

        now = datetime.datetime.now()
        year = 2000 + int(inTime_split[0])
        inT = datetime.datetime(year, int(inTime_split[1]), int(inTime_split[2]), int(inTime_split[3]),
                                int(inTime_split[4]), int(inTime_split[5]))
        print("현재시간:", now)
        print("입차시간:", inT)
        delta = now - inT
        TotalTime = delta.seconds

    def calcFee(self):
        global InNum, OutNum, CarNum, InTime, TotalTime, InTimeDB, TotalFee

        hours = int(TotalTime / 3600)
        TotalFee = hours * 1000
        if hours == 0:
            TotalFee = 1000
        self.statusBar().showMessage('주차요금: %d원' % TotalFee)

    def InCar(self):
        global InNum, OutNum, CarNum, InTime, TotalTime, InTimeDB, TotalFee, filename
        print("IN")
        filename = 'C:\\images\\mp4\\car3.mp4'
        self.VideoToPhoto()
        self.CarPlate()
        InNum = CarNum
        self.CarTime()
        self.SaveCar()

    def OutCar(self):
        global InNum, OutNum, CarNum, InTime, TotalTime, InTimeDB, TotalFee, filename
        print("OUT")
        filename = 'C:\\images\\mp4\\car2.mp4'
        self.VideoToPhoto()
        self.CarPlate()
        OutNum = CarNum
        self.OpenCar()
        self.calctime()
        self.calcFee()

    def ssdNet(self, image):
        global CLASSES, idx, size_car, size_val
        CONF_VALUE = 0.9
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
        net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]  ##정확도
            if confidence > CONF_VALUE:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                if ((endX - startX) >= size_val * w and (endY - startY) >= size_val * h):
                    size_car = True
                else:
                    size_car = False
        return image

    def snapshot(self, f):
        cv2.imwrite('car' + '.jpg', f)

    def VideoToPhoto(self):
        global filename

        capture = cv2.VideoCapture(filename)
        s_factor = 0.5  # 화면크기 비율 (조절 가능)
        framecount = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            framecount += 1
            if framecount % 10:  ##숫자 조절 --> frame 개수 조절
                frame = cv2.resize(frame, None, fx=s_factor, fy=s_factor, interpolation=cv2.INTER_AREA)
                ##한 장짜리 ssd 딥러닝
                retImage = self.ssdNet(frame)
                ######처리된 화면 출력
                cv2.imshow('Car Video', retImage)
                cv2.moveWindow('Car Video', 500, 160)

            key = cv2.waitKey(1)  # 화면 속도 조절
            if key == 27:  # ESC키
                break

            elif CLASSES[idx] == 'car' and size_car == True:
                self.snapshot(frame)
                print("진입차량 캡처")
                break
        reply = QMessageBox.question(self, '차량 인식', '차량번호를 확인중입니다.',
                                     QMessageBox.Yes)
        capture.release()

    def payFee(self, event):
        global InNum, OutNum, CarNum, InTime, TotalTime, InTimeDB, TotalFee
        reply = QMessageBox.question(self, '주차요금 정산', '주차요금은 %s원입니다.\n 정산하시겠습니까?' % TotalFee,
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)


    def serachCar(self):
        global InNum, OutNum, CarNum, InTime, TotalTime, InTimeDB, TotalFee, searchText
        p = re.compile(searchText)

        for i in glob.glob(r'DB.txt'):
            with open(i, 'r') as f:
                for x, y in enumerate(f.readlines(), 1):
                    m = p.findall(y)
                    if m:
                        InTimeDB = y[7:]
                        cnum = y[0:7]
        print(InTimeDB)
        f.close()

        self.calctime()
        self.calcFee()
        if searchText == '':
            TotalFee = 0
            InTimeDB = ''
        print('주차요금: %s' % TotalFee)
        reply = QMessageBox.question(self, '주차요금 검색',
                                     '차번: %s\n' % cnum + '입차시간: %s' % InTimeDB + '현재 주차요금: %s원' % TotalFee,
                                     QMessageBox.Yes)

    def onChanged(self, Text):
        global searchText
        self.lbl.setText(Text)
        self.lbl.adjustSize()
        searchText = Text

    def nvr(self):
        global sFilename

        capture = cv2.VideoCapture(sFilename)
        s_factor = 0.5  # 화면크기 비율 (조절 가능)
        framecount = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            framecount += 1
            if framecount % 2:  ##숫자 조절 --> frame 개수 조절
                retImage = cv2.resize(frame, None, fx=s_factor, fy=s_factor, interpolation=cv2.INTER_AREA)
                ######처리된 화면 출력
                frame = cv2.flip(retImage, -1)
                cv2.imshow('NVR', frame)
                cv2.moveWindow('NVR', 300, 200)

            key = cv2.waitKey(20)  # 화면 속도 조절
            if key == 27:  # ESC키
                break

            elif key == ord('C') or key == ord('c'):
                self.snapshot(frame)

        capture.release()
        cv2.destroyWindow('NVR')

    def findBlank(self):
        global InNum, OutNum, CarNum, InTime, TotalTime, InTimeDB, TotalFee, searchText, numBlank

        f = open('DB.txt', 'r')
        numcar = f.read().count("\n")

        numBlank = availableCar - numcar
        f.close()

    def __init__(self):  ##무조건 실행
        global InNum, OutNum, CarNum, InTime, TotalTime, InTimeDB, TotalFee, searchText, numBlank
        super().__init__()
        self.findBlank()
        self.datetime = QDateTime.currentDateTime()
        self.initUI()  # 클래스 안에있는 자신

    def initUI(self):  # 메소드 안에 self
        global InNum, OutNum, CarNum, InTime, TotalTime, InTimeDB, TotalFee, numBlank
        exitAction = QAction(QIcon('exit.png'), 'Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('종료')
        exitAction.triggered.connect(qApp.quit)

        searchCar = QAction(QIcon('search.png'), 'search', self)
        searchCar.setShortcut('Ctrl+F')
        searchCar.setStatusTip('검색')
        searchCar.triggered.connect(self.serachCar)

        nvr = QAction(QIcon('video.png'), 'nvr', self)
        nvr.setStatusTip('실시간 영상')
        nvr.triggered.connect(self.nvr)


        self.statusBar()

        self.toolbar = self.addToolBar('search')
        self.toolbar.addAction(searchCar)

        self.toolbar = self.addToolBar('Video')
        self.toolbar.addAction(nvr)

        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(exitAction)

        self.lbl = QLabel(self)
        self.lbl.move(120, 55)

        qle = QLineEdit(self)
        qle.move(10, 50)
        qle.textChanged[str].connect(self.onChanged)

        btnIn = QPushButton('입차', self)
        btnIn.move(50, 100)
        btnIn.setCheckable(True)
        btnIn.clicked.connect(self.InCar)
        # btnIn.setStatusTip('입차')

        btnOut = QPushButton('출차', self)
        btnOut.move(50, 200)
        btnOut.setCheckable(True)
        btnOut.clicked.connect(self.OutCar)
        # btnOut.setStatusTip('출차')

        btnFee = QPushButton('주차요금 정산', self)
        btnFee.move(250, 200)
        btnFee.setCheckable(True)
        btnFee.clicked.connect(self.payFee)

        self.statusBar().showMessage(self.datetime.toString(Qt.DefaultLocaleShortDate) + ' / 빈 자리 : %d개' % numBlank)
        self.setWindowTitle('주차 관리 시스템')
        self.setGeometry(100, 200, 400, 300)
        self.setWindowIcon(QIcon('parking.png'))
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


#######################메인함수부######################
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()  # 인스턴트 객체
    sys.exit(app.exec_())

import sys
from cv2 import resize, imread, imwrite, imdecode
from numpy import ndarray, fromfile, uint8
from img_process import image_det_reg_process, model_initial
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel, QPushButton



def cv_imread(filepath):
    cv_img = imdecode(fromfile(filepath, dtype=uint8), -1)
    return cv_img


class win(QDialog):
    def __init__(self):
        self.SLPNet_model = model_initial()
        self.img_initial = ndarray(())
        self.img_for_show = ndarray(())
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(400, 400)
        self.setWindowTitle('number plate extraction')
        self.btnOpen = QPushButton('Open', self)
        self.btnSave = QPushButton('Save', self)
        self.btnProcess = QPushButton('Process', self)
        self.btnQuit = QPushButton('Quit', self)
        self.label = QLabel()

        
        layout = QGridLayout(self)
        layout.addWidget(self.label, 0, 1, 3, 4)
        layout.addWidget(self.btnOpen, 4, 1, 1, 1)
        layout.addWidget(self.btnProcess, 4, 2, 1, 1)
        layout.addWidget(self.btnSave, 4, 3, 1, 1)
        layout.addWidget(self.btnQuit, 4, 4, 1, 1)

        
        self.btnOpen.clicked.connect(self.openSlot)
        self.btnSave.clicked.connect(self.saveSlot)
        self.btnProcess.clicked.connect(self.processSlot)
        self.btnQuit.clicked.connect(self.close)

    def openSlot(self):
        max_len = 800
        fileName, tmp = QFileDialog.getOpenFileName(self, 'Open Image', 'Image', '*.png *.jpg *.bmp')
        if fileName is '':
            return
        print('Now process image: %s' % fileName.split('/')[-1])
        
        self.img_initial = imread(fileName, -1)
        if self.img_initial.size == 1:
            return
        img_h, img_w = self.img_initial.shape[:2]
        if img_h > img_w and img_h > max_len:
            tar_h = max_len
            tar_w = int(max_len * (img_w / img_h))
            self.img_for_show = resize(self.img_initial, (tar_w, tar_h))
        elif img_w > img_h and img_w > max_len:
            tar_w = max_len
            tar_h = int(max_len * (img_h / img_w))
            self.img_for_show = resize(self.img_initial, (tar_w, tar_h))
        else:
            self.img_for_show = self.img_initial
        self.refreshShow()

    def saveSlot(self):
        
        fileName, tmp = QFileDialog.getSaveFileName(self, 'Save Image', 'Image', '*.png *.jpg *.bmp')
        if fileName is '':
            return
        if self.img_for_show.size == 1:
            return
        
        imwrite(fileName, self.img_for_show)

    def processSlot(self):
        if self.img_for_show.size == 1:
            return
        
        self.img_for_show = image_det_reg_process(self.SLPNet_model, self.img_initial, self.img_for_show)
        self.refreshShow()

    def refreshShow(self):
        
        height, width, channel = self.img_for_show.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img_for_show.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        
        self.label.setPixmap(QPixmap.fromImage(self.qImg))


if __name__ == '__main__':
    a = QApplication(sys.argv)
    w = win()
    w.show()
    sys.exit(a.exec_())

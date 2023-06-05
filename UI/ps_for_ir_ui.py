# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ps_for_ir.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from MyOpenGLWidget import GLCoordinate # 这个类是姚哥自己写的子类

import ps_for_ir_rc

class Ui_PS_for_IRClass(object):
    def setupUi(self, PS_for_IRClass):
        if not PS_for_IRClass.objectName():
            PS_for_IRClass.setObjectName(u"PS_for_IRClass")
        PS_for_IRClass.resize(1304, 905)
        self.video_action = QAction(PS_for_IRClass)
        self.video_action.setObjectName(u"video_action")
        self.video_action.setCheckable(True)
        self.camera_action = QAction(PS_for_IRClass)
        self.camera_action.setObjectName(u"camera_action")
        self.camera_action.setCheckable(True)
        self.new_action = QAction(PS_for_IRClass)
        self.new_action.setObjectName(u"new_action")
        self.start_collect_action = QAction(PS_for_IRClass)
        self.start_collect_action.setObjectName(u"start_collect_action")
        self.stop_collect_action = QAction(PS_for_IRClass)
        self.stop_collect_action.setObjectName(u"stop_collect_action")
        self.optimize_action = QAction(PS_for_IRClass)
        self.optimize_action.setObjectName(u"optimize_action")
        self.image_show_action = QAction(PS_for_IRClass)
        self.image_show_action.setObjectName(u"image_show_action")
        self.image_show_action.setCheckable(True)
        self.data_show_action = QAction(PS_for_IRClass)
        self.data_show_action.setObjectName(u"data_show_action")
        self.data_show_action.setCheckable(True)
        self.visualization_action = QAction(PS_for_IRClass)
        self.visualization_action.setObjectName(u"visualization_action")
        self.visualization_action.setCheckable(True)
        self.listen_action = QAction(PS_for_IRClass)
        self.listen_action.setObjectName(u"listen_action")
        self.not_listen_action = QAction(PS_for_IRClass)
        self.not_listen_action.setObjectName(u"not_listen_action")
        self.start_action = QAction(PS_for_IRClass)
        self.start_action.setObjectName(u"start_action")
        self.stop_action = QAction(PS_for_IRClass)
        self.stop_action.setObjectName(u"stop_action")
        self.TM_action = QAction(PS_for_IRClass)
        self.TM_action.setObjectName(u"TM_action")
        self.cam_setup_action = QAction(PS_for_IRClass)
        self.cam_setup_action.setObjectName(u"cam_setup_action")
        self.clear_action = QAction(PS_for_IRClass)
        self.clear_action.setObjectName(u"clear_action")
        self.centralWidget = QWidget(PS_for_IRClass)
        self.centralWidget.setObjectName(u"centralWidget")
        self.verticalLayout = QVBoxLayout(self.centralWidget)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.img_label = QLabel(self.centralWidget)
        self.img_label.setObjectName(u"img_label")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.img_label.sizePolicy().hasHeightForWidth())
        self.img_label.setSizePolicy(sizePolicy)
        self.img_label.setMinimumSize(QSize(1280, 480))
        self.img_label.setFocusPolicy(Qt.NoFocus)
        self.img_label.setLayoutDirection(Qt.LeftToRight)
        self.img_label.setStyleSheet(u"background-color: rgb(0, 0, 0);")
        self.img_label.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.img_label)

        self.splitter = QSplitter(self.centralWidget)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.data_label = QLabel(self.splitter)
        self.data_label.setObjectName(u"data_label")
        sizePolicy.setHeightForWidth(self.data_label.sizePolicy().hasHeightForWidth())
        self.data_label.setSizePolicy(sizePolicy)
        self.data_label.setMinimumSize(QSize(415, 320))
        self.data_label.setAutoFillBackground(False)
        self.data_label.setStyleSheet(u"background-color: rgb(138, 138, 138);\n"
"color: rgb(255, 255, 255);")
        self.data_label.setAlignment(Qt.AlignCenter)
        self.splitter.addWidget(self.data_label)
        self.MyOGL = GLCoordinate(self.splitter)
        self.MyOGL.setObjectName(u"MyOGL")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.MyOGL.sizePolicy().hasHeightForWidth())
        self.MyOGL.setSizePolicy(sizePolicy1)
        self.MyOGL.setMinimumSize(QSize(500, 320))
        self.MyOGL.setStyleSheet(u"background-color: rgb(194, 194, 194);")
        self.splitter.addWidget(self.MyOGL)

        self.verticalLayout.addWidget(self.splitter)

        PS_for_IRClass.setCentralWidget(self.centralWidget)
        self.menuBar = QMenuBar(PS_for_IRClass)
        self.menuBar.setObjectName(u"menuBar")
        self.menuBar.setGeometry(QRect(0, 0, 1304, 24))
        self.menu = QMenu(self.menuBar)
        self.menu.setObjectName(u"menu")
        self.menu_2 = QMenu(self.menuBar)
        self.menu_2.setObjectName(u"menu_2")
        self.menu_4 = QMenu(self.menuBar)
        self.menu_4.setObjectName(u"menu_4")
        self.menu_5 = QMenu(self.menuBar)
        self.menu_5.setObjectName(u"menu_5")
        PS_for_IRClass.setMenuBar(self.menuBar)
        self.mainToolBar = QToolBar(PS_for_IRClass)
        self.mainToolBar.setObjectName(u"mainToolBar")
        PS_for_IRClass.addToolBar(Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QStatusBar(PS_for_IRClass)
        self.statusBar.setObjectName(u"statusBar")
        PS_for_IRClass.setStatusBar(self.statusBar)

        self.menuBar.addAction(self.menu.menuAction())
        self.menuBar.addAction(self.menu_2.menuAction())
        self.menuBar.addAction(self.menu_4.menuAction())
        self.menuBar.addAction(self.menu_5.menuAction())
        self.menu.addAction(self.video_action)
        self.menu.addSeparator()
        self.menu.addAction(self.camera_action)
        self.menu_2.addAction(self.new_action)
        self.menu_2.addAction(self.optimize_action)
        self.menu_4.addAction(self.start_collect_action)
        self.menu_4.addAction(self.stop_collect_action)
        self.menu_4.addAction(self.clear_action)
        self.menu_5.addAction(self.image_show_action)
        self.menu_5.addAction(self.data_show_action)
        self.menu_5.addAction(self.visualization_action)
        self.mainToolBar.addAction(self.start_action)
        self.mainToolBar.addAction(self.stop_action)
        self.mainToolBar.addSeparator()
        self.mainToolBar.addAction(self.TM_action)
        self.mainToolBar.addAction(self.cam_setup_action)
        self.mainToolBar.addSeparator()

        self.retranslateUi(PS_for_IRClass)

        QMetaObject.connectSlotsByName(PS_for_IRClass)
    # setupUi

    def retranslateUi(self, PS_for_IRClass):
        PS_for_IRClass.setWindowTitle(QCoreApplication.translate("PS_for_IRClass", u"PS_for_IR", None))
        self.video_action.setText(QCoreApplication.translate("PS_for_IRClass", u"\u89c6\u9891\u6e90", None))
        self.camera_action.setText(QCoreApplication.translate("PS_for_IRClass", u"\u76f8\u673a\u6e90", None))
        self.new_action.setText(QCoreApplication.translate("PS_for_IRClass", u"\u65b0\u5efa", None))
        self.start_collect_action.setText(QCoreApplication.translate("PS_for_IRClass", u"\u5f00\u59cb\u91c7\u96c6", None))
        self.stop_collect_action.setText(QCoreApplication.translate("PS_for_IRClass", u"\u7ed3\u675f\u91c7\u96c6", None))
        self.optimize_action.setText(QCoreApplication.translate("PS_for_IRClass", u"\u4f18\u5316", None))
        self.image_show_action.setText(QCoreApplication.translate("PS_for_IRClass", u"\u56fe\u50cf", None))
        self.data_show_action.setText(QCoreApplication.translate("PS_for_IRClass", u"\u7ed3\u679c", None))
        self.visualization_action.setText(QCoreApplication.translate("PS_for_IRClass", u"\u53ef\u89c6\u5316", None))
        self.listen_action.setText(QCoreApplication.translate("PS_for_IRClass", u"\u4fa6\u542c", None))
        self.not_listen_action.setText(QCoreApplication.translate("PS_for_IRClass", u"\u53d6\u6d88\u4fa6\u542c", None))
        self.start_action.setText(QCoreApplication.translate("PS_for_IRClass", u"\u5f00\u59cb", None))
        self.stop_action.setText(QCoreApplication.translate("PS_for_IRClass", u"\u505c\u6b62", None))
        self.TM_action.setText(QCoreApplication.translate("PS_for_IRClass", u"\u6a21\u677f\u7ba1\u7406\u5668", None))
        self.cam_setup_action.setText(QCoreApplication.translate("PS_for_IRClass", u"\u76f8\u673a\u8bbe\u7f6e", None))
        self.clear_action.setText(QCoreApplication.translate("PS_for_IRClass", u"\u6e05\u9664\u91c7\u96c6", None))
        self.img_label.setText(QCoreApplication.translate("PS_for_IRClass", u"TextLabel", None))
        self.data_label.setText(QCoreApplication.translate("PS_for_IRClass", u"\u5339\u914d\u7ed3\u679c", None))
        self.menu.setTitle(QCoreApplication.translate("PS_for_IRClass", u"\u8f93\u5165", None))
        self.menu_2.setTitle(QCoreApplication.translate("PS_for_IRClass", u"\u6a21\u677f", None))
        self.menu_4.setTitle(QCoreApplication.translate("PS_for_IRClass", u"\u6570\u636e", None))
        self.menu_5.setTitle(QCoreApplication.translate("PS_for_IRClass", u"\u89c6\u56fe", None))
    # retranslateUi


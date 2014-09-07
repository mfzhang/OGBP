#############################################################################
# Makefile for building: face
# Generated by qmake (3.0) (Qt 5.3.1)
# Project:  ../face/face.pro
# Template: app
# Command: /home/cyy/Qt5.3.1/5.3/gcc/bin/qmake -spec linux-g++ CONFIG+=debug -o Makefile ../face/face.pro
#############################################################################

MAKEFILE      = Makefile

####### Compiler, tools and options

CC            = gcc
CXX           = g++
DEFINES       = -DQT_CORE_LIB
CFLAGS        = -pipe -g -Wall -W -D_REENTRANT -fPIE $(DEFINES)
CXXFLAGS      = -pipe -g -Wall -W -D_REENTRANT -fPIE $(DEFINES)
INCPATH       = -I../../Qt5.3.1/5.3/gcc/mkspecs/linux-g++ -I../face -I/usr/local/include -I/usr/local/include/opencv -I/usr/local/include/opencv2 -I../../Qt5.3.1/5.3/gcc/include -I../../Qt5.3.1/5.3/gcc/include/QtCore -I. -I.
LINK          = g++
LFLAGS        = -Wl,-rpath,/home/cyy/Qt5.3.1/5.3/gcc -Wl,-rpath,/home/cyy/Qt5.3.1/5.3/gcc/lib
LIBS          = $(SUBLIBS) /usr/local/lib/libopencv_highgui.so /usr/local/lib/libopencv_core.so /usr/local/lib/libopencv_imgproc.so /usr/local/lib/libopencv_contrib.so /usr/local/lib/libopencv_objdetect.so -L/home/cyy/Qt5.3.1/5.3/gcc/lib -lQt5Core -lpthread 
AR            = ar cqs
RANLIB        = 
QMAKE         = /home/cyy/Qt5.3.1/5.3/gcc/bin/qmake
TAR           = tar -cf
COMPRESS      = gzip -9f
COPY          = cp -f
SED           = sed
COPY_FILE     = cp -f
COPY_DIR      = cp -f -R
STRIP         = strip
INSTALL_FILE  = install -m 644 -p
INSTALL_DIR   = $(COPY_DIR)
INSTALL_PROGRAM = install -m 755 -p
DEL_FILE      = rm -f
SYMLINK       = ln -f -s
DEL_DIR       = rmdir
MOVE          = mv -f
CHK_DIR_EXISTS= test -d
MKDIR         = mkdir -p

####### Output directory

OBJECTS_DIR   = ./

####### Files

SOURCES       = ../face/main.cpp \
		detectObject.cpp \
		OGBP.cpp \
		preprocessFace.cpp 
OBJECTS       = main.o \
		detectObject.o \
		OGBP.o \
		preprocessFace.o
DIST          = ../../Qt5.3.1/5.3/gcc/mkspecs/features/spec_pre.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/common/shell-unix.conf \
		../../Qt5.3.1/5.3/gcc/mkspecs/common/unix.conf \
		../../Qt5.3.1/5.3/gcc/mkspecs/common/linux.conf \
		../../Qt5.3.1/5.3/gcc/mkspecs/common/gcc-base.conf \
		../../Qt5.3.1/5.3/gcc/mkspecs/common/gcc-base-unix.conf \
		../../Qt5.3.1/5.3/gcc/mkspecs/common/g++-base.conf \
		../../Qt5.3.1/5.3/gcc/mkspecs/common/g++-unix.conf \
		../../Qt5.3.1/5.3/gcc/mkspecs/qconfig.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_bluetooth.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_bluetooth_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_bootstrap_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_clucene_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_concurrent.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_concurrent_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_core.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_core_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_dbus.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_dbus_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_declarative.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_declarative_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_designer.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_designer_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_designercomponents_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_enginio.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_enginio_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_gui.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_gui_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_help.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_help_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_multimedia.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_multimedia_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_multimediawidgets.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_multimediawidgets_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_network.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_network_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_nfc.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_nfc_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_opengl.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_opengl_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_openglextensions.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_openglextensions_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_platformsupport_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_positioning.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_positioning_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_printsupport.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_printsupport_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_qml.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_qml_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_qmldevtools_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_qmltest.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_qmltest_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_qtmultimediaquicktools_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_quick.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_quick_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_quickparticles_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_quickwidgets.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_quickwidgets_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_script.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_script_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_scripttools.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_scripttools_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_sensors.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_sensors_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_serialport.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_serialport_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_sql.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_sql_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_svg.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_svg_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_testlib.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_testlib_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_uitools.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_uitools_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_webkit.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_webkit_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_webkitwidgets.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_webkitwidgets_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_websockets.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_websockets_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_widgets.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_widgets_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_x11extras.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_x11extras_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_xml.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_xml_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_xmlpatterns.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_xmlpatterns_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/qt_functions.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/qt_config.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/linux-g++/qmake.conf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/spec_post.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/exclusive_builds.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/default_pre.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/resolve_config.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/default_post.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/warn_on.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/qt.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/resources.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/moc.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/unix/thread.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/testcase_targets.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/exceptions.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/yacc.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/lex.prf \
		../face/face.pro ../face/main.cpp \
		detectObject.cpp \
		OGBP.cpp \
		preprocessFace.cpp
QMAKE_TARGET  = face
DESTDIR       = #avoid trailing-slash linebreak
TARGET        = face


first: all
####### Implicit rules

.SUFFIXES: .o .c .cpp .cc .cxx .C

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cc.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cxx.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.C.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.c.o:
	$(CC) -c $(CFLAGS) $(INCPATH) -o "$@" "$<"

####### Build rules

all: Makefile $(TARGET)

$(TARGET):  $(OBJECTS)  
	$(LINK) $(LFLAGS) -o $(TARGET) $(OBJECTS) $(OBJCOMP) $(LIBS)

Makefile: ../face/face.pro ../../Qt5.3.1/5.3/gcc/mkspecs/linux-g++/qmake.conf ../../Qt5.3.1/5.3/gcc/mkspecs/features/spec_pre.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/common/shell-unix.conf \
		../../Qt5.3.1/5.3/gcc/mkspecs/common/unix.conf \
		../../Qt5.3.1/5.3/gcc/mkspecs/common/linux.conf \
		../../Qt5.3.1/5.3/gcc/mkspecs/common/gcc-base.conf \
		../../Qt5.3.1/5.3/gcc/mkspecs/common/gcc-base-unix.conf \
		../../Qt5.3.1/5.3/gcc/mkspecs/common/g++-base.conf \
		../../Qt5.3.1/5.3/gcc/mkspecs/common/g++-unix.conf \
		../../Qt5.3.1/5.3/gcc/mkspecs/qconfig.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_bluetooth.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_bluetooth_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_bootstrap_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_clucene_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_concurrent.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_concurrent_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_core.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_core_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_dbus.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_dbus_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_declarative.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_declarative_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_designer.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_designer_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_designercomponents_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_enginio.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_enginio_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_gui.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_gui_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_help.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_help_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_multimedia.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_multimedia_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_multimediawidgets.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_multimediawidgets_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_network.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_network_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_nfc.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_nfc_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_opengl.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_opengl_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_openglextensions.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_openglextensions_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_platformsupport_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_positioning.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_positioning_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_printsupport.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_printsupport_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_qml.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_qml_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_qmldevtools_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_qmltest.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_qmltest_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_qtmultimediaquicktools_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_quick.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_quick_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_quickparticles_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_quickwidgets.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_quickwidgets_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_script.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_script_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_scripttools.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_scripttools_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_sensors.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_sensors_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_serialport.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_serialport_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_sql.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_sql_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_svg.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_svg_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_testlib.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_testlib_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_uitools.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_uitools_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_webkit.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_webkit_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_webkitwidgets.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_webkitwidgets_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_websockets.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_websockets_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_widgets.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_widgets_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_x11extras.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_x11extras_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_xml.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_xml_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_xmlpatterns.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_xmlpatterns_private.pri \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/qt_functions.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/qt_config.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/linux-g++/qmake.conf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/spec_post.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/exclusive_builds.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/default_pre.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/resolve_config.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/default_post.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/warn_on.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/qt.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/resources.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/moc.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/unix/thread.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/testcase_targets.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/exceptions.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/yacc.prf \
		../../Qt5.3.1/5.3/gcc/mkspecs/features/lex.prf \
		../face/face.pro \
		/home/cyy/Qt5.3.1/5.3/gcc/lib/libQt5Core.prl
	$(QMAKE) -spec linux-g++ CONFIG+=debug -o Makefile ../face/face.pro
../../Qt5.3.1/5.3/gcc/mkspecs/features/spec_pre.prf:
../../Qt5.3.1/5.3/gcc/mkspecs/common/shell-unix.conf:
../../Qt5.3.1/5.3/gcc/mkspecs/common/unix.conf:
../../Qt5.3.1/5.3/gcc/mkspecs/common/linux.conf:
../../Qt5.3.1/5.3/gcc/mkspecs/common/gcc-base.conf:
../../Qt5.3.1/5.3/gcc/mkspecs/common/gcc-base-unix.conf:
../../Qt5.3.1/5.3/gcc/mkspecs/common/g++-base.conf:
../../Qt5.3.1/5.3/gcc/mkspecs/common/g++-unix.conf:
../../Qt5.3.1/5.3/gcc/mkspecs/qconfig.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_bluetooth.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_bluetooth_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_bootstrap_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_clucene_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_concurrent.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_concurrent_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_core.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_core_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_dbus.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_dbus_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_declarative.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_declarative_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_designer.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_designer_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_designercomponents_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_enginio.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_enginio_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_gui.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_gui_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_help.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_help_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_multimedia.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_multimedia_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_multimediawidgets.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_multimediawidgets_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_network.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_network_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_nfc.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_nfc_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_opengl.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_opengl_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_openglextensions.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_openglextensions_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_platformsupport_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_positioning.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_positioning_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_printsupport.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_printsupport_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_qml.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_qml_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_qmldevtools_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_qmltest.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_qmltest_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_qtmultimediaquicktools_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_quick.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_quick_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_quickparticles_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_quickwidgets.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_quickwidgets_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_script.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_script_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_scripttools.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_scripttools_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_sensors.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_sensors_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_serialport.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_serialport_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_sql.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_sql_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_svg.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_svg_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_testlib.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_testlib_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_uitools.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_uitools_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_webkit.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_webkit_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_webkitwidgets.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_webkitwidgets_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_websockets.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_websockets_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_widgets.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_widgets_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_x11extras.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_x11extras_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_xml.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_xml_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_xmlpatterns.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/modules/qt_lib_xmlpatterns_private.pri:
../../Qt5.3.1/5.3/gcc/mkspecs/features/qt_functions.prf:
../../Qt5.3.1/5.3/gcc/mkspecs/features/qt_config.prf:
../../Qt5.3.1/5.3/gcc/mkspecs/linux-g++/qmake.conf:
../../Qt5.3.1/5.3/gcc/mkspecs/features/spec_post.prf:
../../Qt5.3.1/5.3/gcc/mkspecs/features/exclusive_builds.prf:
../../Qt5.3.1/5.3/gcc/mkspecs/features/default_pre.prf:
../../Qt5.3.1/5.3/gcc/mkspecs/features/resolve_config.prf:
../../Qt5.3.1/5.3/gcc/mkspecs/features/default_post.prf:
../../Qt5.3.1/5.3/gcc/mkspecs/features/warn_on.prf:
../../Qt5.3.1/5.3/gcc/mkspecs/features/qt.prf:
../../Qt5.3.1/5.3/gcc/mkspecs/features/resources.prf:
../../Qt5.3.1/5.3/gcc/mkspecs/features/moc.prf:
../../Qt5.3.1/5.3/gcc/mkspecs/features/unix/thread.prf:
../../Qt5.3.1/5.3/gcc/mkspecs/features/testcase_targets.prf:
../../Qt5.3.1/5.3/gcc/mkspecs/features/exceptions.prf:
../../Qt5.3.1/5.3/gcc/mkspecs/features/yacc.prf:
../../Qt5.3.1/5.3/gcc/mkspecs/features/lex.prf:
../face/face.pro:
/home/cyy/Qt5.3.1/5.3/gcc/lib/libQt5Core.prl:
qmake: FORCE
	@$(QMAKE) -spec linux-g++ CONFIG+=debug -o Makefile ../face/face.pro

qmake_all: FORCE

dist: 
	@test -d .tmp/face1.0.0 || mkdir -p .tmp/face1.0.0
	$(COPY_FILE) --parents $(DIST) .tmp/face1.0.0/ && $(COPY_FILE) --parents detectObject.h OGBP.h preprocessFace.h .tmp/face1.0.0/ && $(COPY_FILE) --parents ../face/main.cpp detectObject.cpp OGBP.cpp preprocessFace.cpp .tmp/face1.0.0/ && (cd `dirname .tmp/face1.0.0` && $(TAR) face1.0.0.tar face1.0.0 && $(COMPRESS) face1.0.0.tar) && $(MOVE) `dirname .tmp/face1.0.0`/face1.0.0.tar.gz . && $(DEL_FILE) -r .tmp/face1.0.0


clean:compiler_clean 
	-$(DEL_FILE) $(OBJECTS)
	-$(DEL_FILE) *~ core *.core *.o


distclean: clean 
	-$(DEL_FILE) $(TARGET) 
	-$(DEL_FILE) Makefile


####### Sub-libraries

mocclean: compiler_moc_header_clean compiler_moc_source_clean

mocables: compiler_moc_header_make_all compiler_moc_source_make_all

check: first

compiler_rcc_make_all:
compiler_rcc_clean:
compiler_moc_header_make_all:
compiler_moc_header_clean:
compiler_moc_source_make_all:
compiler_moc_source_clean:
compiler_yacc_decl_make_all:
compiler_yacc_decl_clean:
compiler_yacc_impl_make_all:
compiler_yacc_impl_clean:
compiler_lex_make_all:
compiler_lex_clean:
compiler_clean: 

####### Compile

main.o: ../face/main.cpp /usr/local/include/opencv2/core/core.hpp \
		/usr/local/include/opencv2/core/types_c.h \
		/usr/local/include/opencv2/core/version.hpp \
		/usr/local/include/opencv2/core/operations.hpp \
		/usr/local/include/opencv2/core/mat.hpp \
		/usr/local/include/opencv2/contrib/contrib.hpp \
		/usr/local/include/opencv2/imgproc/imgproc.hpp \
		/usr/local/include/opencv2/imgproc/types_c.h \
		/usr/local/include/opencv2/core/core_c.h \
		/usr/local/include/opencv2/features2d/features2d.hpp \
		/usr/local/include/opencv2/flann/miniflann.hpp \
		/usr/local/include/opencv2/flann/defines.h \
		/usr/local/include/opencv2/flann/config.h \
		/usr/local/include/opencv2/objdetect/objdetect.hpp \
		/usr/local/include/opencv2/contrib/retina.hpp \
		/usr/local/include/opencv2/contrib/openfabmap.hpp \
		/usr/local/include/opencv2/highgui/highgui.hpp \
		/usr/local/include/opencv2/highgui/highgui_c.h \
		/usr/local/include/opencv2/opencv.hpp \
		/usr/local/include/opencv2/imgproc/imgproc_c.h \
		/usr/local/include/opencv2/photo/photo.hpp \
		/usr/local/include/opencv2/photo/photo_c.h \
		/usr/local/include/opencv2/video/video.hpp \
		/usr/local/include/opencv2/video/tracking.hpp \
		/usr/local/include/opencv2/video/background_segm.hpp \
		/usr/local/include/opencv2/calib3d/calib3d.hpp \
		/usr/local/include/opencv2/ml/ml.hpp \
		detectObject.h \
		preprocessFace.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o main.o ../face/main.cpp

detectObject.o: detectObject.cpp detectObject.h \
		/usr/local/include/opencv2/opencv.hpp \
		/usr/local/include/opencv2/core/core_c.h \
		/usr/local/include/opencv2/core/types_c.h \
		/usr/local/include/opencv2/core/core.hpp \
		/usr/local/include/opencv2/core/version.hpp \
		/usr/local/include/opencv2/core/operations.hpp \
		/usr/local/include/opencv2/core/mat.hpp \
		/usr/local/include/opencv2/flann/miniflann.hpp \
		/usr/local/include/opencv2/flann/defines.h \
		/usr/local/include/opencv2/flann/config.h \
		/usr/local/include/opencv2/imgproc/imgproc_c.h \
		/usr/local/include/opencv2/imgproc/types_c.h \
		/usr/local/include/opencv2/imgproc/imgproc.hpp \
		/usr/local/include/opencv2/photo/photo.hpp \
		/usr/local/include/opencv2/photo/photo_c.h \
		/usr/local/include/opencv2/video/video.hpp \
		/usr/local/include/opencv2/video/tracking.hpp \
		/usr/local/include/opencv2/video/background_segm.hpp \
		/usr/local/include/opencv2/features2d/features2d.hpp \
		/usr/local/include/opencv2/objdetect/objdetect.hpp \
		/usr/local/include/opencv2/calib3d/calib3d.hpp \
		/usr/local/include/opencv2/ml/ml.hpp \
		/usr/local/include/opencv2/highgui/highgui_c.h \
		/usr/local/include/opencv2/highgui/highgui.hpp \
		/usr/local/include/opencv2/contrib/contrib.hpp \
		/usr/local/include/opencv2/contrib/retina.hpp \
		/usr/local/include/opencv2/contrib/openfabmap.hpp
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o detectObject.o detectObject.cpp

OGBP.o: OGBP.cpp OGBP.h \
		/usr/local/include/opencv2/objdetect/objdetect.hpp \
		/usr/local/include/opencv2/core/core.hpp \
		/usr/local/include/opencv2/core/types_c.h \
		/usr/local/include/opencv2/core/version.hpp \
		/usr/local/include/opencv2/core/operations.hpp \
		/usr/local/include/opencv2/core/mat.hpp \
		/usr/local/include/opencv2/highgui/highgui.hpp \
		/usr/local/include/opencv2/highgui/highgui_c.h \
		/usr/local/include/opencv2/core/core_c.h \
		/usr/local/include/opencv2/imgproc/imgproc.hpp \
		/usr/local/include/opencv2/imgproc/types_c.h \
		/usr/local/include/opencv2/contrib/contrib.hpp \
		/usr/local/include/opencv2/features2d/features2d.hpp \
		/usr/local/include/opencv2/flann/miniflann.hpp \
		/usr/local/include/opencv2/flann/defines.h \
		/usr/local/include/opencv2/flann/config.h \
		/usr/local/include/opencv2/contrib/retina.hpp \
		/usr/local/include/opencv2/contrib/openfabmap.hpp
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o OGBP.o OGBP.cpp

preprocessFace.o: preprocessFace.cpp detectObject.h \
		/usr/local/include/opencv2/opencv.hpp \
		/usr/local/include/opencv2/core/core_c.h \
		/usr/local/include/opencv2/core/types_c.h \
		/usr/local/include/opencv2/core/core.hpp \
		/usr/local/include/opencv2/core/version.hpp \
		/usr/local/include/opencv2/core/operations.hpp \
		/usr/local/include/opencv2/core/mat.hpp \
		/usr/local/include/opencv2/flann/miniflann.hpp \
		/usr/local/include/opencv2/flann/defines.h \
		/usr/local/include/opencv2/flann/config.h \
		/usr/local/include/opencv2/imgproc/imgproc_c.h \
		/usr/local/include/opencv2/imgproc/types_c.h \
		/usr/local/include/opencv2/imgproc/imgproc.hpp \
		/usr/local/include/opencv2/photo/photo.hpp \
		/usr/local/include/opencv2/photo/photo_c.h \
		/usr/local/include/opencv2/video/video.hpp \
		/usr/local/include/opencv2/video/tracking.hpp \
		/usr/local/include/opencv2/video/background_segm.hpp \
		/usr/local/include/opencv2/features2d/features2d.hpp \
		/usr/local/include/opencv2/objdetect/objdetect.hpp \
		/usr/local/include/opencv2/calib3d/calib3d.hpp \
		/usr/local/include/opencv2/ml/ml.hpp \
		/usr/local/include/opencv2/highgui/highgui_c.h \
		/usr/local/include/opencv2/highgui/highgui.hpp \
		/usr/local/include/opencv2/contrib/contrib.hpp \
		/usr/local/include/opencv2/contrib/retina.hpp \
		/usr/local/include/opencv2/contrib/openfabmap.hpp \
		preprocessFace.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o preprocessFace.o preprocessFace.cpp

####### Install

install:   FORCE

uninstall:   FORCE

FORCE:


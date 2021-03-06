# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3

# Include any dependencies generated for this target.
include traffic_detection/CMakeFiles/test-traffic-detection.dir/depend.make

# Include the progress variables for this target.
include traffic_detection/CMakeFiles/test-traffic-detection.dir/progress.make

# Include the compile flags for this target's objects.
include traffic_detection/CMakeFiles/test-traffic-detection.dir/flags.make

traffic_detection/CMakeFiles/test-traffic-detection.dir/test-traffic-detection.cpp.o: traffic_detection/CMakeFiles/test-traffic-detection.dir/flags.make
traffic_detection/CMakeFiles/test-traffic-detection.dir/test-traffic-detection.cpp.o: traffic_detection/test-traffic-detection.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object traffic_detection/CMakeFiles/test-traffic-detection.dir/test-traffic-detection.cpp.o"
	cd /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/test-traffic-detection.dir/test-traffic-detection.cpp.o -c /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection/test-traffic-detection.cpp

traffic_detection/CMakeFiles/test-traffic-detection.dir/test-traffic-detection.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test-traffic-detection.dir/test-traffic-detection.cpp.i"
	cd /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection/test-traffic-detection.cpp > CMakeFiles/test-traffic-detection.dir/test-traffic-detection.cpp.i

traffic_detection/CMakeFiles/test-traffic-detection.dir/test-traffic-detection.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test-traffic-detection.dir/test-traffic-detection.cpp.s"
	cd /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection/test-traffic-detection.cpp -o CMakeFiles/test-traffic-detection.dir/test-traffic-detection.cpp.s

traffic_detection/CMakeFiles/test-traffic-detection.dir/test-traffic-detection.cpp.o.requires:
.PHONY : traffic_detection/CMakeFiles/test-traffic-detection.dir/test-traffic-detection.cpp.o.requires

traffic_detection/CMakeFiles/test-traffic-detection.dir/test-traffic-detection.cpp.o.provides: traffic_detection/CMakeFiles/test-traffic-detection.dir/test-traffic-detection.cpp.o.requires
	$(MAKE) -f traffic_detection/CMakeFiles/test-traffic-detection.dir/build.make traffic_detection/CMakeFiles/test-traffic-detection.dir/test-traffic-detection.cpp.o.provides.build
.PHONY : traffic_detection/CMakeFiles/test-traffic-detection.dir/test-traffic-detection.cpp.o.provides

traffic_detection/CMakeFiles/test-traffic-detection.dir/test-traffic-detection.cpp.o.provides.build: traffic_detection/CMakeFiles/test-traffic-detection.dir/test-traffic-detection.cpp.o

# Object files for target test-traffic-detection
test__traffic__detection_OBJECTS = \
"CMakeFiles/test-traffic-detection.dir/test-traffic-detection.cpp.o"

# External object files for target test-traffic-detection
test__traffic__detection_EXTERNAL_OBJECTS =

bin/Release/test-traffic-detection: traffic_detection/CMakeFiles/test-traffic-detection.dir/test-traffic-detection.cpp.o
bin/Release/test-traffic-detection: traffic_detection/CMakeFiles/test-traffic-detection.dir/build.make
bin/Release/test-traffic-detection: bin/Release/libtraffic-detection.a
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_cudastereo.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_cudabgsegm.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_videostab.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_ml.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_dnn.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_shape.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_superres.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_cudaobjdetect.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_photo.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_stitching.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_cudaoptflow.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_cudacodec.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_cudafeatures2d.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_cudawarping.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_cudalegacy.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_cudaimgproc.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_video.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_objdetect.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_cudafilters.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_calib3d.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_features2d.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_highgui.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_videoio.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_flann.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_imgcodecs.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_imgproc.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_cudaarithm.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_core.so.3.4.1
bin/Release/test-traffic-detection: /usr/local/lib/libopencv_cudev.so.3.4.1
bin/Release/test-traffic-detection: /usr/lib/arm-linux-gnueabihf/libpython3.4m.so
bin/Release/test-traffic-detection: traffic_detection/CMakeFiles/test-traffic-detection.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../bin/Release/test-traffic-detection"
	cd /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-traffic-detection.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
traffic_detection/CMakeFiles/test-traffic-detection.dir/build: bin/Release/test-traffic-detection
.PHONY : traffic_detection/CMakeFiles/test-traffic-detection.dir/build

traffic_detection/CMakeFiles/test-traffic-detection.dir/requires: traffic_detection/CMakeFiles/test-traffic-detection.dir/test-traffic-detection.cpp.o.requires
.PHONY : traffic_detection/CMakeFiles/test-traffic-detection.dir/requires

traffic_detection/CMakeFiles/test-traffic-detection.dir/clean:
	cd /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection && $(CMAKE_COMMAND) -P CMakeFiles/test-traffic-detection.dir/cmake_clean.cmake
.PHONY : traffic_detection/CMakeFiles/test-traffic-detection.dir/clean

traffic_detection/CMakeFiles/test-traffic-detection.dir/depend:
	cd /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3 /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3 /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection/CMakeFiles/test-traffic-detection.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : traffic_detection/CMakeFiles/test-traffic-detection.dir/depend


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
include traffic_detection/CMakeFiles/traffic_detection.dir/depend.make

# Include the progress variables for this target.
include traffic_detection/CMakeFiles/traffic_detection.dir/progress.make

# Include the compile flags for this target's objects.
include traffic_detection/CMakeFiles/traffic_detection.dir/flags.make

traffic_detection/CMakeFiles/traffic_detection.dir/DetecterTrafficSign_NII.cpp.o: traffic_detection/CMakeFiles/traffic_detection.dir/flags.make
traffic_detection/CMakeFiles/traffic_detection.dir/DetecterTrafficSign_NII.cpp.o: traffic_detection/DetecterTrafficSign_NII.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object traffic_detection/CMakeFiles/traffic_detection.dir/DetecterTrafficSign_NII.cpp.o"
	cd /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/traffic_detection.dir/DetecterTrafficSign_NII.cpp.o -c /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection/DetecterTrafficSign_NII.cpp

traffic_detection/CMakeFiles/traffic_detection.dir/DetecterTrafficSign_NII.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/traffic_detection.dir/DetecterTrafficSign_NII.cpp.i"
	cd /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection/DetecterTrafficSign_NII.cpp > CMakeFiles/traffic_detection.dir/DetecterTrafficSign_NII.cpp.i

traffic_detection/CMakeFiles/traffic_detection.dir/DetecterTrafficSign_NII.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/traffic_detection.dir/DetecterTrafficSign_NII.cpp.s"
	cd /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection/DetecterTrafficSign_NII.cpp -o CMakeFiles/traffic_detection.dir/DetecterTrafficSign_NII.cpp.s

traffic_detection/CMakeFiles/traffic_detection.dir/DetecterTrafficSign_NII.cpp.o.requires:
.PHONY : traffic_detection/CMakeFiles/traffic_detection.dir/DetecterTrafficSign_NII.cpp.o.requires

traffic_detection/CMakeFiles/traffic_detection.dir/DetecterTrafficSign_NII.cpp.o.provides: traffic_detection/CMakeFiles/traffic_detection.dir/DetecterTrafficSign_NII.cpp.o.requires
	$(MAKE) -f traffic_detection/CMakeFiles/traffic_detection.dir/build.make traffic_detection/CMakeFiles/traffic_detection.dir/DetecterTrafficSign_NII.cpp.o.provides.build
.PHONY : traffic_detection/CMakeFiles/traffic_detection.dir/DetecterTrafficSign_NII.cpp.o.provides

traffic_detection/CMakeFiles/traffic_detection.dir/DetecterTrafficSign_NII.cpp.o.provides.build: traffic_detection/CMakeFiles/traffic_detection.dir/DetecterTrafficSign_NII.cpp.o

# Object files for target traffic_detection
traffic_detection_OBJECTS = \
"CMakeFiles/traffic_detection.dir/DetecterTrafficSign_NII.cpp.o"

# External object files for target traffic_detection
traffic_detection_EXTERNAL_OBJECTS =

bin/Release/libtraffic_detection.a: traffic_detection/CMakeFiles/traffic_detection.dir/DetecterTrafficSign_NII.cpp.o
bin/Release/libtraffic_detection.a: traffic_detection/CMakeFiles/traffic_detection.dir/build.make
bin/Release/libtraffic_detection.a: traffic_detection/CMakeFiles/traffic_detection.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library ../bin/Release/libtraffic_detection.a"
	cd /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection && $(CMAKE_COMMAND) -P CMakeFiles/traffic_detection.dir/cmake_clean_target.cmake
	cd /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/traffic_detection.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
traffic_detection/CMakeFiles/traffic_detection.dir/build: bin/Release/libtraffic_detection.a
.PHONY : traffic_detection/CMakeFiles/traffic_detection.dir/build

traffic_detection/CMakeFiles/traffic_detection.dir/requires: traffic_detection/CMakeFiles/traffic_detection.dir/DetecterTrafficSign_NII.cpp.o.requires
.PHONY : traffic_detection/CMakeFiles/traffic_detection.dir/requires

traffic_detection/CMakeFiles/traffic_detection.dir/clean:
	cd /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection && $(CMAKE_COMMAND) -P CMakeFiles/traffic_detection.dir/cmake_clean.cmake
.PHONY : traffic_detection/CMakeFiles/traffic_detection.dir/clean

traffic_detection/CMakeFiles/traffic_detection.dir/depend:
	cd /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3 /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3 /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection /home/ubuntu/DriverlessCarChallenge_MTARacer-master/round_1/DriverlessCar/carControl/src/0.3/traffic_detection/CMakeFiles/traffic_detection.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : traffic_detection/CMakeFiles/traffic_detection.dir/depend


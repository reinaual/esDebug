# Copyright (C) 2009-2018 The ESPResSo project
# Copyright (C) 2009,2010 
#   Max-Planck-Institute for Polymer Research, Theory Group
#
# This file is part of ESPResSo.
#
# ESPResSo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ESPResSo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

cmake_minimum_required(VERSION 3.0)
include(FeatureSummary)
project(ESPResSo)
include(cmake/FindPythonModule.cmake)

enable_language(CXX)

set(PROJECT_VERSION "4.0-dev")
string(REGEX REPLACE "^([1-9]+)\\.[0-9]+.*$" "\\1" SOVERSION "${PROJECT_VERSION}")
if (NOT ${SOVERSION} MATCHES "^[1-9]+$")
  message(FATAL_ERROR "Could not determind SOVERSION from ${PROJECT_VERSION}")
endif (NOT ${SOVERSION} MATCHES "^[1-9]+$")

######################################################################
# CMake internal vars
######################################################################

# Cmake modules/macros are in a subdirectory to keep this file cleaner
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# If the version is smaller than 3.4, use custom find HDF5 module
# else, use prefer parallel feature of default module.
if(CMAKE_VERSION VERSION_LESS 3.4)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/hdf5)
else()
  set(HDF5_PREFER_PARALLEL 1)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# Parse repository info from git if available
include(version)

configure_file(
  ${CMAKE_SOURCE_DIR}/src/version.hpp.in
  ${CMAKE_BINARY_DIR}/src/version.hpp
  )

# On Mac OS X, first look for other packages, then frameworks
set(CMAKE_FIND_FRAMEWORK LAST)

######################################################################
# User input options
######################################################################

set(WITH_PYTHON yes)

option(WITH_GSL "Build with GSL support" ON)
option(WITH_CUDA   "Build with GPU support"  ON)
option(WITH_HDF5   "Build with HDF5 support" ON)
option(WITH_TESTS  "Enable tests"            ON)
option(WITH_SCAFACOS "Build with Scafacos support" ON)
option(WITH_VALGRIND_INSTRUMENTATION "Build with valgrind instrumentation markers" OFF)
if( CMAKE_VERSION VERSION_GREATER 3.5.2 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
  option(WITH_CLANG_TIDY "Run Clang-Tidy during compilation" OFF)
endif()
if( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
  option(WITH_COVERAGE "Generate code coverage report" OFF)
  option(WITH_ASAN "Build with address sanitizer" OFF)
  option(WITH_UBSAN "Build with undefined behavior sanitizer" OFF)
endif()
if( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT APPLE )
  option(WITH_MSAN "Build with memory sanitizer (experimental; requires a memory-sanitized Python interpreter)" OFF)
endif()
option(WARNINGS_ARE_ERRORS "Treat warnings as errors during compilation" OFF)
option(WITH_CCACHE "Use ccache compiler invocation." OFF)

if(WITH_CCACHE)
  find_program(CCACHE ccache)
  if(CCACHE)
    message(STATUS "Found ccache ${CCACHE}")
    if(CMAKE_VERSION VERSION_LESS 3.4)
      set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE})
      set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE})
    else()
      set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE} CACHE STRING "ccache executable")
    endif()
  else()
      message(FATAL_ERROR "ccache not found.")
  endif(CCACHE)
endif(WITH_CCACHE)


# Write compile commands to file, for various tools...
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# choose the name of the config file
set(MYCONFIG_NAME "myconfig.hpp"
  CACHE STRING "Default name of the local config file")

# Check which config file to use
include(MyConfig)

######################################################################
# Pretty function
######################################################################

include(CheckCXXSourceCompiles)

set(__PRETTYFUNC__ __func__)
foreach(func_name __PRETTY_FUNCTION__ __FUNCTION__)
   check_cxx_source_compiles("
     #include <string>
     int main() { std::string(${func_name}); }
     " result${func_name})
   if(result${func_name})
     set(__PRETTYFUNC__ ${func_name})
     break()
   endif(result${func_name})
endforeach(func_name __PRETTY_FUNCTION__ __FUNCTION__)

######################################################################
# Libraries
######################################################################

if (WITH_CUDA)
  if (EXISTS "$ENV{NVCC}" AND NOT CUDA_NVCC_EXECUTABLE)
    set(CUDA_NVCC_EXECUTABLE $ENV{NVCC} CACHE FILEPATH "Path to CUDA compiler.")
  endif()
  if (CUDA_NVCC_EXECUTABLE STREQUAL CMAKE_CXX_COMPILER)
    message(STATUS "Found CUDA-capable host compiler: ${CUDA_NVCC_EXECUTABLE}")
    if( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" )
        execute_process(COMMAND ${CUDA_NVCC_EXECUTABLE} ${CMAKE_CXX_FLAGS} --verbose
                ERROR_VARIABLE CUDA_DIR)
        string(REGEX REPLACE "^.*Found CUDA installation: ([^,]+).*\$" "\\1" CUDA_DIR "${CUDA_DIR}")
        message(STATUS "Found CUDA installation: ${CUDA_DIR}")
    else()
        set(CUDA_DIR "/usr/local/cuda")
    endif()
    find_library(CUDART_LIBRARY NAMES cudart PATHS ${CUDA_DIR}/lib64 ${CUDA_DIR}/lib /usr/local/nvidia/lib)
    find_library(CUFFT_LIBRARY NAMES cufft PATHS ${CUDA_DIR}/lib64 ${CUDA_DIR}/lib /usr/local/nvidia/lib)
    list(APPEND LIBRARIES ${CUDART_LIBRARY})
    list(APPEND LIBRARIES ${CUFFT_LIBRARY})
    set(CUDA 1)

    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --cuda-gpu-arch=sm_30")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 3.8.9)
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --cuda-gpu-arch=sm_52")
    endif()

    function(cuda_include_directories)
      include_directories(${ARGV})
    endfunction()
    function(cuda_add_library)
      set(options STATIC SHARED MODULE EXCLUDE_FROM_ALL)
      set(oneValueArgs)
      set(multiValueArgs)
      cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
      list(REMOVE_AT ARG_UNPARSED_ARGUMENTS 0)
      set_source_files_properties(${ARG_UNPARSED_ARGUMENTS} PROPERTIES LANGUAGE "CXX" COMPILE_FLAGS "${CUDA_NVCC_FLAGS}")
      add_library(${ARGV})
      set_target_properties(${ARGV0} PROPERTIES LINKER_LANGUAGE "CXX")
      target_link_libraries(${ARGV0} ${CUDA_LIBRARY} ${CUDART_LIBRARY})
    endfunction()
    function(CUDA_ADD_CUFFT_TO_TARGET target)
      target_link_libraries(${target} "cufft")
    endfunction()
  else()
    find_package(CUDA 7.0)
    if(CUDA_FOUND)
      list(APPEND LIBRARIES ${CUDA_CUFFT_LIBRARIES})
      list(APPEND LIBRARIES ${CUDA_LIBRARIES})
      set(CUDA 1)

      set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} -g -G")
      set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} -O3 -DNDEBUG")
      set(CUDA_NVCC_FLAGS_MINSIZEREL "${CUDA_NVCC_FLAGS_MINSIZEREL} -Os -DNDEBUG")
      set(CUDA_NVCC_FLAGS_RELWITHDEBINFO "${CUDA_NVCC_FLAGS_RELWITHDEBINFO} -g -G -O2")
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52")
      list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
      if (CUDA_VERSION VERSION_LESS "8.0" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "5.0")
        list(APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
      endif()
      SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

      if (CMAKE_OSX_SYSROOT)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -isysroot -Xcompiler ${CMAKE_OSX_SYSROOT}")
      endif()
    endif(CUDA_FOUND)
  endif()
endif(WITH_CUDA)

find_package(PythonInterp REQUIRED)

if (WITH_PYTHON)
  find_package(Cython 0.23 REQUIRED)
  execute_process(COMMAND ${PYTHON_EXECUTABLE}
                -c "import distutils.sysconfig as cg; print(cg.get_python_inc())"
                OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_package(NumPy REQUIRED)
  if(NOT PYTHON_INSTDIR)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} 
                -c "import distutils.sysconfig as cg; print(cg.get_python_lib(1,0,prefix='${CMAKE_INSTALL_EXEC_PREFIX}'))"
                OUTPUT_VARIABLE PYTHON_INSTDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif(NOT PYTHON_INSTDIR)
  if(PYTHON_VERSION_MAJOR EQUAL 2)
    set(CYTHON_FLAGS "-2" CACHE STRING "Flags used by the Cython compiler during all build types.")
  else()
    set(CYTHON_FLAGS "-3" CACHE STRING "Flags used by the Cython compiler during all build types.")
  endif()

  if(WARNINGS_ARE_ERRORS)
    set(CYTHON_FLAGS "--warning-errors;${CYTHON_FLAGS}")
  endif()
endif(WITH_PYTHON)


find_package(FFTW3)
if (FFTW3_FOUND)
  include_directories(SYSTEM ${FFTW3_INCLUDE_DIR})
  list(APPEND LIBRARIES ${FFTW3_LIBRARIES})
  set(FFTW 3)
endif(FFTW3_FOUND)


# If we build either interface turn on script interface
if (WITH_PYTHON)
  set(WITH_SCRIPT_INTERFACE ON)
endif()

# We need the parallel hdf5 version!
if (WITH_HDF5)
  find_package(HDF5 "1.8" COMPONENTS C)
  if(HDF5_IS_PARALLEL)
    set(H5MD 1)
    include(FindPythonModule)
    find_python_module(h5py)
    list(APPEND LIBRARIES ${HDF5_LIBRARIES})
    include_directories(SYSTEM ${HDF5_INCLUDE_DIRS})
    add_feature_info(HDF5 ON "parallel")
  else()
    unset(H5MD)
    unset(HDF5_FOUND)
  endif(HDF5_IS_PARALLEL)
endif(WITH_HDF5)

# Check for the h5xx submodule
# and try to check it out if not found or update it if found.
if(WITH_HDF5 AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
  # Try to find git
  find_package(Git)
  if(GIT_FOUND)
    if (NOT EXISTS "${CMAKE_SOURCE_DIR}/libs/h5xx/.git")
      execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init -- libs/h5xx
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    else()
      execute_process(COMMAND ${GIT_EXECUTABLE} submodule update -- libs/h5xx
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endif()
  endif()
endif()

if(WITH_SCAFACOS)
  find_package(PkgConfig)
  pkg_check_modules(SCAFACOS scafacos)
  if(SCAFACOS_FOUND)
    set(SCAFACOS 1)
  endif(SCAFACOS_FOUND)
endif(WITH_SCAFACOS)

if(WITH_GSL)
  find_package(GSL)
  if (GSL_FOUND)
    set(GSL 1)
  endif(GSL_FOUND)
endif(WITH_GSL)

if(WITH_VALGRIND_INSTRUMENTATION)
  find_package(PkgConfig)
  pkg_check_modules(VALGRIND valgrind)
  if(VALGRIND_FOUND)
    set(VALGRIND_INSTRUMENTATION 1)
    message(STATUS ${VALGRIND_INCLUDE_DIRS})
    include_directories(SYSTEM ${VALGRIND_INCLUDE_DIRS})
  endif(VALGRIND_FOUND)
endif(WITH_VALGRIND_INSTRUMENTATION)

#######################################################################
# Process MPI settings
#######################################################################

find_package(MPI REQUIRED)
# Workaround for https://gitlab.kitware.com/cmake/cmake/issues/18349
foreach(_MPI_FLAG ${MPI_CXX_COMPILE_FLAGS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_MPI_FLAG}")
endforeach()
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_LINK_FLAGS}")
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
list(APPEND LIBRARIES ${MPI_LIBRARIES})
add_definitions(-DH5XX_USE_MPI)

#######################################################################
# Boost
#######################################################################


list(APPEND BOOST_COMPONENTS mpi serialization filesystem system)

if(WITH_TESTS)
  list(APPEND BOOST_COMPONENTS unit_test_framework)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set(BOOST_MINIMUM_VERSION "1.65.0")
  if (CUDA_FOUND AND NOT CUDA_VERSION VERSION_LESS "9.0")
    set(BOOST_MINIMUM_VERSION "1.66.0")
  endif()
else()
  set(BOOST_MINIMUM_VERSION "1.53.0")
endif()

find_package(Boost ${BOOST_MINIMUM_VERSION} REQUIRED ${BOOST_COMPONENTS})
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
list(APPEND LIBRARIES ${Boost_LIBRARIES})

if(Boost_VERSION VERSION_GREATER 106399 AND Boost_VERSION VERSION_LESS 106500)
  # Boost 1.64 has incompatible Serialization and MPI modules, see https://svn.boost.org/trac10/ticket/12723 .
  # Some distributions, like Fedora, have backported the patch.
  FILE(READ "${Boost_INCLUDE_DIR}/boost/mpi/detail/mpi_datatype_primitive.hpp" boost_mpi_datatype_file)
  if(boost_mpi_datatype_file MATCHES "boost::serialization::detail::get_data")
    message(FATAL_ERROR "Boost 1.64 is unsupported")
  endif()
endif()

#######################################################################
# Paths
#######################################################################

if (NOT DEFINED DATA)
  set(DATA "share/espresso")
endif(NOT DEFINED DATA)

if (NOT DEFINED LIBDIR)
  set(LIBDIR "lib")
endif(NOT DEFINED LIBDIR)

if (NOT DEFINED BINDIR)
  set(BINDIR "bin")
endif(NOT DEFINED BINDIR)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${LIBDIR}")


#######################################################################
# Flags
#######################################################################

# C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(CMAKE_VERSION VERSION_LESS 3.1)
  include(RequireCXX11)
endif()


if(APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -undefined dynamic_lookup")
endif()
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")
endif()

if(WARNINGS_ARE_ERRORS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  if (NOT CUDA_NVCC_EXECUTABLE STREQUAL CMAKE_CXX_COMPILER)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -Werror")
  endif()
endif(WARNINGS_ARE_ERRORS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-sign-compare -Wno-unused-function -Wno-unused-variable")
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-psabi")
endif()
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # G++ and Intel don't know this flag
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field")
endif()
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.8.5")
  # older versions don't support -Wno-pedantic which we need in src/python
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
endif()
if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "16.0")
  # workaround for compiler crash related to decltype() and variadic template usage inside Boost
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX11_VARIADIC_TEMPLATES")
endif()

set(CMAKE_MACOSX_RPATH TRUE)

if( WITH_ASAN AND WITH_MSAN )
  message(FATAL_ERROR "Address sanitizer and memory sanitizer cannot be enabled simultaneously")
endif()
if( WITH_ASAN )
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -O1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
endif()
if( WITH_MSAN )
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -O1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory -fno-omit-frame-pointer")
endif()
if( WITH_UBSAN )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
endif()

#######################################################################
# Testing
#######################################################################

if( WITH_CLANG_TIDY )
  string(REGEX REPLACE "^([1-9]+\\.[0-9]+).*$" "\\1" CLANG_MINOR_VERSION "${CMAKE_CXX_COMPILER_VERSION}")
  find_program(
    CLANG_TIDY_EXE
    NAMES "clang-tidy-${CLANG_MINOR_VERSION}" "clang-tidy"
    DOC "Path to clang-tidy executable"
    )
  if(NOT CLANG_TIDY_EXE)
    message(STATUS "clang-tidy not found.")
  else()
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}" "-checks=*,-clang-analyzer-alpha.*,-misc-unused-*,-android-*,-fuchsia-*,-google-*,-llvm-*,-modernize-*,-readability-*,-cppcoreguidelines-*,-clang-analyzer-deadcode*,-cert-*,-clang-analyzer-core.NullDereference,-clang-analyzer-core.DivideZero,-misc-misplaced-widening-cast,-misc-macro-parentheses,-clang-analyzer-security.FloatLoopCounter,-clang-analyzer-optin.mpi.MPI-Checker,-boost-use-to-string,-hicpp-*" "-header-filter=.*" "-warnings-as-errors=*")
  endif()
endif()

if(WITH_TESTS)
  enable_testing()
  if(Boost_UNIT_TEST_FRAMEWORK_FOUND)
    set(WITH_UNIT_TESTS ON)
    list(APPEND LIBRARIES ${Boost_LIBRARIES})
  endif(Boost_UNIT_TEST_FRAMEWORK_FOUND)
  add_custom_target(check)
  add_subdirectory(testsuite)
endif(WITH_TESTS)

#######################################################################
# Subdirectories
#######################################################################

add_subdirectory(doc)
add_subdirectory(src)
add_subdirectory(config)

#######################################################################
# Feature summary
#######################################################################

include(FeatureSummary)
feature_summary(WHAT ALL)

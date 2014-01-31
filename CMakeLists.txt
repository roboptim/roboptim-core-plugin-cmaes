# Copyright 2014, Benjamin Chretien, CNRS-LIRMM.
#
# This file is part of roboptim-core.
# roboptim-core is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# roboptim-core is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Lesser Public License for more details.
# You should have received a copy of the GNU Lesser General Public License
# along with roboptim-core.  If not, see <http://www.gnu.org/licenses/>.

# Requires at least CMake 2.8 to configure the package.
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

INCLUDE(cmake/base.cmake)
INCLUDE(cmake/boost.cmake)
INCLUDE(cmake/cpack.cmake)

SET(PROJECT_NAME roboptim-core-plugin-cmaes)
SET(PROJECT_DESCRIPTION "RobOptim core CMA-ES plug-in")
SET(PROJECT_URL "http://github.com/roboptim/roboptim-core-plugin-cmaes")

SET(HEADERS
  ${CMAKE_SOURCE_DIR}/include/roboptim/core/plugin/cmaes/cmaes.hh
  ${CMAKE_SOURCE_DIR}/include/roboptim/core/plugin/cmaes/debug.hh
  ${CMAKE_SOURCE_DIR}/include/roboptim/core/plugin/cmaes/doc.hh
  )

SET(PKG_CONFIG_ADDITIONAL_VARIABLES
  plugindir ${PKG_CONFIG_ADDITIONAL_VARIABLES})


SETUP_PROJECT()

# If compiler support symbol visibility, enable it.
INCLUDE(CheckCCompilerFlag)
CHECK_C_COMPILER_FLAG(-fvisibility=hidden HAS_VISIBILITY)
IF(HAS_VISIBILITY)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
ENDIF()

# Search for dependencies.
SET(BOOST_COMPONENTS
  date_time filesystem system thread program_options unit_test_framework)
SEARCH_FOR_BOOST()

# Libtool dynamic loading
# This project does not use Libtool directly but still uses ltdl for
# plug-in loading.
# Libtool dynamic loading
# This project does not use Libtool directly but still uses ltdl for
# plug-in loading.
INCLUDE(CheckIncludeFileCXX)
CHECK_INCLUDE_FILE_CXX(ltdl.h LTDL_H_FOUND)
IF (NOT LTDL_H_FOUND)
  MESSAGE(FATAL_ERROR
    "Failed to find ltdl.h, check that Libtool ltdl is installed.")
ENDIF()
#FIXME: check that libltdl.so is available instead of adding it blindly.
PKG_CONFIG_APPEND_LIBS(ltdl)

ADD_REQUIRED_DEPENDENCY("roboptim-core >= 0.5")

HEADER_INSTALL("${HEADERS}")

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(tests)

SETUP_PROJECT_FINALIZE()
SETUP_PROJECT_CPACK()

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

# Define the directory where plug-ins will be installed.
SET(PLUGINDIR lib/roboptim-core)

# CMA-ES library.
ADD_LIBRARY(cmaes SHARED
  ${HEADERS}
  cma/cmaes.c
  cma/boundary_transformation.c
  )

SET_TARGET_PROPERTIES(cmaes PROPERTIES
  INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/include/roboptim/core/plugin/cmaes/cma)
INSTALL(TARGETS cmaes DESTINATION ${ROBOPTIM_CORE_PLUGINDIR})

# CMA-ES plug-in.
ADD_LIBRARY(roboptim-core-plugin-cmaes MODULE
  ${HEADERS}
  roboptim-core-plugin-cmaes.cc
  )

SET_TARGET_PROPERTIES(roboptim-core-plugin-cmaes PROPERTIES
  PREFIX ""
  SOVERSION 1.1.0)
INSTALL(TARGETS roboptim-core-plugin-cmaes DESTINATION ${ROBOPTIM_CORE_PLUGINDIR})
PKG_CONFIG_USE_DEPENDENCY(roboptim-core-plugin-cmaes roboptim-core)
TARGET_LINK_LIBRARIES(roboptim-core-plugin-cmaes cmaes)

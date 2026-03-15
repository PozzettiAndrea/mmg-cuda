## =============================================================================
##  This file is part of the mmg software package for the tetrahedral
##  mesh modification.
##**  Copyright (c) Bx INP/Inria/UBordeaux/UPMC, 2004- .
##
##  mmg is free software: you can redistribute it and/or modify it
##  under the terms of the GNU Lesser General Public License as published
##  by the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  mmg is distributed in the hope that it will be useful, but WITHOUT
##  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
##  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
##  License for more details.
##
##  You should have received a copy of the GNU Lesser General Public
##  License and of the GNU General Public License along with mmg (in
##  files COPYING.LESSER and COPYING). If not, see
##  <http://www.gnu.org/licenses/>. Please read their terms carefully and
##  use this copy of the mmg distribution only if you accept them.
## =============================================================================

## =============================================================================
##
## Compilation of mmgs executable, libraries and tests
##
## =============================================================================

SET(MMGS_SOURCE_DIR      ${PROJECT_SOURCE_DIR}/src/mmgs)
SET(MMGS_BINARY_DIR      ${PROJECT_BINARY_DIR}/src/mmgs)
SET(MMGS_SHRT_INCLUDE    mmg/mmgs )
SET(MMGS_INCLUDE         ${PROJECT_BINARY_DIR}/include/${MMGS_SHRT_INCLUDE} )

FILE(MAKE_DIRECTORY ${MMGS_BINARY_DIR})

############################################################################
#####
#####         Fortran header: libmmgsf.h
#####
############################################################################

if (PERL_FOUND)
  GENERATE_FORTRAN_HEADER ( mmgs
    ${MMGS_SOURCE_DIR} libmmgs.h
    mmg/common
    ${MMGS_BINARY_DIR} libmmgsf.h
    )
endif (PERL_FOUND)

###############################################################################
#####
#####         Sources and libraries
#####
###############################################################################

# Source files
FILE(
  GLOB
  mmgs_library_files
  ${MMGS_SOURCE_DIR}/*.c
  ${MMGCOMMON_SOURCE_DIR}/*.c
  ${MMGS_SOURCE_DIR}/*.h
  ${MMGCOMMON_SOURCE_DIR}/*.h
  ${MMGS_SOURCE_DIR}/inoutcpp_s.cpp
  )
LIST(REMOVE_ITEM mmgs_library_files
  ${MMGS_SOURCE_DIR}/mmgs.c
  ${MMGCOMMON_SOURCE_DIR}/apptools.c
  ${REMOVE_FILE} )

IF ( VTK_FOUND )
  LIST(APPEND  mmgs_library_files
   ${MMGCOMMON_SOURCE_DIR}/vtkparser.cpp )
ENDIF ( )

# CUDA acceleration sources
IF ( BUILD_CUDA )
  FILE(GLOB mmgs_cuda_c_files
    ${MMGS_SOURCE_DIR}/cuda/*.c
    ${MMGS_SOURCE_DIR}/cuda/*.h)
  FILE(GLOB mmgs_cuda_cu_files
    ${MMGS_SOURCE_DIR}/cuda/*.cu)
  LIST(APPEND mmgs_library_files ${mmgs_cuda_c_files})
  IF ( mmgs_cuda_cu_files )
    LIST(APPEND mmgs_library_files ${mmgs_cuda_cu_files})
    SET_SOURCE_FILES_PROPERTIES(${mmgs_cuda_cu_files} PROPERTIES LANGUAGE CUDA)
  ENDIF()
ENDIF()

FILE(
  GLOB
  mmgs_main_file
  ${MMGS_SOURCE_DIR}/mmgs.c
  ${MMGCOMMON_SOURCE_DIR}/apptools.c
  )

############################################################################
#####
#####         Compile mmgs libraries
#####
############################################################################

# RXMesh remeshing bridge (separate static lib, compiled with nvcc + RXMesh headers)
IF ( BUILD_RXMESH )
  ADD_LIBRARY(mmgs_rxmesh_bridge STATIC
    ${MMGS_SOURCE_DIR}/cuda/rxmesh_remesh/mmgs_rxmesh_bridge.cu
  )
  TARGET_INCLUDE_DIRECTORIES(mmgs_rxmesh_bridge PRIVATE
    ${RXMESH_SOURCE_DIR}/include
    ${RXMESH_SOURCE_DIR}/external
    ${RXMESH_SOURCE_DIR}/external/glm
    ${MMGCOMMON_SOURCE_DIR}
    ${MMGS_SOURCE_DIR}
    ${MMGS_SOURCE_DIR}/cuda/rxmesh_remesh
    ${PROJECT_BINARY_DIR}/src/common
    ${PROJECT_BINARY_DIR}/include/mmg
    ${PROJECT_BINARY_DIR}/include/mmg/mmgs
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/mmgs
  )
  TARGET_LINK_LIBRARIES(mmgs_rxmesh_bridge PRIVATE RXMesh CUDA::cudart)
  SET_TARGET_PROPERTIES(mmgs_rxmesh_bridge PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  # Standalone test executable for RXMesh aniso remeshing
  # Usage: ./rxmesh_aniso_remesh input.obj [target_len] [num_iter]
  ADD_EXECUTABLE(rxmesh_aniso_remesh
    ${MMGS_SOURCE_DIR}/cuda/rxmesh_remesh/standalone_test.cu
  )
  TARGET_INCLUDE_DIRECTORIES(rxmesh_aniso_remesh PRIVATE
    ${RXMESH_SOURCE_DIR}/include
    ${RXMESH_SOURCE_DIR}/external
    ${RXMESH_SOURCE_DIR}/external/glm
    ${MMGS_SOURCE_DIR}/cuda/rxmesh_remesh
  )
  TARGET_LINK_LIBRARIES(rxmesh_aniso_remesh PRIVATE RXMesh CUDA::cudart)
  SET_TARGET_PROPERTIES(rxmesh_aniso_remesh PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
ENDIF()

# Compile static library
IF ( LIBMMGS_STATIC )
  ADD_AND_INSTALL_LIBRARY ( lib${PROJECT_NAME}s_a STATIC copy_s_headers
    "${mmgs_library_files}" ${PROJECT_NAME}s )
  IF ( BUILD_CUDA )
    SET_TARGET_PROPERTIES(lib${PROJECT_NAME}s_a PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    TARGET_LINK_LIBRARIES(lib${PROJECT_NAME}s_a PRIVATE CUDA::cudart)
    # Note: RXMesh bridge NOT linked to libmmgs — it's a separate executable for now
  ENDIF()
ENDIF()

# Compile shared library
IF ( LIBMMGS_SHARED )
  ADD_AND_INSTALL_LIBRARY ( lib${PROJECT_NAME}s_so SHARED copy_s_headers
    "${mmgs_library_files}" ${PROJECT_NAME}s )
  IF ( BUILD_CUDA )
    SET_TARGET_PROPERTIES(lib${PROJECT_NAME}s_so PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    TARGET_LINK_LIBRARIES(lib${PROJECT_NAME}s_so PRIVATE CUDA::cudart)
  ENDIF()
ENDIF()

# mmgs header files needed for library
#
# Remark: header installation would need to be cleaned, for now, to allow
# independent build of each project and because mmgs and mmg2d have been added
# to mmg3d without rethinking the install architecture, the header files that
# are common between codes are copied in all include directories (mmg/,
# mmg/mmg3d/, mmg/mmgs/, mmg/mmg2d/).  they are also copied in build directory
# to enable library call without installation.
SET( mmgs_headers
  ${MMGS_SOURCE_DIR}/mmgs_export.h
  ${MMGS_SOURCE_DIR}/libmmgs.h
  )

IF ( PERL_FOUND )
  LIST ( APPEND mmgs_headers   ${MMGS_BINARY_DIR}/libmmgsf.h )
ENDIF()

IF ( MMG_INSTALL_PRIVATE_HEADERS )
  LIST ( APPEND mmgs_headers
    ${MMGS_SOURCE_DIR}/libmmgs_private.h
    ${MMGS_SOURCE_DIR}/mmgsexterns_private.h
    )
ENDIF()

# install man pages
INSTALL(FILES ${PROJECT_SOURCE_DIR}/doc/man/mmgs.1.gz DESTINATION ${CMAKE_INSTALL_MANDIR}/man1)

# Install header files in /usr/local or equivalent
INSTALL(FILES ${mmgs_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mmg/mmgs COMPONENT headers)

# Copy header files in project directory at build step
COPY_HEADERS_AND_CREATE_TARGET ( ${MMGS_SOURCE_DIR} ${MMGS_BINARY_DIR} ${MMGS_INCLUDE} s )

###############################################################################
#####
#####         Compile MMGS executable
#####
###############################################################################
ADD_AND_INSTALL_EXECUTABLE ( ${PROJECT_NAME}s copy_s_headers
  "${mmgs_library_files}" "${mmgs_main_file}" )

###############################################################################
#####
#####         Continuous integration
#####
###############################################################################

SET(MMG2D_CI_TESTS ${CI_DIR}/mmg2d )
SET(MMGS_CI_TESTS  ${CI_DIR}/mmgs )
SET(MMG_CI_TESTS   ${CI_DIR}/mmg )

##-------------------------------------------------------------------##
##-------------- Library examples and APIs      ---------------------##
##-------------------------------------------------------------------##
IF ( TEST_LIBMMGS )
  # Build executables for library examples and add library tests if needed
  INCLUDE(libmmgs_tests)
ENDIF()

##-------------------------------------------------------------------##
##------------------------ Test Mmgs executable ---------------------##
##-------------------------------------------------------------------##
IF ( BUILD_TESTING )

  # Add runtime that we want to test for mmgs
  IF( MMGS_CI )

    SET ( EXECUT_MMGS      $<TARGET_FILE:${PROJECT_NAME}s> )
    SET ( SHRT_EXECUT_MMGS ${PROJECT_NAME}s)

    IF ( ONLY_VERY_SHORT_TESTS )
      # Add tests that doesn't require to download meshes
      SET ( CTEST_OUTPUT_DIR ${PROJECT_BINARY_DIR}/TEST_OUTPUTS )

      ADD_TEST(NAME mmgs_very_short   COMMAND ${EXECUT_MMGS}
        "${PROJECT_SOURCE_DIR}/libexamples/mmgs/adaptation_example0/example0_a/cube.mesh"
        "${CTEST_OUTPUT_DIR}/libmmgs_Adaptation_0_a-cube.o"
        )

    ELSE ( )
      # Add mmgs tests that require to download meshes
      INCLUDE( mmgs_tests )
    ENDIF ( )

  ENDIF ( MMGS_CI )

ENDIF ( BUILD_TESTING )

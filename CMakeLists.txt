# Instructions:
# $ mkdir build
# $ cd build
# $ cmake ..
# $ make

project(shogun)
cmake_minimum_required(VERSION 2.8.4)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
include(ShogunUtils)

# CCACHE
OPTION(ENABLE_CCACHE "Enable ccache for compilation" ON)
FIND_PACKAGE(CCache)
if(CCACHE_FOUND AND ENABLE_CCACHE)
	SET(CCACHE "ccache" CACHE STRING "ccache")
	SET(CMAKE_C_COMPILER_ARG1 ${CMAKE_C_COMPILER})
	SET(CMAKE_C_COMPILER ${CCACHE})
	SET(CMAKE_CXX_COMPILER_ARG1 ${CMAKE_CXX_COMPILER})
	SET(CMAKE_CXX_COMPILER ${CCACHE})
	SET(CMAKE_C_FLAGS "${CCACHE_FLAGS} ${CMAKE_C_FLAGS}")
	SET(CMAKE_CXX_FLAGS "${CCACHE_FLAGS} ${CMAKE_CXX_FLAGS}")
endif()

SET(CPACK_PACKAGE_CONTACT shogun@shogun-toolbox.org)

################# VARIABLES #####################
SET(DEFINES "SWIG_TYPE_TABLE=shogun;SHOGUN")
SET(INCLUDES "")

SET(EXT_SRC_HEADER "h")
SET(EXT_SRC_C "c")
SET(EXT_SRC_CPP "cpp")
SET(EXT_SRC_SWIG "_wrap.cxx")
SET(EXT_SRC_TEST "_unittest.cc")
SET(EXT_INP_SWIG "i")
SET(EXT_IF_SWIG "py")
SET(EXT_CPP_TMP "${EXT_SRC_CPP}.templ")
SET(EXT_CPP_SH "${EXT_SRC_CPP}.sh")
SET(EXT_CPP_PY "${EXT_SRC_CPP}.py")
SET(EXT_SRC_TEST_TMP "${EXT_SRC_TEST}.jinja2")

SET(THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/third_party)
SET(LIBSHOGUN_SRC_DIR ${CMAKE_SOURCE_DIR}/src/shogun)
SET(COMMON_MODULAR_SRC_DIR ${CMAKE_SOURCE_DIR}/src/interfaces/modular/)

# check whether any of the modular interfaces are turned ON
IF (
	PythonModular OR LuaModular OR RModular OR
	OctaveModular OR JavaModular OR PerlModular OR
	RubyModular OR CSharpModular
   )
	FIND_PACKAGE(SWIG 2.0.4 REQUIRED)
	#TODO: add ccache-swig support
#	IF(ENABLE_CCACHE)
#		SET(SWIG_EXECUTABLE /opt/local/bin/ccache-swig)
#	ENDIF()
	SET(COMPILE_MODULAR_INTERFACE 1)
ENDIF()

# Detect OS
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	SET(DARWIN 1)
ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	SET(LINUX 1)
ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
	SET(FREEBSD 1)
ENDIF()

SET(EXT_LIB_SWIG_RUBY_MODULAR ".so")
if(DARWIN)
	SET(EXT_LIB_SWIG_RUBY_MODULAR ".bundle")
ENDIF()

################ COMPILER #######################
# g++ version needs to be => 4.3
IF(CMAKE_COMPILER_IS_GNUCXX)
	# in order to support cmake 2.8.7 and older
	IF(NOT CMAKE_CXX_COMPILER_VERSION)
		include(CheckCompiler)
	ENDIF()
	IF("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS "4.3.0")
		MESSAGE(FATAL_ERROR "g++ version is too old")
	ENDIF()
ENDIF()

#Build type
if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING
       "Choose the type of build, options are: Debug Release Distribution."
       FORCE )
endif()

# set the flags for the build types
IF(MSVC)
	SET(COMPILER_WARNINGS "/Wall")
ELSE()
	SET(COMPILER_WARNINGS "-Wall -Wno-unused-parameter -Wformat -Wformat-security -Wparentheses -Wshadow -Wno-unknown-pragmas -Wno-deprecated")
ENDIF()

IF(CMAKE_COMPILER_IS_GNUCXX)
	SET(RELEASE_COMPILER_FLAGS "-fexpensive-optimizations -frerun-cse-after-loop -fcse-follow-jumps -finline-functions -fschedule-insns2 -fthread-jumps -fforce-addr -fstrength-reduce -funroll-loops -mfpmath=sse")
ELSEIF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
	SET(RELEASE_COMPILER_FLAGS "-finline-functions -funroll-loops")	
ENDIF()
SET(SWIG_CXX_COMPILER_FLAGS "-O0 -g")
SET(CMAKE_C_FLAGS "${COMPILER_WARNINGS} -g ${CMAKE_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${COMPILER_WARNINGS} -g ${CMAKE_CXX_FLAGS}")
SET(CMAKE_C_FLAGS_RELEASE "-O3 ${RELEASE_COMPILER_FLAGS}")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 ${RELEASE_COMPILER_FLAGS}")
SET(CMAKE_C_FLAGS_DISTRIBUTION "-O2")
SET(CMAKE_CXX_FLAGS_DISTRIBUTION "-O2")

# check for supported c++11 features
#
# clang with -std=c++11 and -stdlib=libc++ does not work
# well with swig generated cxx
# hence disable c++11 for this case.
IF (NOT ((CYGWIN AND ENABLE_TESTING) OR (DARWIN AND COMPILE_MODULAR_INTERFACE)))
	INCLUDE(CheckCXX11Features)

	IF(_HAS_CXX11_FLAG)
		LIST(APPEND DEFINES HAVE_CXX11)
		SET(HAVE_CXX11 1)
		SET(CMAKE_CXX_FLAGS "${CXX11_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")
		SET(SWIG_CXX_COMPILER_FLAGS "${CXX11_COMPILER_FLAGS} ${SWIG_CXX_COMPILER_FLAGS}")
	ENDIF()

	IF(_HAS_CXX0X_FLAG)
		LIST(APPEND DEFINES HAVE_CXX0X)
		SET(HAVE_CXX0X 1)
		SET(CMAKE_CXX_FLAGS "${CXX11_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")
		SET(SWIG_CXX_COMPILER_FLAGS "${CXX11_COMPILER_FLAGS} ${SWIG_CXX_COMPILER_FLAGS}")
	ENDIF()

	IF (HAVE_CXX11_ATOMIC)
		LIST(APPEND DEFINES HAVE_CXX11_ATOMIC)
	ENDIF()
ENDIF()

IF(CMAKE_BUILD_TYPE MATCHES "Release")
	# there's a bug on FreeBSD -march=native.
	# for some CPU types it does not detect the right flags
	# see: http://www.freebsd.org/cgi/query-pr.cgi?pr=178792&cat=
	IF (NOT (FREEBSD AND ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")))
		SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
		SET(SWIG_CXX_COMPILER_FLAGS "${SWIG_CXX_COMPILER_FLAGS} -march=native")
	ENDIF()
ENDIF()

###### MALLOC LIBRARY ###########
SET(EXTERNAL_MALLOC_CFLAGS "-fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
if (MALLOC_REPLACEMENT MATCHES "Jemalloc")
	include(external/Jemalloc)

	SET(USE_JEMALLOC 1)
	LIST(APPEND DEFINES USE_JEMALLOC)
	LIST(INSERT INCLUDES 0 ${Jemalloc_INCLUDE_DIR})
	SET(CMAKE_CXX_FLAGS "${EXTERNAL_MALLOC_CFLAGS} ${CMAKE_CXX_FLAGS}")
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${Jemalloc_LIBRARIES})
elseif(MALLOC_REPLACEMENT MATCHES "TCMalloc")
	include(external/TCMalloc)

	SET(USE_TCMALLOC 1)
	LIST(APPEND DEFINES USE_TCMALLOC)
	LIST(INSERT INCLUDES 0 ${TCMalloc_INCLUDE_DIR})
	SET(CMAKE_CXX_FLAGS "${EXTERNAL_MALLOC_CFLAGS} ${CMAKE_CXX_FLAGS}")
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${TCMalloc_LIBRARIES})	
elseif(MALLOC_REPLACEMENT MATCHES "Hoard")
	find_package(Hoard)
	if (Hoard_FOUND)
		SET(USE_HOARD 1)
		LIST(APPEND DEFINES USE_HOARD)
		SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${Hoard_LIBRARIES})
  	else ()
    	message(FATAL_ERROR "Unable to use hoard malloc: library not found")
  	endif ()
else()
	message(STATUS "Using system's malloc")
endif()

############### VERSION #####################
FILE(STRINGS "@CMAKE_SOURCE_DIR@/NEWS" NEWS LIMIT_COUNT 5)
STRING(REGEX REPLACE ".*SHOGUN Release version ([0-9.]*).*" "\\1" VERSION "${NEWS}")
STRING(REGEX REPLACE ".*SHOGUN Release version.*\\(libshogun ([0-9.]*).*" "\\1" LIBSHOGUNVER "${NEWS}")
STRING(REGEX REPLACE ".*SHOGUN Release version.*\\(libshogun ([0-9]*).*" "\\1" LIBSHOGUNSO "${NEWS}")
STRING(REGEX REPLACE ".*SHOGUN Release version.*data ([0-9.]*).*" "\\1" DATAVER "${NEWS}")
STRING(REGEX REPLACE "([0-9]*).[0-9]*.[0-9]*" "\\1" SHOGUN_VERSION_MAJOR "${VERSION}")
STRING(REGEX REPLACE "[0-9]*.([0-9]*).[0-9]*" "\\1" SHOGUN_VERSION_MINOR "${VERSION}")
STRING(REGEX REPLACE "[0-9]*.[0-9]*.([0-9]*)" "\\1" SHOGUN_VERSION_PATCH "${VERSION}")

# Get processor type, sets MACHINE macro
IF(NOT WIN32)
	execute_process(COMMAND uname -m
	    OUTPUT_VARIABLE MACHINE_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE)
	SET(MACHINE ${MACHINE_OUTPUT})
ENDIF()


SET(BUILD_STATIC "Build a statically linked binary" OFF)
################# EXAMPLES ##################
OPTION(BUILD_EXAMPLES "Build Examples" ON)

################# DATATYPES #################
LIST(APPEND DEFINES USE_CHAR)
LIST(APPEND DEFINES USE_BOOL)
LIST(APPEND DEFINES USE_UINT8)
LIST(APPEND DEFINES USE_UINT16)
#LIST(APPEND DEFINES USE_UINT32)
LIST(APPEND DEFINES USE_UINT64)
#LIST(APPEND DEFINES USE_INT8)
#LIST(APPEND DEFINES USE_INT16)
LIST(APPEND DEFINES USE_INT32)
LIST(APPEND DEFINES USE_INT64)
LIST(APPEND DEFINES USE_FLOAT32)
LIST(APPEND DEFINES USE_FLOAT64)
#LIST(APPEND DEFINES USE_FLOATMAX)

IF(MSVC)
	include(external/MSIntTypes)
	LIST(APPEND INCLUDES ${THIRD_PARTY_DIR}/MSIntTypes)

	include(external/MSDirent)
	LIST(APPEND INCLUDES ${MSDIRENT_INCLUDE_DIR})
ENDIF()

# detect word size
IF(CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT DARWIN)
	SET(CMAKE_SWIG_FLAGS -DSWIGWORDSIZE64)
ENDIF()

#interfaces
OPTION(PythonModular "Python Modular" OFF)
OPTION(LuaModular "Lua Modular" OFF)
OPTION(RModular "R Modular" OFF)
OPTION(OctaveModular "Octave Modular" OFF)
OPTION(JavaModular "Java Modular" OFF)
OPTION(PerlModular "Perl Modular" OFF)
OPTION(RubyModular "Ruby Modular" OFF)
OPTION(CSharpModular "CSharp Modular" OFF)

OPTION(PythonStatic "Python Static" OFF)
OPTION(RStatic "R Static" OFF)
OPTION(CmdLineStatic "CmdLine Static" OFF)
OPTION(MatlabStatic "Matlab Static" OFF)
OPTION(OctaveStatic "Octave Static" OFF)

# SVMLight
OPTION(USE_SVMLIGHT "SVMLight" ON)
SET(SVMLightWarning "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
SET(SVMLightWarning "${SVMLightWarning}\nWARNING: SHOGUN is built using SVMlight which was written")
SET(SVMLightWarning "${SVMLightWarning}\nby Thorsten Joachims and uses a different non GPL compatible license.")
SET(SVMLightWarning "${SVMLightWarning}\nTo build a fully GPL'd SHOGUN use")
SET(SVMLightWarning "${SVMLightWarning}\nThe SVMlight license can be found in LICENSE.SVMlight.  In case")
SET(SVMLightWarning "${SVMLightWarning}\nyou do not know or cannot agree to the licensing terms expressed in")
SET(SVMLightWarning "${SVMLightWarning}\nLICENSE.SVMlight press ctrl+c to abort configure now.")
SET(SVMLightWarning "${SVMLightWarning}\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
IF(USE_SVMLIGHT)
	LIST(APPEND DEFINES USE_SVMLIGHT)
ENDIF()

#Trace memory allocs
OPTION(TRACE_MEMORY_ALLOCS "Memory allocation tracing" OFF)
IF(TRACE_MEMORY_ALLOCS)
	LIST(APPEND DEFINES TRACE_MEMORY_ALLOCS)
ENDIF()

# HMM
OPTION(USE_HMMDEBUG "HMM cache" OFF)
IF(USE_HMMDEBUG)
	LIST(APPEND DEFINES USE_HMMDEBUG)
ENDIF()

OPTION(USE_HMMCACHE "HMM cache" ON)
IF(USE_HMMCACHE)
	LIST(APPEND DEFINES USE_HMMCACHE)
ENDIF()

OPTION(USE_HMMPARALLEL "Parallel structures in hmm training" OFF)
IF(USE_HMMPARALLEL)
	SET(USE_HMMPARALLELSTRUCTURES 1)
	LIST(APPEND DEFINES USE_HMMPARALLELSTRUCTURES USE_HMMPARALLEL)
ENDIF()

# Viterbi path debug
OPTION(USE_PATHDEBUG "Viterbi path debugging" OFF)
IF(USE_PATHDEBUG)
	LIST(APPEND DEFINES USE_PATHDEBUG)
ENDIF()

# big states
OPTION(USE_BIGSTATES "Big (16bit) state" ON)
IF(USE_BIGSTATES)
	LIST(APPEND DEFINES USE_BIGSTATES)
ENDIF()

# Large file
OPTION(HAVE_LARGEFILE "Large file support" ON)
IF(HAVE_LARGEFILE)
	LIST(APPEND DEFINES HAVE_LARGEFILE)
ENDIF()

#kernelcache to use 4-byte-floating-point values instead of 8-byte-doubles
OPTION(USE_SHORTREAL_KERNELCACHE "Kernelcache to use 4-byte-floating-point values instead of 8-byte-doubles" ON)
IF(USE_SHORTREAL_KERNELCACHE)
	LIST(APPEND DEFINES USE_SHORTREAL_KERNELCACHE)
ENDIF()

# PRNG settings
SET(MEXP "19937" CACHE STRING "PRNG Mersenne exponent")
SET(SFMT_MEXP ${MEXP})
SET(DSFMT_MEXP ${MEXP})
LIST(APPEND DEFINES DSFMT_MEXP=${MEXP} SFMT_MEXP=${MEXP})

# Reference counting
OPTION(USE_REFERENCE_COUNTING "Reference Counting" ON)
IF(USE_REFERENCE_COUNTING)
	LIST(APPEND DEFINES USE_REFERENCE_COUNTING)
ENDIF()

OPTION(USE_LOGCACHE "Use (1+exp(x)) log cache (is much faster but less accurate)" OFF)
IF(USE_LOGCACHE)
	LIST(APPEND DEFINES USE_LOGCACHE)
ENDIF()

OPTION(USE_LOGSUMARRAY "Use sum array, supposed to be a bit more accurate" OFF)
IF(USE_LOGSUMARRAY)
	LIST(APPEND DEFINES USE_LOGSUMARRAY)
ENDIF()

######################### LIBRARIES #########################

# check for math functions
include(CheckFunctionExists)
IF(UNIX)
	SET(CMAKE_REQUIRED_LIBRARIES m)
ENDIF()
CHECK_FUNCTION_EXISTS(log2 HAVE_LOG2)
IF(HAVE_LOG2)
	LIST(APPEND DEFINES HAVE_LOG2)
ENDIF()

CHECK_FUNCTION_EXISTS(powl HAVE_POWL)
IF(HAVE_POWL)
	LIST(APPEND DEFINES HAVE_POWL)
ENDIF()

CHECK_FUNCTION_EXISTS(lgammal HAVE_LGAMMAL)
IF(HAVE_LGAMMAL)
	LIST(APPEND DEFINES HAVE_LGAMMAL)
ENDIF()

CHECK_FUNCTION_EXISTS(sqrtl HAVE_SQRTL)
IF(HAVE_SQRTL)
	LIST(APPEND DEFINES HAVE_SQRTL)
ENDIF()

# check SSE and SSE2 intrinsics header
include(CheckIncludeFile)
CHECK_INCLUDE_FILE(xmmintrin.h HAVE_BUILTIN_VECTOR)
CHECK_INCLUDE_FILE(emmintrin.h HAVE_SSE2)
IF(HAVE_SSE2)
	LIST(APPEND DEFINES HAVE_SSE2)
ENDIF()

FIND_PACKAGE(Threads)
IF (CMAKE_USE_PTHREADS_INIT)
	SET(HAVE_PTHREAD 1)
	LIST(APPEND DEFINES HAVE_PTHREAD)
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${CMAKE_THREAD_LIBS_INIT})
ENDIF()

FIND_PACKAGE(OpenMP)
if (OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

FIND_PACKAGE(PkgConfig)

FIND_PACKAGE(LAPACK)
IF (LAPACK_FOUND)
	# find out whether it is Accelerate.framework we found for LaPack/BLAS
	IF("${LAPACK_LIBRARIES}" MATCHES ".*/Accelerate.framework$")
		SET(HAVE_MVEC 1)
		LIST(APPEND DEFINES HAVE_MVEC)

		#Accelerate framework provides HAVE_CATLAS
		SET(HAVE_CATLAS 1)
		LIST(APPEND DEFINES HAVE_CATLAS)

		SET(HAVE_LAPACK 1)
		LIST(APPEND DEFINES HAVE_LAPACK)
		SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${LAPACK_LIBRARIES})
	ELSEIF("${LAPACK_LIBRARIES}" MATCHES ".*/mkl_.*")
		SET(HAVE_LAPACK 1)
		SET(HAVE_MKL 1)
		LIST(APPEND DEFINES HAVE_LAPACK HAVE_MKL)
	ELSE()
		FIND_PACKAGE(Atlas)
		IF(ATLAS_FOUND)
			SET(HAVE_ATLAS 1)
			SET(HAVE_LAPACK 1)
			LIST(APPEND DEFINES HAVE_LAPACK HAVE_ATLAS)
			LIST(APPEND INCLUDES ${ATLAS_INCLUDES})
			SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${ATLAS_LIBRARIES})
		ELSE()
			FIND_PACKAGE(CBLAS)
		    if(CBLAS_LIBRARY)
		    	SET(HAVE_LAPACK 1)
		    	LIST(APPEND DEFINES HAVE_LAPACK)
				SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${LAPACK_LIBRARIES} ${CBLAS_LIBRARY})
			else()
				SET(HAVE_LAPACK 1)
				LIST(APPEND DEFINES HAVE_LAPACK)
				SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${LAPACK_LIBRARIES})
		    endif()
		ENDIF()
	ENDIF()
ENDIF()

FIND_PACKAGE(GLPK)
IF (GLPK_FOUND)
	SET(USE_GLPK 1)
	LIST(APPEND DEFINES USE_GLPK)
	LIST(APPEND INCLUDES ${GLPK_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${GLPK_LIBRARY})
ENDIF()

FIND_PACKAGE(CPLEX)
IF (CPLEX_FOUND)
	SET(USE_CPLEX 1)
	LIST(APPEND DEFINES USE_CPLEX)
	LIST(APPEND INCLUDES ${CPLEX_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${CPLEX_LIBRARY})
ENDIF()

FIND_PACKAGE(ARPACK)
IF (ARPACK_FOUND)
	SET(HAVE_ARPACK 1)
	LIST(APPEND DEFINES HAVE_ARPACK)
	LIST(APPEND INCLUDES ${ARPACK_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${ARPACK_LIB})
ENDIF()

OPTION(BUNDLE_EIGEN "Bundle Eigen" OFF)
IF(BUNDLE_EIGEN)
	include(external/Eigen3)
	SET(HAVE_EIGEN3 1)
	LIST(APPEND DEFINES HAVE_EIGEN3)
	LIST(APPEND SYSTEM_INCLUDES ${EIGEN_INCLUDE_DIR})
ELSE()
	FIND_PACKAGE(Eigen3 3.1.2)
	IF (EIGEN3_FOUND)
		SET(HAVE_EIGEN3 1)
		LIST(APPEND DEFINES HAVE_EIGEN3)
		LIST(APPEND SYSTEM_INCLUDES ${EIGEN_INCLUDE_DIR})
	ENDIF()
ENDIF()

OPTION(BUNDLE_NLOPT "Bundle NLOPT" OFF)
IF(BUNDLE_NLOPT)
	include(external/NLopt)
	SET(HAVE_NLOPT 1)
	LIST(APPEND DEFINES HAVE_NLOPT)
	LIST(APPEND INCLUDES ${NLOPT_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${NLOPT_LIBRARIES})
ELSE()
	FIND_PACKAGE(NLopt)
	IF (NLOPT_FOUND)
		SET(HAVE_NLOPT 1)
		LIST(APPEND DEFINES HAVE_NLOPT)
		LIST(APPEND INCLUDES ${NLOPT_INCLUDE_DIR})
		SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${NLOPT_LIBRARIES})
	ENDIF()
ENDIF()

FIND_PACKAGE(LpSolve)
IF (LPSOLVE_FOUND)
	SET(USE_LPSOLVE 1)
	LIST(APPEND DEFINES USE_LPSOLVE YY_NEVER_INTERACTIVE PARSER_LP INVERSE_ACTIVE=INVERSE_LUSOL RoleIsExternalInvEngine)
	LIST(APPEND INCLUDES ${LPSOLVE_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${LPSOLVE_LIBRARIES})
ENDIF()

FIND_PACKAGE(ColPack)
IF(COLPACK_FOUND)
	SET(HAVE_COLPACK 1)
	LIST(APPEND DEFINES HAVE_COLPACK)
	LIST(APPEND INCLUDES ${COLPACK_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${COLPACK_LIBRARIES})	
ENDIF()

FIND_PACKAGE(Doxygen)
IF(DOXYGEN_FOUND)
	SET(HAVE_DOXYGEN 1)
	LIST(APPEND DEFINES HAVE_DOXYGEN)
ENDIF()

# detect PYTHON
FIND_PACKAGE(PythonInterp REQUIRED)

# JSON
OPTION(BUNDLE_JSON "Bundle JSON" OFF)
IF(BUNDLE_JSON)
	include(external/JSON)
	SET(HAVE_JSON 1)
	LIST(APPEND DEFINES HAVE_JSON)
	LIST(APPEND INCLUDES ${JSON_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${JSON_LDFLAGS})
ELSE()
	IF (PKG_CONFIG_FOUND)
		pkg_search_module(JSON libjson>=0.11 json>=0.11 json-c>=0.11)
		IF (JSON_FOUND)
			SET(HAVE_JSON 1)
			LIST(APPEND DEFINES HAVE_JSON)
			LIST(APPEND INCLUDES ${JSON_INCLUDE_DIRS})
			SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${JSON_LDFLAGS})
		ENDIF()
	ENDIF()
ENDIF()

FIND_PACKAGE(LibXml2)
IF (LIBXML2_FOUND)
	SET(HAVE_XML 1)
	LIST(APPEND DEFINES HAVE_XML)
	LIST(APPEND INCLUDES ${LIBXML2_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${LIBXML2_LIBRARIES})
ENDIF()

FIND_PACKAGE(HDF5)
IF (HDF5_FOUND)
	SET(HAVE_HDF5 1)
	LIST(APPEND DEFINES HAVE_HDF5)
	LIST(APPEND INCLUDES ${HDF5_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${HDF5_LIBRARIES})
ENDIF()

FIND_PACKAGE(ZLIB)
IF (ZLIB_FOUND)
	SET(USE_GZIP 1)
	LIST(APPEND DEFINES USE_GZIP)
	LIST(APPEND INCLUDES ${ZLIB_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${ZLIB_LIBRARIES})
ENDIF()

FIND_PACKAGE(BZip2)
IF (BZIP2_FOUND)
	SET(USE_BZIP2 1)
	LIST(APPEND DEFINES USE_BZIP2)
	LIST(APPEND INCLUDES ${BZIP_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${BZIP2_LIBRARIES})
ENDIF()

FIND_PACKAGE(LibLZMA)
IF (LIBLZMA_FOUND)
	SET(USE_LZMA 1)
	LIST(APPEND DEFINES USE_LZMA)
	LIST(APPEND INCLUDES ${LIBLZMA_INCLUDE_DIRS})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${LIBLZMA_LIBRARIES})
ENDIF()

FIND_PACKAGE(SNAPPY)
IF (SNAPPY_FOUND)
	SET(USE_SNAPPY 1)
	LIST(APPEND DEFINES USE_SNAPPY)
	LIST(APPEND INCLUDES ${SNAPPY_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${SNAPPY_LIBRARIES})
ENDIF()

FIND_PACKAGE(LZO)
IF (LZO_FOUND)
	SET(USE_LZO 1)
	LIST(APPEND DEFINES USE_LZO)
	LIST(APPEND INCLUDES ${LZO_INCLUDE_DIR})
	SET(POSTLINKFLAGS ${POSTLINKFLAGS} ${LZO_LIBS})
ENDIF()

FIND_PACKAGE(Spinlock)
IF (SPINLOCK_FOUND)
	SET(USE_SPINLOCKS 1)
	LIST(APPEND DEFINES USE_SPINLOCKS)
ENDIF()

#SWIG Interfaces
SET(CMAKE_SWIG_FLAGS "${CMAKE_SWIG_FLAGS};-w473;-w454;-w312;-w325;-fvirtual")

OPTION(USE_SWIG_DIRECTORS "" OFF)

# python modular
IF (PythonModular OR PythonStatic)
	# find python library for the found python interpreter
	# i.e. versions should match!
	set(Python_ADDITIONAL_VERSIONS ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
	FIND_PACKAGE(PythonLibs REQUIRED)
	FIND_PACKAGE(NumPy REQUIRED)

	LIST(APPEND DEFINES HAVE_PYTHON)
	
	IF(PythonModular)
		#custom swig flags for python modular interface
		IF(${PYTHON_VERSION_MAJOR} VERSION_EQUAL 3)
			SET(TARGET_SWIGFLAGS "-builtin\;-modern\;-modernargs\;-py3")
			#SET(SWIG_CXX_COMPILER_FLAGS "${SWIG_CXX_COMPILER_FLAGS -DPYTHON3")
		ELSE()
			SET(TARGET_SWIGFLAGS "-builtin\;-modern\;-modernargs")
		ENDIF()

		add_subdirectory(src/interfaces/python_modular)
	ENDIF()

	IF(PythonStatic)
		add_subdirectory(src/interfaces/python_static)
	ENDIF()
ENDIF()

# lua modular
IF (LuaModular)
	#FIND_PACKAGE(Lua51 REQUIRED)
	pkg_search_module(LUA REQUIRED lua<=5.1 lua5.1)
	SET(LUA_EXECUTABLE lua)
	UNSET(TARGET_SWIGFLAGS)
	add_subdirectory(src/interfaces/lua_modular)
ENDIF()

# java modular
IF (JavaModular)
	FIND_PACKAGE(Java REQUIRED)
	FIND_PACKAGE(JNI REQUIRED)

	#find jblas
	INCLUDE(UseJava)
	FIND_JAR(JBLAS NAMES jblas jblas-1.2.0
			PATHS /opt/local/share/java
				  ENV JBLAS_PATH)
	IF(JBLAS)
		LIST(APPEND DEFINES HAVE_JBLAS)
		SET(HAVE_JBLAS 1)
	ELSE()
		MESSAGE(FATAL_ERROR "JBlas could not be found!\n"
			"It is required for java modular interface!!!")
	ENDIF()

	SET(TARGET_SWIGFLAGS "-package\;org.shogun")
	add_subdirectory(src/interfaces/java_modular)
ENDIF()

# ruby modular
IF (RubyModular)
	FIND_PACKAGE(Ruby REQUIRED)
	FIND_PACKAGE(RubyNArray REQUIRED)
	SET(NARRAY_LIB ${RUBY_NARRAY_LIBRARY})
	UNSET(TARGET_SWIGFLAGS)
	add_subdirectory(src/interfaces/ruby_modular)
ENDIF()

# octave modular
IF (OctaveModular OR OctaveStatic)		
	FIND_PACKAGE(Octave 3.6 REQUIRED)
	LIST(APPEND DEFINES HAVE_OCTAVE)
	LIST(APPEND DEFINES "OCTAVE_APIVERSION=${OCTAVE_API_VERSION}")
	SET(OCTAVE_APIVERSION ${OCTAVE_API_VERSION})

	IF(OctaveModular)
		UNSET(TARGET_SWIGFLAGS)
		add_subdirectory(src/interfaces/octave_modular)
	ENDIF()

	IF(OctaveStatic)
		add_subdirectory(src/interfaces/octave_static)
	ENDIF()
ENDIF()

# csharp modular
IF (CSharpModular)
	FIND_PACKAGE(CSharp REQUIRED)
	UNSET(TARGET_SWIGFLAGS)
	add_subdirectory(src/interfaces/csharp_modular)
ENDIF()

# r modular
IF (RModular OR RStatic)
	FIND_PACKAGE(R REQUIRED)

	IF (RModular)
		UNSET(TARGET_SWIGFLAGS)
		add_subdirectory(src/interfaces/r_modular)
	ENDIF()

	IF (RStatic)
		add_subdirectory(src/interfaces/r_static)
	ENDIF()
ENDIF()

# perl modular
IF (PerlModular)
	FIND_PACKAGE(FindPerlLibs REQUIRED)
	UNSET(TARGET_SWIGFLAGS)
	#add_subdirectory(src/interfaces/perl_modular)
ENDIF()

IF (MatlabStatic)
	FIND_PACKAGE(Matlab REQUIRED)
	IF(MATLAB_FOUND)
		add_subdirectory(src/interfaces/matlab_static)
	ELSE()
		MESSAGE(FATAL_ERROR "Could not find Matlab, which is required for compiling matlab_static interface. Try setting MATLAB_ROOT enviroment variable to the right path")
	ENDIF()
ENDIF()

IF (CmdLineStatic)
	add_subdirectory(src/interfaces/cmdline_static)
ENDIF()

IF (SVMLight)
	MESSAGE(STATUS ${SVMLightWarning})
ENDIF()

################# TESTING ###################
#TODO: finish!!!
OPTION(BUILD_DASHBOARD_REPORTS "Set to ON to activate reporting of Shogun builds" OFF)
IF(BUILD_DASHBOARD_REPORTS)
	include(CTest)
ENDIF()

OPTION(ENABLE_TESTING "Enable testing" OFF)
OPTION(TRAVIS_DISABLE_UNIT_TESTS "Disable unit testing to speed up jobs on travis-ci" OFF)
OPTION(TRAVIS_DISABLE_LIBSHOGUN_TESTS "Disable libshogun tests to speed up jobs on travis-ci" OFF)
IF(ENABLE_TESTING)
	FIND_PACKAGE(Jinja2 REQUIRED)
	IF(NOT BUILD_DASHBOARD_REPORTS)	
		enable_testing()
	ENDIF()

	# add integration tests
	add_subdirectory(tests/integration)
	

	# add unit tests
	IF (NOT TRAVIS_DISABLE_UNIT_TESTS)
		add_subdirectory(tests/unit)
	ENDIF()
ENDIF()

# save configuration options
MergeCFLAGS()
SET(CONFIGURE_OPTIONS "TODO")
SET(COMPFLAGS_CPP "${MERGED_CXX_FLAGS}")
SET(LINKFLAGS "${POSTLINKFLAGS}")

add_subdirectory(src/shogun)

IF(BUILD_EXAMPLES OR ENABLE_TESTING)
	add_subdirectory(examples)
ENDIF()

# general cpack settings
set(CPACK_PACKAGE_NAME "shogun")
set(CPACK_PACKAGE_VENDOR "shogun")
#SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/README")
#SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/src/COPYRIGHT")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Shogun Machine Learning Toolbox")
set(CPACK_PACKAGE_VERSION ${VERSION})
set(CPACK_PACKAGE_VERSION_MAJOR ${SHOGUN_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${SHOGUN_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${SHOGUN_VERSION_PATCH})

# debian package settings
set(CPACK_DEB_COMPONENT_INSTALL ON)

# rpm package settings
set(CPACK_RPM_COMPONENT_INSTALL ON)

INCLUDE(CPack)

message(STATUS "===================================================================================================================")
message(STATUS "Summary of Configuration Variables")
message(STATUS "")
message(STATUS "Detected Libraries")
get_cmake_property(_variableNames VARIABLES)

foreach (_variableName ${_variableNames})
IF ( _variableName MATCHES "_FOUND$")
	IF (${_variableName})
		message(STATUS "  ${_variableName}=TRUE")
	ENDIF()
ENDIF()
endforeach()

message(STATUS "")
message(STATUS "Missing Libraries")
foreach (_variableName ${_variableNames})
IF ( _variableName MATCHES "_FOUND$")
	IF (NOT ${_variableName})
		message(STATUS "  ${_variableName}=FALSE")
	ENDIF()
ENDIF()
endforeach()

message(STATUS "===================================================================================================================")
message(STATUS "Enabled Interfaces")

message(STATUS "  libshogun is ON")

IF ( PythonModular )
	message(STATUS "  python modular is ON")
ELSE()
	message(STATUS "  python modular is OFF   - enable with -DPythonModular=ON")
ENDIF()

IF ( OctaveModular )
	message(STATUS "  octave modular is ON")
ELSE()
	message(STATUS "  octave modular is OFF   - enable with -DOctaveModular=ON")
ENDIF()

IF ( JavanModular )
	message(STATUS "  java modular is ON")
ELSE()
	message(STATUS "  java modular is OFF     - enable with -DJavaModular=ON")
ENDIF()

IF ( PerlModular )
	message(STATUS "  perl modular is ON")
ELSE()
	message(STATUS "  perl modular is OFF     - enable with -DPerlModular=ON")
ENDIF()

IF ( RubyModular )
	message(STATUS "  ruby modular is ON")
ELSE()
	message(STATUS "  ruby modular is OFF     - enable with -DRubyModular=ON")
ENDIF()

IF ( CSharpModular )
	message(STATUS "  csharp modular is ON")
ELSE()
	message(STATUS "  csharp modular is OFF   - enable with -DCSharpModular=ON")
ENDIF()

IF ( RModular )
	message(STATUS "  R modular is ON")
ELSE()
	message(STATUS "  R modular is OFF        - enable with -DRModular=ON")
ENDIF()

message(STATUS "")
message(STATUS "Enabled legacy interfaces")
IF ( CmdLineStatic )
	message(STATUS "  cmdline static is ON")
ELSE()
	message(STATUS "  cmdline static is OFF   - enable with -DCmdLineStatic=ON")
ENDIF()

IF ( PythonStatic )
	message(STATUS "  python static is ON")
ELSE()
	message(STATUS "  python static is OFF    - enable with -DPythonStatic=ON")
ENDIF()

IF ( OctaveStatic )
	message(STATUS "  octave static is ON")
ELSE()
	message(STATUS "  octave static is OFF    - enable with -DOctaveStatic=ON")
ENDIF()

IF ( MatlabStatic )
	message(STATUS "  Matlab static is ON")
ELSE()
	message(STATUS "  Matlab static is OFF    - enable with -DMatlabStatic=ON")
ENDIF()

IF ( RStatic )
	message(STATUS "  R static is ON")
ELSE()
	message(STATUS "  R static is OFF         - enable with -DRStatic=ON")
ENDIF()

message(STATUS "===================================================================================================================")

message(STATUS "To compile shogun type")
message(STATUS "  make")
message(STATUS "")
message(STATUS "To install shogun to ${CMAKE_INSTALL_PREFIX} type")
message(STATUS "  make install")
message(STATUS "")
message(STATUS "or to install to a custom directory")
message(STATUS "  make install -DCMAKE_INSTALL_PREFIX=/my/special/path")

message(STATUS "===================================================================================================================")



cmake_minimum_required(VERSION 3.16)
project(btc_predictor VERSION 3.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimisation flags
set(OPT_FLAGS "-O3 -march=native -mfpmath=sse -funroll-loops -ftree-vectorize -ffast-math")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPT_FLAGS}")

# LTO
include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERR)
if(LTO_SUPPORTED)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

# ── pybind11: locate via Python pip package (MSYS2/MinGW-safe) ──────────────
# Priority order:
#   1. User-set pybind11_DIR (cmake -Dpybind11_DIR=...)
#   2. pip-installed package (python -m pybind11 --cmakedir)
#   3. System cmake find_package fallback
if(NOT DEFINED pybind11_DIR OR pybind11_DIR STREQUAL "")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(NOT PYBIND11_CMAKE_DIR STREQUAL "")
    set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
    message(STATUS "pybind11 cmake dir (from pip): ${pybind11_DIR}")
  else()
    # Try python3 explicitly (MSYS2 may not have a bare 'python')
    execute_process(
      COMMAND python3 -m pybind11 --cmakedir
      OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(NOT PYBIND11_CMAKE_DIR STREQUAL "")
      set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
      message(STATUS "pybind11 cmake dir (from python3 pip): ${pybind11_DIR}")
    else()
      # Last resort: try python
      execute_process(
        COMMAND python -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      if(NOT PYBIND11_CMAKE_DIR STREQUAL "")
        set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
        message(STATUS "pybind11 cmake dir (from python pip): ${pybind11_DIR}")
      endif()
    endif()
  endif()
endif()

# ── Python3: find interpreter + development headers ─────────────────────────
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# ── Now find pybind11 using the resolved DIR ─────────────────────────────────
find_package(pybind11 CONFIG REQUIRED)

# ── OpenMP — MinGW/GCC needs -fopenmp injected manually if CMake fails ───────
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found via find_package: ${OpenMP_CXX_FLAGS}")
else()
  # MinGW's CMake OpenMP finder often fails even though GCC supports -fopenmp.
  # Create an INTERFACE target that mimics OpenMP::OpenMP_CXX so the rest of
  # the CMakeLists doesn't need to change.
  message(STATUS "OpenMP not found by CMake — injecting -fopenmp manually (MinGW workaround)")
  add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
  set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
    INTERFACE_COMPILE_OPTIONS "-fopenmp"
    INTERFACE_LINK_LIBRARIES  "-fopenmp"
  )
endif()

# ── BLAS — prefer OpenBLAS from MSYS2, fall back to generic or stub ──────────
find_package(BLAS)
if(NOT BLAS_FOUND)
  # Try locating openblas directly (MSYS2: pacman -S mingw-w64-x86_64-openblas)
  find_library(OPENBLAS_LIB NAMES openblas libopenblas
               HINTS "${MSYS2_ROOT}/mingw64/lib" "C:/msys64/mingw64/lib")
  if(OPENBLAS_LIB)
    message(STATUS "BLAS not found by CMake but OpenBLAS located: ${OPENBLAS_LIB}")
    set(BLAS_LIBRARIES "${OPENBLAS_LIB}")
    set(BLAS_FOUND TRUE)
  else()
    message(WARNING "BLAS not found — engine will compile without BLAS acceleration.\n"
                    "Install with: pacman -S mingw-w64-x86_64-openblas")
    set(BLAS_LIBRARIES "")
  endif()
endif()

# Try to locate cblas.h — OpenBLAS on MSYS2 puts it in openblas/ subdirectory
find_path(CBLAS_INCLUDE_DIR
  NAMES cblas.h openblas/cblas.h
  HINTS
    "${MSYS2_ROOT}/mingw64/include"
    "C:/msys64/mingw64/include"
    "C:/msys64/mingw64/include/openblas"
    "/usr/include"
    "/usr/include/openblas"
)
if(CBLAS_INCLUDE_DIR)
  message(STATUS "cblas.h found: ${CBLAS_INCLUDE_DIR}")
  add_compile_definitions(HAVE_CBLAS)
else()
  message(STATUS "cblas.h not found — BLAS GEMM calls will be disabled in core.cpp")
endif()

# ── CURL — required for CoinGecko/Binance HTTP calls ─────────────────────────
find_package(CURL)
if(NOT CURL_FOUND)
  # Try MSYS2 libcurl directly
  find_library(CURL_LIB NAMES curl libcurl
               HINTS "${MSYS2_ROOT}/mingw64/lib" "C:/msys64/mingw64/lib")
  find_path(CURL_INCLUDE_DIR curl/curl.h
            HINTS "${MSYS2_ROOT}/mingw64/include" "C:/msys64/mingw64/include")
  if(CURL_LIB AND CURL_INCLUDE_DIR)
    message(STATUS "CURL located manually: ${CURL_LIB}")
    add_library(CURL::libcurl INTERFACE IMPORTED)
    set_target_properties(CURL::libcurl PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${CURL_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES      "${CURL_LIB}"
    )
    set(CURL_FOUND TRUE)
  else()
    message(FATAL_ERROR "libcurl not found.\n"
                        "Install with: pacman -S mingw-w64-x86_64-curl")
  endif()
endif()

# ── Optional NUMA (Linux only — no-op on Windows) ────────────────────────────
find_library(NUMA_LIB numa)
if(NUMA_LIB)
  add_compile_definitions(HAVE_NUMA)
  message(STATUS "NUMA support enabled")
endif()

# ── Core shared library (12 models) ──────────────────────────────────────────
add_library(btc_core SHARED
  core.cpp
  optimization.cpp
  external.cpp
)
target_include_directories(btc_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  $<$<BOOL:${CBLAS_INCLUDE_DIR}>:${CBLAS_INCLUDE_DIR}>
)
target_link_libraries(btc_core
  PUBLIC  OpenMP::OpenMP_CXX
  PRIVATE CURL::libcurl
)
if(BLAS_FOUND AND BLAS_LIBRARIES)
  target_link_libraries(btc_core PRIVATE ${BLAS_LIBRARIES})
endif()
if(NUMA_LIB)
  target_link_libraries(btc_core PRIVATE ${NUMA_LIB})
endif()

# ── pybind11 extension module ─────────────────────────────────────────────────
pybind11_add_module(btc_engine bindings.cpp)
target_link_libraries(btc_engine PRIVATE btc_core)

install(TARGETS btc_engine DESTINATION .)
message(STATUS "BTC Predictor v3 — 12 models: LSTM Transformer TCN WaveNet N-BEATS Informer NHiTS TFT PatchTST TimesNet DLinear Crossformer")

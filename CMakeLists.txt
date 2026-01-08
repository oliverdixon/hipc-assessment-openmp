cmake_minimum_required(VERSION 4.0)
project(HIPC_Assessment_OpenMP LANGUAGES C)

include(CheckCCompilerFlag)
check_c_compiler_flag(-Wunknown-warning-option WARN_UNKNOWN_WARNING)
if (WARN_UNKNOWN_WARNING)
    add_compile_options(-Wno-unknown-warning-option)
endif()

find_package(OpenMP REQUIRED COMPONENTS C)

add_executable(Airfoil_OpenMP
        Source/airfoil.c
        Source/instance.c
        Source/instance.h
        Source/region.c
        Source/region.h
        Source/types.h
)

set_property(TARGET Airfoil_OpenMP PROPERTY CMAKE_C_STANDARD 11)
target_link_libraries(Airfoil_OpenMP PRIVATE OpenMP::OpenMP_C m)

if (CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM")
    target_link_options(Airfoil_OpenMP PRIVATE -fuse-ld=lld)
    target_compile_options(Airfoil_OpenMP PRIVATE
            -axCORE-AVX2
            -fast
            -Rpass=loop-vectorize
            -Rpass-analysis=loop-vectorize
            -Rpass-missed=loop-vectorize
            -fsave-optimization-record
    )
endif ()

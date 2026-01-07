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

target_link_libraries(Airfoil_OpenMP PRIVATE OpenMP::OpenMP_C m)

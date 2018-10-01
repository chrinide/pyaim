#include_directories(${CINT_INCLUDE_DIR})

add_library(aim SHARED
  surf.c)

set_target_properties(aim PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(aim cgto dft)

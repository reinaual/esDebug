set(ICCShapes_SRC
    iccWall.cpp
    iccShape.cpp
)
add_library(iccShapes SHARED ${ICCShapes_SRC})
install(TARGETS iccShapes LIBRARY DESTINATION ${LIBDIR})
set_target_properties(iccShapes PROPERTIES SOVERSION ${SOVERSION})
add_dependencies(iccShapes EspressoConfig)
target_include_directories(iccShapes PRIVATE ${CMAKE_SOURCE_DIR}/src/core ${CMAKE_BINARY_DIR}/src/core)

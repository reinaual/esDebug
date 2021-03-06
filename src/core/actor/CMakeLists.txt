set(Actor_SRC
    ActorList.cpp
    DipolarBarnesHut.cpp
    DipolarDirectSum.cpp
    HarmonicOrientationWell.cpp
    HarmonicWell.cpp
    Mmm1dgpuForce.cpp
)

add_library(Actor SHARED ${Actor_SRC})
install(TARGETS Actor LIBRARY DESTINATION ${LIBDIR} ARCHIVE DESTINATION ${LIBDIR})
set_target_properties(Actor PROPERTIES SOVERSION ${SOVERSION})
add_dependencies(Actor EspressoConfig)
target_include_directories(Actor PRIVATE ${CMAKE_SOURCE_DIR}/src/core ${CMAKE_BINARY_DIR}/src/core)

if(CUDA)
  set(ActorCuda_SRC
        DipolarBarnesHut_cuda.cu
        DipolarDirectSum_cuda.cu
        HarmonicOrientationWell_cuda.cu
        HarmonicWell_cuda.cu
        Mmm1dgpuForce_cuda.cu
  )
  cuda_add_library(ActorCuda SHARED ${ActorCuda_SRC})
  install(TARGETS ActorCuda DESTINATION ${LIBDIR})
  set_target_properties(ActorCuda PROPERTIES SOVERSION ${SOVERSION})
  add_dependencies(ActorCuda EspressoConfig)

  add_dependencies(Actor ActorCuda)
  CUDA_ADD_CUFFT_TO_TARGET(ActorCuda)
  target_link_libraries(Actor ActorCuda)
endif(CUDA)

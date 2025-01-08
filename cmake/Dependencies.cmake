# These lists are later turned into target properties on main KuiperLLama library target
set(KuiperLLama_LINKER_LIBS "")
set(KuiperLLama_INCLUDE_DIRS "")
set(KuiperLLama_DEFINITIONS "")
set(KuiperLLama_COMPILE_OPTIONS "")


if(USE_CPM)
  # Use CPM to manage dependencies
  include(cmake/CPM.cmake)

  CPMAddPackage(
    NAME GTest
    GITHUB_REPOSITORY google/googletest
    VERSION 1.15.0
  )

  CPMAddPackage(
    NAME glog
    GITHUB_REPOSITORY google/glog
    VERSION 0.7.1
    OPTIONS "BUILD_TESTING Off"
  )

  CPMAddPackage(
    NAME Armadillo
    GITLAB_REPOSITORY conradsnicta/armadillo-code
    GIT_TAG 14.0.1
  )

  CPMAddPackage(
    NAME sentencepiece
    GITHUB_REPOSITORY google/sentencepiece
    VERSION 0.2.0
  )
  find_package(sentencepiece REQUIRED)

  if (LLAMA3_SUPPORT OR QWEN2_SUPPORT)
    CPMAddPackage(
        NAME absl
        GITHUB_REPOSITORY abseil/abseil-cpp
        GIT_TAG 20240722.0
        OPTIONS "BUILD_TESTING Off" "ABSL_PROPAGATE_CXX_STD ON" "ABSL_ENABLE_INSTALL ON"
    )
    CPMAddPackage(
        NAME re2
        GITHUB_REPOSITORY google/re2
        GIT_TAG 2024-07-02
    )
    CPMAddPackage(
        NAME nlohmann_json
        GITHUB_REPOSITORY nlohmann/json
        VERSION 3.11.3
    )
  endif()
endif()

# -----| phtread
if(!WIN32)
    list(APPEND KuiperLLama_LINKER_LIBS PUBLIC pthread)
endif()

# -----| OpenMP
find_package(OpenMP REQUIRED)
list(APPEND KuiperLLama_LINKER_LIBS PUBLIC OpenMP::OpenMP_CXX)
list(APPEND KuiperLLama_COMPILE_OPTIONS PUBLIC ${OpenMP_CXX_FLAGS})

# -----| glog
find_package(glog REQUIRED)
list(APPEND KuiperLLama_INCLUDE_DIRS PUBLIC ${glog_INCLUDE_DIRS})
list(APPEND KuiperLLama_LINKER_LIBS PUBLIC glog::glog)

# -----| Armadillo
find_package(Armadillo REQUIRED)
list(APPEND KuiperLLama_INCLUDE_DIRS PUBLIC ${ARMADILLO_INCLUDE_DIRS})
list(APPEND KuiperLLama_LINKER_LIBS PUBLIC ${ARMADILLO_LIBRARIES})


# ---------------------------------

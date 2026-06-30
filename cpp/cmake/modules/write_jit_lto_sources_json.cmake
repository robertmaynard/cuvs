# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Write ${CMAKE_BINARY_DIR}/jit_lto_sources.json from the global properties
# accumulated by generate_jit_lto_kernels().  No-op in non-Debug builds.
function(cuvs_write_jit_lto_sources_json)
  if(NOT CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    return()
  endif()

  get_property(tags  GLOBAL PROPERTY _JIT_LTO_COVERAGE_TAGS)
  get_property(ksrcs GLOBAL PROPERTY _JIT_LTO_COVERAGE_KERNEL_SOURCES)
  get_property(msrcs GLOBAL PROPERTY _JIT_LTO_COVERAGE_MATRIX_SOURCES)

  if(NOT tags)
    return()
  endif()

  get_filename_component(repo_root "${CMAKE_SOURCE_DIR}" DIRECTORY)

  list(LENGTH tags tag_count)
  math(EXPR last "${tag_count} - 1")

  set(json "{\n  \"fragment_tags\": {")
  foreach(i RANGE "${last}")
    list(GET tags   ${i} tag)
    list(GET ksrcs  ${i} ksrc)
    list(GET msrcs  ${i} msrc)

    file(RELATIVE_PATH ksrc_rel "${repo_root}" "${ksrc}")
    if(msrc)
      file(RELATIVE_PATH msrc_rel "${repo_root}" "${msrc}")
    else()
      set(msrc_rel "")
    endif()

    if(i GREATER 0)
      string(APPEND json ",")
    endif()
    string(APPEND json
      "\n    \"${tag}\": {\"kernel_source\": \"${ksrc_rel}\", \"matrix_source\": \"${msrc_rel}\"}"
    )
  endforeach()

  string(APPEND json "\n  }\n}\n")
  file(WRITE "${CMAKE_BINARY_DIR}/jit_lto_sources.json" "${json}")
  message(STATUS "Written ${CMAKE_BINARY_DIR}/jit_lto_sources.json (${tag_count} entries)")
endfunction()

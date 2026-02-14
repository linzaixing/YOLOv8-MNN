
if(WIN32)
    add_library(mnn STATIC IMPORTED GLOBAL)
    find_library(MNN_LIB_DEBUG
        NAMES MNN_D
        PATHS ${CMAKE_CURRENT_SOURCE_DIR}/win64/lib/debug
        )
    find_library(MNN_LIB_RELEASE
        NAMES MNN
        PATHS ${CMAKE_CURRENT_SOURCE_DIR}/win64/lib/release
        )
    set_target_properties(mnn PROPERTIES
        IMPORTED_LOCATION_DEBUG ${MNN_LIB_DEBUG}
        IMPORTED_LOCATION_RELEASE ${MNN_LIB_RELEASE} 
        IMPORTED_LOCATION_RELWITHDEBINFO ${MNN_LIB_RELEASE}
        IMPORTED_LOCATION_MINSIZEREL ${MNN_LIB_RELEASE}
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/win64/include"
        )

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/win64/lib/debug/MNN_D.dll
        ${EXECUTABLE_OUTPUT_PATH}/Debug/MNN_D.dll
        COPYONLY
        )
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/win64/lib/release/MNN.dll
        ${EXECUTABLE_OUTPUT_PATH}/Release/MNN.dll
        COPYONLY
        )
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/win64/lib/release/MNN.dll
        ${EXECUTABLE_OUTPUT_PATH}/RelWithDebInfo/MNN.dll
        COPYONLY
        )
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/win64/lib/release/MNN.lib  DESTINATION lib)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/win64/lib/release/MNN.dll  DESTINATION lib)
elseif(UNIX AND NOT APPLE)
    if(QEMU_ARMV7L)
        add_library(mnn STATIC IMPORTED GLOBAL)
        set_target_properties(mnn PROPERTIES
            IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/linux/armv7l/lib/libMNN_acc.a
            INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/linux/armv7l/include
            )
        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/linux/armv7l/lib/libMNN.a  DESTINATION lib)
    else()
        add_library(mnn SHARED IMPORTED GLOBAL)
        set_target_properties(mnn PROPERTIES
            IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/linux/x86_64/lib/libMNN.so
            INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/linux/x86_64/include
            )
        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/linux/x86_64/lib/libMNN.so  DESTINATION lib)
    endif()
elseif(ANDROID)
    add_library(mnn SHARED IMPORTED GLOBAL)
    set_target_properties(mnn PROPERTIES
        IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/android/lib/${ANDROID_ABI}/libMNN.so"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/android/include"
        )
endif()
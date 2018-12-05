find_library(systemd_LIBRARY systemd DOC "systemd runtime library")
find_path(systemd_INCLUDE_DIR "systemd/sd-daemon.h" DOC "systemd include directory")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(systemd
	FOUND_VAR systemd_FOUND
	REQUIRED_VARS systemd_LIBRARY systemd_INCLUDE_DIR
)

if (systemd_FOUND)
	add_library(systemd SHARED IMPORTED)
	set_target_properties(systemd PROPERTIES
		IMPORTED_LOCATION "${systemd_LIBRARY}"
		INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${systemd_INCLUDE_DIR}"
	)

	set(systemd_LIBRARIES    "${systemd_LIBRARY}")
	set(systemd_INCLUDE_DIRS "${systemd_INCLUDE_DIR}")
endif()

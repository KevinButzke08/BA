idf_component_register(SRCS "model.cpp" "main.cpp" "model.h"
                    INCLUDE_DIRS "."
                    REQUIRES spiffs)
                    spiffs_create_partition_image(storage ../partition FLASH_IN_PROJECT)

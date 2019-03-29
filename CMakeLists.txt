# CMakeList.txt: CMake-Projekt für "main". Schließen Sie die Quelle ein, und definieren Sie
# projektspezifische Logik hier.
#
cmake_minimum_required (VERSION 3.8)

add_subdirectory("activation")
add_subdirectory("helpers")

# Fügen Sie der ausführbaren Datei dieses Projekts eine Quelle hinzu.
add_executable (main "main.cpp" "main.h")
# TODO: Fügen Sie bei Bedarf Tests hinzu, und installieren Sie Ziele.

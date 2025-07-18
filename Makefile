CXX = g++
CXXFLAGS = -std=c++17 -I/usr/local/include/opencv4
LDFLAGS = -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui
TARGET = main
SRC = main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

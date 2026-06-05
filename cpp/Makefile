CXX = g++
CXXFLAGS = -std=c++17 -O2 -g -fPIC -I$(FLAME_INCLUDE)

FLAME_PREFIX = /home/yaso/local/tlash-flame
FLAME_INCLUDE = $(FLAME_PREFIX)/include
FLAME_LIBDIR = $(FLAME_PREFIX)/lib
FLAME_STATIC = $(FLAME_LIBDIR)/libflame.a

LDFLAGS =
LIBS = $(FLAME_STATIC) -lflexiblas -lpthread -lm

TARGET = main
SRC = main.cpp jacobi_config.cpp tensor_utils.cpp jacobi.cpp reporting.cpp partition.cpp angle.cpp checking.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGET) test_pieces
CXX = g++
CXXFLAGS = -std=c++17 -O2 -g -fPIC -I$(FLAME_INCLUDE)

FLAME_PREFIX = /home/yaso/local/tlash-flame
FLAME_INCLUDE = $(FLAME_PREFIX)/include
FLAME_LIBDIR = $(FLAME_PREFIX)/lib
FLAME_STATIC = $(FLAME_LIBDIR)/libflame.a

LDFLAGS =
LIBS = $(FLAME_STATIC) -lflexiblas -lpthread -lm

TARGET = main
SRC = source/main.cpp source/jacobi_config.cpp source/tensor_utils.cpp source/jacobi.cpp source/reporting.cpp source/partition.cpp source/angle.cpp source/checking.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGET) test_pieces

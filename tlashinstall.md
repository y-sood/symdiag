# Tlash / libflame installation notes

This README documents the installation path that was used to get the local Tlash/libflame setup working on this machine.

## Goal

The goal was to install a working local copy of libflame for the Tlash project and make the CUDA build use that copy explicitly instead of an older system installation.

## Final install location

The working local install prefix is:

```bash
/home/yaso/local/tlash-flame
```

The important installed paths are:

```bash
/home/yaso/local/tlash-flame/include
/home/yaso/local/tlash-flame/lib/libflame.a
```

## Problem encountered

The original source tree configured successfully, but the generated build configuration contained unresolved Autotools placeholders such as:

```text
@FLIBS@
@fla_enable_autodetect_f77_ldflags@
@fla_enable_autodetect_f77_name_mangling@
```

That meant the build system in this archived tree was not fully substituting all variables during configuration.

A shared-library build also failed earlier, so the reliable path was to switch to a static-only build.

## Configure command used

The working configuration approach was:

```bash
./configure \
  --prefix=$HOME/local/tlash-flame \
  --enable-static-build \
  --disable-dynamic-build \
  --enable-lapack2flame \
  --disable-supermatrix \
  --with-cc=gcc \
  LIBS="-lflexiblas -lpthread -lm"
```

### Notes

- `--disable-dynamic-build` was important because the shared-library path was the one that failed first.
- The prefix was placed under `$HOME` so the install stays local to the user account.
- `sudo ./configure` is unnecessary for a user-local prefix and should be avoided.

## Patch applied before building

After configuration, the generated file

```bash
config/x86_64-unknown-linux-gnu/config.mk
```

still contained unresolved `@FLIBS@` tokens. The practical workaround was to remove them:

```bash
cp config/x86_64-unknown-linux-gnu/config.mk \
   config/x86_64-unknown-linux-gnu/config.mk.bak
sed -i 's/@FLIBS@//g' config/x86_64-unknown-linux-gnu/config.mk
```

This was enough to proceed with the static build.

## Build and install

After patching `config.mk`, the project was built and installed with:

```bash
make
make install
```

The install output confirmed the static archive and headers were installed, including:

```text
Installing libflame-x86_64-r.a into /home/yaso/local/tlash-flame/lib/
Installing symlink libflame.a into /home/yaso/local/tlash-flame/lib/
Installing C header files into /home/yaso/local/tlash-flame/include-x86_64-r
Installing symlink include into /home/yaso/local/tlash-flame/
```

## Checking for multiple libflame installations

Because an older libflame installation may exist elsewhere on the system, it is useful to inspect installed files with:

```bash
find /usr /usr/local /home/yaso -type f \( -name "libflame*.a" -o -name "libflame*.so*" -o -name "FLAME.h" \) 2>/dev/null
find /usr /usr/local /home/yaso -type l \( -name "libflame*.a" -o -name "libflame*.so*" \) 2>/dev/null
ldconfig -p | grep flame
```

For this project, the main concern is not just whether multiple copies exist, but which copy the project actually links against.

## Makefile changes that matter

The original Makefile pointed to an older installation:

```make
FLAME_INCLUDE = /usr/local/bin/libflame/include
FLAME_LIB = /usr/local/bin/libflame/lib
```

That must be changed so the build uses the local working install under `/home/yaso/local/tlash-flame`.

A robust version is shown below:

```make
# Compiler
NVCC = nvcc

FLAME_PREFIX  = /home/yaso/local/tlash-flame
FLAME_INCLUDE = $(FLAME_PREFIX)/include
FLAME_LIBDIR  = $(FLAME_PREFIX)/lib
FLAME_STATIC  = $(FLAME_LIBDIR)/libflame.a

NVCC_FLAGS = -I$(FLAME_INCLUDE) -g -Xcompiler -fPIC
LDFLAGS =
LIBS = $(FLAME_STATIC) -lflexiblas -lpthread -lm

TARGET = main
SRC = main.cu partition.cu angle.cu checking.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET) $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGET) test_pieces

run: $(TARGET)
	./$(TARGET)

TEST_SRC = test_pieces.cu partition.cu angle.cu checking.cu

test_pieces: $(TEST_SRC)
	$(NVCC) $(NVCC_FLAGS) $(TEST_SRC) -o test_pieces $(LDFLAGS) $(LIBS)

run_tests: test_pieces
	./test_pieces | python3 test_pieces.py

.PHONY: all clean run test_pieces run_tests
```

## Why the Makefile was changed this way

Two decisions matter here:

1. The include path now points to the local install under `/home/yaso/local/tlash-flame/include`.
2. The link step uses the full static archive path `$(FLAME_STATIC)` instead of `-lflame`.

Using the full path to `libflame.a` reduces the chance of accidentally linking against some older system copy of libflame.

## Recommended verification commands

Before building the CUDA project, verify the installed files:

```bash
ls -l /home/yaso/local/tlash-flame/lib/libflame.a
readlink -f /home/yaso/local/tlash-flame/lib/libflame.a
ls -l /home/yaso/local/tlash-flame/include
```

Then rebuild the project:

```bash
make clean
make
```

## Summary

The working installation path was:

1. Configure libflame as a static-only build under `/home/yaso/local/tlash-flame`.
2. Patch unresolved `@FLIBS@` tokens from generated `config.mk`.
3. Run `make` and `make install`.
4. Update the project Makefile to use the local headers and the exact local `libflame.a` archive.

This ensures the CUDA/Tlash project uses the known-good local installation rather than any older libflame copy elsewhere on the system.


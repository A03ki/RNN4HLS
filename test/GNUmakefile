PROGRAM = lstm
LIBRARY = build/libsinf.so
SRFS = ../src/main.c ../src/sine_wave_inf.c ../src/layer/lstm.c ../src/layer/affine.c ../src/function/sigmoid.c ../src/function/relu.c
SRFS2 = ../src/sine_wave_inf.c ../src/layer/lstm.c ../src/layer/affine.c ../src/function/sigmoid.c ../src/function/relu.c
CC = gcc
OPTION = -shared -fPIC

all: $(PROGRAM) $(LIBRARY)

$(PROGRAM): $(SRFS)
	$(CC) -o $@ $^

$(LIBRARY): $(SRFS2)
	$(CC) $(OPTION) -o $@ $^

.PHONY: clean
clean:
	$(RM) $(PROGRAM) $(LIBRARY)

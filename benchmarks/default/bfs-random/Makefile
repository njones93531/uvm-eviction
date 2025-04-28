CC=hipcc
SRC=$(wildcard *.cpp)
BIN=$(patsubst %.cpp,%,$(SRC))
#ARCHS=gfx90a:xnack+ gfx90a:xnack- gfx906:xnack-
CFLAGS=#patsubst %,--offload-arch=%,$(ARCHS))
DBG=-g -DCHECK
DBGBIN=$(patsubst %,%-dbg,$(BIN))

all: $(BIN)

%: %.cpp
	$(CC) $(CFLAGS) -o $@ $<

check: $(DBGBIN)

%-dbg: %.cpp
	$(CC) $(CFLAGS) $(DBG) -o $@ $<	

.PHONY=clean
clean:
	rm -rf $(BIN) $(DBGBIN)

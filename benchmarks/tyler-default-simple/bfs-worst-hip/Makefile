CC=hipcc
SRC=$(wildcard *.cpp)
BIN=$(patsubst %.cpp,%,$(SRC))
CFLAGS=-g
DBG=-DCHECK
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

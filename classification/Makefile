main: build/main.o
	clang++ -std=c++11 -o main build/main.o

build/main.o: src/main.cpp
	clang++ -std=c++11 -c src/main.cpp -o build/main.o

clean:
	rm -f build/* main

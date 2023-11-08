all: lab.out

lab.out: lab.cu input.txt
	nvcc -o lab.out lab.cu

run: lab.out
	./lab.out < input.txt

clean:	
	rm -rf lab.out

crun: lab.out
	make
	make run


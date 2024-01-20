# compile and run

nvcc hw1-color2gray.cu -lpng -o your_program

./your_program

# benchmark(time)

run 1000 calculations:

gpu: 30ms

cpu(16 threads): 3750ms

cpu(1 thread): 10000ms


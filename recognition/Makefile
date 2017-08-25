CC = g++
CPPFLAGS = -lopencv_core -lopencv_features2d -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_ml -lopencv_flann -Wall
headers = utils.hpp

all: train score

%.o: %.cpp $(headers)
	$(CC) -c $< -o $@ -Wall

train: train.o utils.o
	$(CC) $^ -o $@ $(CPPFLAGS)

score: score.o utils.o
	$(CC) $^ -o $@ $(CPPFLAGS)

mlp.yml vocabulary.yml: train
	./train ../pictures/train_good ../pictures/train_bad

run: score mlp.yml vocabulary.yml
	./score ../pictures/drive/positive-public
	./score ../pictures/drive/negative-public
	./score ../pictures/utilitar-private-open-dataset/ppeace-private
	./score ../pictures/utilitar-private-open-dataset/nwar-private

clean:
	rm -rf train score *.o

cleanall: clean
	rm -rf mlp.yml vocabulary.yml
	find ../pictures -type f -name "*.yml" -delete
	find ../pictures -type f -name "*.yaml" -delete

.PHONY: all run clean cleanall

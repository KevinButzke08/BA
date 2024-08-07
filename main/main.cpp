#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <assert.h>
#include "esp_log.h"
#include "esp_spiffs.h"
#include "model.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

namespace FAST_INFERENCE {}
using namespace FAST_INFERENCE;
TaskHandle_t benchmarkTaskHandle1 = NULL;
TaskHandle_t benchmarkTaskHandle2 = NULL;
TaskHandle_t mainTaskHandle = NULL;
struct TaskParams {
    unsigned int repeat;
	unsigned int batchSize;
	unsigned int lineNumbers;
};
void init_spiffs() {
    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/storage",
        .partition_label = NULL,
        .max_files = 5,
        .format_if_mount_failed = true
    };

    esp_vfs_spiffs_register(&conf);
}

auto read_csv(std::string &path, unsigned int batchingSize, unsigned lineNumber) {
	std::vector<std::vector<double>> X;
	std::vector<unsigned int> Y;

	std::ifstream file(path);
	std::string header;
	std::getline(file, header);

	//HARDCODED TO SAVE EVERYTIME ITERATING AND EVERYTIME THE SAME TESTING.CSV
	unsigned int label_pos = 784;
	unsigned currentLine = 1;
	std::stringstream ss(header);
	std::string entry;
	
	if (file.is_open()) {
		std::string line;
		
		while (std::getline(file, line)) {
			if(currentLine != lineNumber) {
				currentLine++;
			}
			else {
				break;
			 }
		}
		for(int batchIndex = 1; batchIndex < batchingSize; batchIndex++) {
			std::getline(file, line);
				if (line.size() > 0) {
					std::stringstream ss(line);
					entry = "";

					unsigned int i = 0;
					std::vector<double> x;
					while (std::getline(ss, entry, ',')) {
						if (i == label_pos) {
						Y.push_back(static_cast<unsigned int>(std::stoi(entry)));
						} else {
						x.push_back(static_cast<double>(std::stof(entry)));
						}
					++i;
					}
				X.push_back(x);
				}
		}
		file.close();
	}
		return std::make_tuple(X,Y);
}
void benchmark(void *params) {
    //double output[N_CLASSES] = {0};
	std::string path = std::string("/storage/testing.csv");
	TaskParams *taskParams = static_cast<TaskParams*>(params);
	unsigned int batchSize = taskParams->batchSize;
	unsigned int repeat = taskParams->repeat;
	unsigned int lineNumbers = taskParams->lineNumbers;
	unsigned int summedUpMatches = 0;
	unsigned int xSize = 0;
	float accuracy;
	for(int testDataLine = 1; testDataLine < lineNumbers; testDataLine += batchSize) {
		auto data = read_csv(path, batchSize, testDataLine);
		std::vector<std::vector<double>> &X = std::get<0>(data);
    	std::vector<unsigned int> &Y = std::get<1>(data);
		double * output = new double[N_CLASSES];
		unsigned int matches = 0;
		xSize = X.size();
    	for (unsigned int k = 0; k < repeat; ++k) {	
    		matches = 0;
	    	for (unsigned int i = 0; i < X.size(); ++i) {
				//std::cout<< i << std::endl;
	        	std::fill(output, output+N_CLASSES, 0);
	        	unsigned int label = Y[i];
				double const * const x = &X[i][0];
				predict_SmallCnnActionBINARY5(x, output);
				if constexpr (N_CLASSES >= 2) {
					double max = output[0];
					unsigned int argmax = 0;
					for (unsigned int j = 1; j < N_CLASSES; j++) {
						if (output[j] > max) {
							max = output[j];
							argmax = j;
						}
					}

					if (argmax == label) {
						++matches;
					}
				} else {
					if ( (output[0] < 0 && label == 0) || (output[0] >= 0 && label == 1) ) {
						++matches;
					}
				} 
			}
    	}
		std::cout << matches << std::endl;
    delete[] output;
	//TODO:SUMMEDUPMATCHES ONLY 3, SHOULD BE 5!
	summedUpMatches = summedUpMatches + matches;
	}
	xSize++;
	xSize = xSize*(lineNumbers/batchSize) - 1;
	std::cout << summedUpMatches << std::endl;
	std::cout << xSize << "X SIZE" << std::endl;
	std::cout << static_cast<float>(summedUpMatches) / xSize;
    accuracy = static_cast<float>(summedUpMatches) / xSize * 100.f;
	#ifdef REF_ACCURACY
		float difference = accuracy - REF_ACCURACY;
		std::cout << "Reference Accuracy: " << REF_ACCURACY << " %" << std::endl;
		std::cout << "Difference: " << difference << std::endl;
	    
        std::cout << accuracy << "," << REF_ACCURACY << "," << difference << std::endl;
	#else
        std::cout << accuracy << "," << "," << "," << results.second << std::endl;
    #endif
	xTaskNotifyGive(mainTaskHandle);
	vTaskDelete(NULL);
}
extern "C" void app_main(void){
    init_spiffs();
    
    std::string path = std::string("/storage/testing.csv");
    unsigned int repeat = 2;
	unsigned int batchSize = 2;
	unsigned int lineNumbers = 10;
	
    std::cout << "RUNNING BENCHMARK WITH " << repeat << " REPETITIONS" << std::endl;
    //auto results = benchmark(std::get<0>(data), std::get<1>(data), repeat/2);
	TaskParams params1{repeat, batchSize, lineNumbers};
	mainTaskHandle = xTaskGetCurrentTaskHandle();
	auto start = std::chrono::high_resolution_clock::now();
	xTaskCreate(benchmark, "Task1", 20000, &params1, 1, &benchmarkTaskHandle1);
	//xTaskCreatePinnedToCore(benchmark, "Task2", 70000, &params1, 1, &benchmarkTaskHandle2, 1);
	ulTaskNotifyTake(pdTRUE,portMAX_DELAY);
	auto end = std::chrono::high_resolution_clock::now();

    //auto runtime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()) / (std::get<0>(data).size() * repeat);
	std::cout << "TOTAL RUNTIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << std::endl;
    //std::cout << "Latency: " << runtime << " [ms/elem]" << std::endl;
	
}


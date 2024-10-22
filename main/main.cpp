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

void read_csv(std::string &path, unsigned int batchingSize, unsigned lineNumber, unsigned maxLine,std::vector<std::vector<double>> &X, std::vector<unsigned int> &Y ) {
	X.clear();
    Y.clear();
	std::ifstream file(path);
	std::string header;
	std::getline(file, header);

	//HARDCODED TO SAVE EVERYTIME ITERATING AND EVERYTIME THE SAME TESTING.CSV
	unsigned int label_pos = 784;
	unsigned currentLine = 2;
	std::stringstream ss(header);
	std::string entry;
	if (file.is_open()) {
		std::string line;
		while (currentLine <= lineNumber) {
			if(currentLine != lineNumber) {
				std::getline(file, line);
				currentLine++;
			}
			else {
				break;
			 }
		}
		for(int batchIndex = 1; batchIndex <= batchingSize; batchIndex++) {
			std::getline(file, line);
				if (line.size() > 0) {
					if(currentLine > maxLine) {
						break;
					}
					else {
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
					currentLine++;
					}
				}
		}
		file.close();
	}
}
void benchmark(void *params) {
	std::string path = std::string("/storage/testing.csv");
	TaskParams *taskParams = static_cast<TaskParams*>(params);
	unsigned int batchSize = taskParams->batchSize;
	unsigned int repeat = taskParams->repeat;
	unsigned int lineNumbers = taskParams->lineNumbers;
	unsigned int summedUpMatches = 0;
	unsigned int xSize = lineNumbers - 1;
	float accuracy;
	double output[N_CLASSES];
	std::vector<std::vector<double>> X;
    std::vector<unsigned int> Y;

	for(int testDataLine = 2; testDataLine <= lineNumbers; testDataLine += batchSize) {
		read_csv(path, batchSize, testDataLine, lineNumbers, X, Y);
		unsigned int matches = 0;
    	for (unsigned int k = 0; k < repeat; ++k) {	
    		matches = 0;
	    	for (unsigned int i = 0; i < X.size(); ++i) {
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
	summedUpMatches = summedUpMatches + matches;
	X.clear();
	X.shrink_to_fit();
	Y.clear();
	Y.shrink_to_fit();
	}
	std::cout << summedUpMatches << " MATCHES" << std::endl;
	std::cout << xSize << " X SIZE" << std::endl;
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
    unsigned int repeat = 8;
	unsigned int batchSize = 5;
	unsigned int lineNumbers = 100;
	
    std::cout << "RUNNING BENCHMARK WITH " << repeat << " REPETITIONS" << std::endl;
	TaskParams params1{repeat, batchSize, lineNumbers};
	mainTaskHandle = xTaskGetCurrentTaskHandle();
	auto start = std::chrono::high_resolution_clock::now();
	xTaskCreate(benchmark, "Task1", 5000, &params1, 1, &benchmarkTaskHandle1);
	//xTaskCreatePinnedToCore(benchmark, "Task2", 70000, &params1, 1, &benchmarkTaskHandle2, 1);
	ulTaskNotifyTake(pdTRUE,portMAX_DELAY);
	auto end = std::chrono::high_resolution_clock::now();
    auto runtime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()) / ((lineNumbers-1) * repeat);
	std::cout << "TOTAL RUNTIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << " ms" << std::endl;
    std::cout << "Latency: " << runtime << " [ms/elem]" << std::endl;
}


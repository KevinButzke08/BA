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
TaskHandle_t benchmarkTaskHandle1, benchmarkTaskHandle2;
TaskHandle_t mainTaskHandle = NULL;
unsigned matchesCore0 = 0;
unsigned matchesCore1 = 0;
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
	using namespace FAST_INFERENCE;
	std::string path = std::string("/storage/testing.csv");
	TaskParams *taskParams = static_cast<TaskParams*>(params);
	unsigned int batchSize = taskParams->batchSize;
	unsigned int repeat = taskParams->repeat;
	unsigned int lineNumbers = taskParams->lineNumbers;
	double output[N_CLASSES];
	std::vector<std::vector<double>> X;
    std::vector<unsigned int> Y;
	for(int testDataLine = (lineNumbers/2 + 2); testDataLine <= (lineNumbers) ; testDataLine += batchSize) {
		read_csv(path, batchSize, testDataLine, lineNumbers, X, Y);
		unsigned int matches = 0;
    	for (unsigned int k = 0; k < repeat; ++k) {	
    		matches = 0;
	    	for (unsigned int i = 0; i < X.size(); ++i) {
	        	std::fill(output, output+N_CLASSES, 0);
	        	unsigned int label = Y[i];
				double const * const x = &X[i][0];
				predict_SimpleMLP152(x, output);
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
	matchesCore0 = matchesCore0 + matches;
	X.clear();
	X.shrink_to_fit();
	Y.clear();
	Y.shrink_to_fit();
	std::cout << "BATCH DONE" << std::endl;
	}
	std::cout << "CORE 0 finished" << std::endl;
	xTaskNotifyGive(mainTaskHandle);
	vTaskDelete(NULL);
}
void benchmark2core(void *params) {
	using namespace FAST_VARIANT;
	std::string path = std::string("/storage/testing.csv");
	TaskParams *taskParams = static_cast<TaskParams*>(params);
	unsigned int batchSize = taskParams->batchSize;
	unsigned int repeat = taskParams->repeat;
	unsigned int lineNumbers = taskParams->lineNumbers;
	double output[N_CLASSES];
	std::vector<std::vector<double>> X;
    std::vector<unsigned int> Y;
	for(int testDataLine = 2; testDataLine <= lineNumbers/2; testDataLine += batchSize) {
		read_csv(path, batchSize, testDataLine, lineNumbers, X, Y);
		unsigned int matches = 0;
    	for (unsigned int k = 0; k < repeat; ++k) {	
    		matches = 0;
			//std::cout << "CORE 1 EXECUTING" << std::endl;
	    	for (unsigned int i = 0; i < X.size(); ++i) {
	        	std::fill(output, output+N_CLASSES, 0);
	        	unsigned int label = Y[i];
				double const * const x = &X[i][0];
				predict_SimpleMLP15(x, output);
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
	matchesCore1 = matchesCore1 + matches;
	X.clear();
	X.shrink_to_fit();
	Y.clear();
	Y.shrink_to_fit();
	std::cout << " CORE 1 BATCH DONE" << std::endl;
	}
	std::cout << "CORE 1 finished" << std::endl;
	xTaskNotifyGive(mainTaskHandle);
	vTaskDelete(NULL);
}
extern "C" void app_main(void){
    init_spiffs();
	size_t totalHeapAtStart = xPortGetFreeHeapSize();
	std::cout << totalHeapAtStart << " BYTES FOR HEAP" << std::endl;
    std::string path = std::string("/storage/testing.csv");
    unsigned int repeat = 8;
	unsigned int batchSize = 5;
	unsigned int lineNumbers = 10;
	unsigned int combinedMatches = 0;
    std::cout << "RUNNING BENCHMARK WITH " << repeat << " REPETITIONS" << std::endl;
	TaskParams params1{repeat, batchSize, lineNumbers};
	mainTaskHandle = xTaskGetCurrentTaskHandle();
	auto start = std::chrono::high_resolution_clock::now();
	xTaskCreatePinnedToCore(benchmark, "BenchmarkTask1", 6000, &params1, 1, &benchmarkTaskHandle1, 0);
	xTaskCreatePinnedToCore(benchmark2core, "BenchmarkTask2", 6000, &params1, 1, &benchmarkTaskHandle2, 1);
	ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
	ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
	combinedMatches = matchesCore0 + matchesCore1;
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << combinedMatches << " MATCHES" << std::endl;
	std::cout << lineNumbers - 1 << " X SIZE" << std::endl;
	auto accuracy = static_cast<float>(combinedMatches) / (lineNumbers - 1) * 100.f;

	{
		using namespace FAST_INFERENCE;
	
	//#ifdef REF_ACCURACY
		float difference = accuracy - REF_ACCURACY;
		std::cout << "Reference Accuracy: " << REF_ACCURACY << " %" << std::endl;
		std::cout << "Difference: " << difference << std::endl;
		std::cout << accuracy << "," << REF_ACCURACY << "," << difference << std::endl;
	//#else
        //std::cout << accuracy << "," << "," << "," << results.second << std::endl;
    //#endif
    auto runtime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()) / ((lineNumbers-1) * repeat);
	std::cout << "TOTAL RUNTIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << " ms" << std::endl;
    std::cout << "Latency: " << runtime << " [ms/elem]" << std::endl;
	size_t min_free_heap_size = xPortGetMinimumEverFreeHeapSize();
	std::cout << "MAXIMUM HEAP USAGE: " << min_free_heap_size << " FREE BYTES" << std::endl;
	}

}


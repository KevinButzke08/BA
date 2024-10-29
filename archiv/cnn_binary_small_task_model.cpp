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
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include <cmath>

namespace FAST_INFERENCE {

static constexpr double layer_2_weight[3][3][1][8] = {{{{-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0}}, {{-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0}}, {{-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0}}}, {{{-1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0}}, {{-1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0}}, {{-1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0}}}, {{{1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0}}, {{1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0}}, {{1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0}}}};
static constexpr double layer_2_bias[8] = {-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0};
static constexpr double layer_3_bias[8] = {0.7844486236572266, 0.9431903958320618, 0.7916160225868225, -1.0173650979995728, -0.9932975769042969, -0.9948334097862244, -0.46668633818626404, 0.8201964497566223};
static constexpr double layer_3_scale[8] = {0.0032525083515793085, 0.0038738118018954992, 0.0036308574490249157, 0.004138038959354162, 0.002381887286901474, 0.004025470931082964, 0.00552032794803381, 0.003692388068884611};
static constexpr double layer_6_weight[3][3][8][8] = {{{{1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0}, {1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0}, {-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, {-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0}, {-1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0}, {-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0}, {-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0}}, {{-1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0}, {-1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0}, {1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0}, {-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0}, {-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0}, {1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0}, {-1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0}, {-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}}, {{-1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0}, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0}, {-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0}, {1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0}, {-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, {1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0}, {-1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0}, {1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0}}}, {{{1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0}, {-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0}, {-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0}, {-1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0}, {-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0}, {-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0}, {-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0}, {1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0}}, {{1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0}, {1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0}, {1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0}, {-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0}, {1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0}, {-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0}, {1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0}}, {{1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0}, {1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0}, {-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0}, {-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0}, {1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0}, {-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0}, {-1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0}, {1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0}}}, {{{1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0}, {1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0}, {1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0}, {-1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0}, {-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0}, {-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0}, {1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0}}, {{-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0}, {-1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0}, {1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0}, {1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0}, {1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0}, {-1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0}, {-1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0}}, {{-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0}, {-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0}, {1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0}, {-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0}, {1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0}, {-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0}, {-1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0}}}};
static constexpr double layer_6_bias[8] = {-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0};
static constexpr double layer_7_bias[8] = {0.10010764002799988, 0.017013445496559143, 0.2510557174682617, -0.5073535442352295, -0.5701065063476562, -0.12974032759666443, 0.08583422750234604, -0.15982460975646973};
static constexpr double layer_7_scale[8] = {0.06263870000839233, 0.09152252972126007, 0.060542747378349304, 0.08511810004711151, 0.11554820090532303, 0.10308130085468292, 0.0899776741862297, 0.09787606447935104};
static constexpr double layer_11_weight[8][200] = {{-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0}, {-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0}, {-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0}, {1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0}, {-1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0}};
static constexpr double layer_11_bias[8] = {1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0};
static constexpr double layer_12_bias[8] = {-0.1592363566160202, 0.16492287814617157, 0.1262357383966446, -0.011199899017810822, 0.24381498992443085, 0.0817921981215477, 0.3568369746208191, 0.12275934964418411};
static constexpr double layer_12_scale[8] = {0.01958995871245861, 0.022022325545549393, 0.023702004924416542, 0.033146556466817856, 0.026454458013176918, 0.041885536164045334, 0.0283559150993824, 0.02722897380590439};
static constexpr double layer_14_weight[10][8] = {{1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0}, {1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0}, {1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0}, {1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0}, {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0}, {-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0}, {1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0}, {1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0}, {-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0}};
static constexpr double layer_14_bias[10] = {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0};

static double layer_2_output[26][26][8];
static double layer_3_output[26][26][8];
static double layer_4_output[26][26][8];
static double layer_5_output[13][13][8];
static double layer_6_output[11][11][8];
static double layer_7_output[11][11][8];
static double layer_8_output[11][11][8];
static double layer_9_output[5][5][8];
static double layer_11_output[8];
static double layer_12_output[8];
static double layer_13_output[8];
static double layer_14_output[10];
static double layer_15_output[10];

struct MaxPoolTaskParams
  {
    unsigned int hp;
    unsigned int wp;
    unsigned int cp;
    unsigned int kHp;
    unsigned int kWp;
    unsigned int prevWp;
    double *output;
    double *output_1;
    TaskHandle_t* taskHandels;
  };
  struct GemmTaskParams
  {
    unsigned int num_rows;
    unsigned int num_columns;
    double* output;
    double* output_1;
    const double* bias;
    const double* weight;
    TaskHandle_t* taskHandels;
  };
  struct ConvTaskParams
  {
    unsigned int hp;
    unsigned int wp;
    unsigned int mp;
    unsigned int kHp;
    unsigned int kWp;
    unsigned int cp;
    unsigned int prev_wp;
    double* output;
    double* output_1;
    const double* bias;
    const double* weight;
    TaskHandle_t* taskHandels;
  };
  struct BatchNormalization1DTaskParams
  {
    unsigned int num_rows;
    double* output;
    double* output_1;
    const double* scale;
    const double* bias;
    TaskHandle_t* taskHandels;
  };
  struct BatchNormalization3DTaskParams
  {
    unsigned int num_rows;
    unsigned int wp;
    unsigned int cp;
    double* output;
    double* output_1;
    const double* scale;
    const double* bias;
    TaskHandle_t* taskHandels;
  };
  struct Step1DTaskParams
  {
    unsigned int hp;
    double* output;
    double* output_1;
    TaskHandle_t* taskHandels;
  };
  struct Step3DTaskParams
  {
    unsigned int hp;
    unsigned int wp;
    unsigned int cp;
    double* output;
    double* output_1;
    TaskHandle_t* taskHandels;
  };
  struct LogSoftmaxTaskParams
  {
    unsigned int num_rows;
    double* output;
    double* output_1;
    double* pred;
    SemaphoreHandle_t mutex;
  };
  
  void ConvTask(void *params) {
    ConvTaskParams *taskParams = static_cast<ConvTaskParams *>(params);
    TaskHandle_t currentTaskHandle = xTaskGetCurrentTaskHandle();
    TaskHandle_t* layerHandles = taskParams->taskHandels;

    if(currentTaskHandle == layerHandles[4]) {
      ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    }
    else {
      vTaskDelay(1);
    }
    unsigned int hp = taskParams->hp;
    unsigned int wp = taskParams->wp;
    unsigned int mp = taskParams->mp;
    unsigned int kHp = taskParams->kHp;
    unsigned int kWp = taskParams->kWp;
    unsigned int cp = taskParams->cp;
    unsigned int prev_wp = taskParams->prev_wp;
    double *layer_p_output = taskParams->output;
    double *layer_p_previous_output = taskParams->output_1;
    const double *layer_p_bias = taskParams->bias;
    const double *layer_p_weight = taskParams->weight;
    unsigned int layer_p_index;
    
    for (int h = 0; h < hp; h++) {
      for (int w = 0; w < wp; w++) {
        for (int m = 0; m < mp; m++) {
          layer_p_index = (h * wp * mp) + (w * mp) + m;
          layer_p_output[layer_p_index] = layer_p_bias[m];
        }
        for (int kH = 0; kH < kHp; kH++) {
          for (int kW = 0; kW < kWp; kW++) {
            for (int c = 0; c < cp; c++) {
              for (int m = 0; m < mp; m++) {
                int weightIndex = (kH * kWp * cp * mp) + (kW * cp * mp) + (c * mp) + m;
                layer_p_index = (h * wp * mp) + (w * mp) + m;
                int layer_p_previous_index = ((h * 1 + kH - 0) * (prev_wp * cp)) + ((w * 1 + kW - 0) * cp) + c;
                layer_p_output[layer_p_index] += layer_p_weight[weightIndex] * layer_p_previous_output[layer_p_previous_index];
              }
            }
          }
        }
      }
    }
    if(currentTaskHandle == layerHandles[0]) {
      //printf("layer2 finished");
      xTaskNotifyGive(layerHandles[1]);
    }
    else {
      //printf("layer6 finished");
      xTaskNotifyGive(layerHandles[5]);
    }
    vTaskDelete(NULL);
  }  
  void BatchNormalization1DTask(void *params) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    BatchNormalization1DTaskParams *taskParams =  static_cast<BatchNormalization1DTaskParams *>(params);
    unsigned int num_rows = taskParams->num_rows;
    double *layer_p_output = taskParams->output;
    double *layer_p_previous_output = taskParams->output_1;
    const double *layer_p_scale = taskParams->scale;
    const double *layer_p_bias = taskParams->bias;
    TaskHandle_t* layerHandles = taskParams->taskHandels;

    for (int d = 0; d < num_rows; d++) {
      layer_p_output[d] = layer_p_previous_output[d] * layer_p_scale[d] + layer_p_bias[d];
    }
    //printf("layer12 finished");
    xTaskNotifyGive(layerHandles[10]);
    vTaskDelete(NULL);
  }

  void BatchNormalization3DTask(void *params) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    BatchNormalization3DTaskParams *taskParams = static_cast<BatchNormalization3DTaskParams *>(params);
    unsigned int num_rows = taskParams->num_rows;
    unsigned int wp = taskParams->wp;
    unsigned int cp = taskParams->cp;
    double *layer_p_output = taskParams->output;
    double *layer_p_previous_output = taskParams->output_1;
    const double *layer_p_scale = taskParams->scale;
    const double *layer_p_bias = taskParams->bias;
    TaskHandle_t* layerHandles = taskParams->taskHandels;

   for (int d = 0; d < num_rows; d++) {
      for (int w = 0; w < wp; w++) {
        for (int c = 0; c < cp; c++) {
          unsigned int layer_index = (d * wp * cp) + (w * cp) + c;
          layer_p_output[layer_index] = layer_p_previous_output[layer_index] * layer_p_scale[c] + layer_p_bias[c];
        }
      }
    }
    TaskHandle_t currentTaskHandle = xTaskGetCurrentTaskHandle();
    if(currentTaskHandle == layerHandles[1]) {
      //printf("layer3 finished");
      xTaskNotifyGive(layerHandles[2]);
    }
    else {
      //printf("layer7 finished");
      xTaskNotifyGive(layerHandles[6]);
    }
    vTaskDelete(NULL);
  }
  void Step1DTask(void *params) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    Step1DTaskParams *taskParams =  static_cast<Step1DTaskParams *>(params);
    unsigned int hp = taskParams->hp;
    double *layer_p_output = taskParams->output;
    double *layer_p_previous_output = taskParams->output_1;
    TaskHandle_t* layerHandles = taskParams->taskHandels;
  
    for (int h = 0; h < hp; h++) {
      layer_p_output[h] = layer_p_previous_output[h] > 0.0 ? 1.0 : -1.0;
    }
    //printf("layer13 finished");
    xTaskNotifyGive(layerHandles[11]);
    vTaskDelete(NULL);
  }

  void Step3DTask(void *params) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    Step3DTaskParams *taskParams =  static_cast<Step3DTaskParams *>(params);
    unsigned int hp = taskParams->hp;
    unsigned int wp = taskParams->wp;
    unsigned int cp = taskParams->cp;
    double *layer_p_output = taskParams->output;
    double *layer_p_previous_output = taskParams->output_1;
    TaskHandle_t* layerHandles = taskParams->taskHandels;
    TaskHandle_t currentTaskHandle = xTaskGetCurrentTaskHandle();
    
    for (int h = 0; h < hp; h++) {
      for (int w = 0; w < wp; w++) {
        for (int c = 0; c < cp; c++) {
          unsigned int layer_index = (h * wp * cp) + (w * cp) + c;
          layer_p_output[layer_index] = layer_p_previous_output[layer_index] > 0.0 ? 1.0 : -1.0;
        }
      }
    }
    if(currentTaskHandle == layerHandles[2]) {
      //printf("layer4 finished");
      xTaskNotifyGive(layerHandles[3]);
    }
    else {
      //printf("layer8 finished");
      xTaskNotifyGive(layerHandles[7]);
    }
    vTaskDelete(NULL);    
  }
void GemmTask(void *params) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    TaskHandle_t currentTaskHandle = xTaskGetCurrentTaskHandle();
    GemmTaskParams *taskParams = static_cast<GemmTaskParams *>(params);
    unsigned int num_rows = taskParams->num_rows;
    unsigned int ip = taskParams->num_columns;
    double *layer_p_output = taskParams->output;
    double *layer_p_previous_output = taskParams->output_1;
    const double *layer_p_bias = taskParams->bias;
    const double *layer_p_weight = taskParams->weight;
    TaskHandle_t* layerHandles = taskParams->taskHandels;
    
    if(currentTaskHandle == layerHandles[8]) {
      auto layer_10_output = (double *) layer_p_previous_output;
      for (int d = 0; d < num_rows; d++) {
      layer_p_output[d] = layer_p_bias[d];
      }
      for (int d = 0; d < num_rows; d++) {
        for (int i = 0; i < ip; i++) {
          int weightIndex = d * ip + i;
          layer_p_output[d] += layer_p_weight[weightIndex] * layer_10_output[i];
        }
      }
    }
    else {
      for (int d = 0; d < num_rows; d++) {
        layer_p_output[d] = layer_p_bias[d];
      }
      for (int d = 0; d < num_rows; d++) {
        for (int i = 0; i < ip; i++) {
          int weightIndex = d * ip + i;
          layer_p_output[d] += layer_p_weight[weightIndex] * layer_p_previous_output[i];
        }
      }
    }
    if(currentTaskHandle == layerHandles[8]) {
      //printf("layer11 finished");
      xTaskNotifyGive(layerHandles[9]);
    }
    else {
      //printf("layer14 finished");
      xTaskNotifyGive(layerHandles[12]);
    }
    vTaskDelete(NULL);
  }
  void MaxPoolTask(void *params) { 
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    MaxPoolTaskParams *maxPoolTaskParams = static_cast<MaxPoolTaskParams *>(params);
    unsigned int hp = maxPoolTaskParams->hp;
    unsigned int wp = maxPoolTaskParams->wp;
    unsigned int cp = maxPoolTaskParams->cp;
    unsigned int kHp = maxPoolTaskParams->kHp;
    unsigned int kWp = maxPoolTaskParams->kWp;
    unsigned int prevWp = maxPoolTaskParams->prevWp;
    double* layer_p_output = maxPoolTaskParams->output;
    double* layer_p_previous_output = maxPoolTaskParams->output_1;
    TaskHandle_t* layerHandles = maxPoolTaskParams->taskHandels;
    int p_output_index;
    int p_previous_output_index;
      for (int h = 0; h < hp; h++) {
        for (int w = 0; w < wp; w++) {
          for (int c = 0; c < cp; c++) {
            p_output_index = (h * wp * cp) + (w * cp) + c;
            layer_p_output[p_output_index] = std::numeric_limits<double>::lowest();
          }
          for (int kH = 0; kH < kHp; kH++) {
            for (int kW = 0; kW < kWp; kW++) {
              for (int c = 0; c < cp; c++) {
                p_output_index = (h * wp * cp) + (w * cp) + c;
                p_previous_output_index = (h * 2 + kH) * (prevWp * cp) + (w * 2 + kW) * cp + c;
                layer_p_output[p_output_index] = std::max(layer_p_previous_output[p_previous_output_index], layer_p_output[p_output_index]);
              }
            }
          }
        }
      }
    TaskHandle_t currentTaskHandle = xTaskGetCurrentTaskHandle();
    if(currentTaskHandle == layerHandles[3]) {
      //printf("layer5 finished");
      xTaskNotifyGive(layerHandles[4]);
    }
    else {
      //printf("layer9 finished");
      xTaskNotifyGive(layerHandles[8]);
    }
    vTaskDelete(NULL);
  }  
  void LogSoftmaxTask(void *params) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    LogSoftmaxTaskParams *taskParams = static_cast<LogSoftmaxTaskParams *>(params);
    unsigned int num_rows = taskParams->num_rows;
    double *layer_p_output = taskParams->output;
    double *layer_p_previous_output = taskParams->output_1;
    double *pred = taskParams->pred;
    SemaphoreHandle_t mutex = taskParams->mutex;
    double max = 0;
    for (int d = 0; d < num_rows; d++)
    {
      max = layer_p_previous_output[d] >= max ? layer_p_previous_output[d] : max;
    }
    double sum = 0;
    for (int d = 0; d < num_rows; d++)
    {
      layer_p_output[d] = std::exp(layer_p_previous_output[d] - max);
      sum += layer_p_output[d];
    }
    for (int d = 0; d < num_rows; d++)
    {
      layer_p_output[d] = std::log(layer_p_output[d] / sum);
    }
    for (int i = 0; i < num_rows; i++)
    {
      pred[i] += layer_p_output[i];
    }
    //printf("layer15 finished");
    xSemaphoreGive(mutex);
    vTaskDelete(NULL);
  }

void predict_SmallCnnActionBINARY8(double const * const x, double * pred) {
    
    static SemaphoreHandle_t mutex;
    TaskHandle_t layer2Handle = NULL, layer3Handle = NULL, layer4Handle = NULL, layer5Handle = NULL, layer6Handle = NULL, layer7Handle = NULL, layer8Handle = NULL, layer9Handle = NULL, layer11Handle = NULL, layer12Handle = NULL, layer13Handle = NULL, layer14Handle = NULL, layer15Handle = NULL;
    TaskHandle_t taskHandleArray[13] = {layer2Handle, layer3Handle, layer4Handle, layer5Handle, layer6Handle, layer7Handle, layer8Handle, layer9Handle, layer11Handle, layer12Handle, layer13Handle, layer14Handle, layer15Handle};
    mutex = xSemaphoreCreateBinary();
    auto layer_0_output = x;
    auto layer_1_output = (double (*)[28][1]) layer_0_output;
    ConvTaskParams layer2params{26, 26, 8, 3, 3, 1, 28, &layer_2_output[0][0][0], &layer_1_output[0][0][0], const_cast<double *>(layer_2_bias), &layer_2_weight[0][0][0][0], taskHandleArray};
    BatchNormalization3DTaskParams layer3params{26, 26, 8, &layer_3_output[0][0][0], &layer_2_output[0][0][0], layer_3_scale, layer_3_bias, taskHandleArray};
    Step3DTaskParams layer4params{26, 26, 8, &layer_4_output[0][0][0], &layer_3_output[0][0][0], taskHandleArray};
    MaxPoolTaskParams layer5params{13, 13, 8, 2, 2, 26, &layer_5_output[0][0][0], &layer_4_output[0][0][0], taskHandleArray};
    ConvTaskParams layer6params{11, 11, 8, 3, 3, 8, 13, &layer_6_output[0][0][0], &layer_5_output[0][0][0], layer_6_bias, &layer_6_weight[0][0][0][0], taskHandleArray};
    BatchNormalization3DTaskParams layer7params{11, 11, 8, &layer_7_output[0][0][0], &layer_6_output[0][0][0], layer_7_scale, layer_7_bias, taskHandleArray};
    Step3DTaskParams layer8params{11, 11, 8, &layer_8_output[0][0][0], &layer_7_output[0][0][0], taskHandleArray};
    MaxPoolTaskParams layer9params{5, 5, 8, 2, 2, 11, &layer_9_output[0][0][0], &layer_8_output[0][0][0], taskHandleArray};
    GemmTaskParams layer11params{8, 200, layer_11_output, &layer_9_output[0][0][0], layer_11_bias, &layer_11_weight[0][0], taskHandleArray};
    BatchNormalization1DTaskParams layer12params{8, layer_12_output, layer_11_output, layer_12_scale, layer_12_bias, taskHandleArray};
    Step1DTaskParams layer13params{8, layer_13_output, layer_12_output, taskHandleArray};
    GemmTaskParams layer14params{10, 8, layer_14_output, layer_13_output, layer_14_bias, &layer_14_weight[0][0], taskHandleArray};
    LogSoftmaxTaskParams layer15params{10, layer_15_output, layer_14_output, pred, mutex};

    xTaskCreate(ConvTask, "Layer2", 2048, &layer2params, 1, &taskHandleArray[0]);
    xTaskCreate(BatchNormalization3DTask, "Layer3", 2048, &layer3params, 1, &taskHandleArray[1]);
    xTaskCreate(Step3DTask, "Layer4", 2048, &layer4params, 1, &taskHandleArray[2]);
    xTaskCreate(MaxPoolTask, "Layer5", 2048, &layer5params, 1, &taskHandleArray[3]);
    xTaskCreate(ConvTask, "Layer6", 2048, &layer6params, 1, &taskHandleArray[4]);
    xTaskCreate(BatchNormalization3DTask, "Layer7", 2048, &layer7params, 1, &taskHandleArray[5]);
    xTaskCreate(Step3DTask, "Layer8", 2048, &layer8params, 1, &taskHandleArray[6]);
    xTaskCreate(MaxPoolTask, "Layer9", 2048, &layer9params, 1, &taskHandleArray[7]);
    xTaskCreate(GemmTask, "Layer11", 2048, &layer11params, 1, &taskHandleArray[8]);
    xTaskCreate(BatchNormalization1DTask, "Layer12", 2048, &layer12params, 1, &taskHandleArray[9]);
    xTaskCreate(Step1DTask, "Layer13", 2048, &layer13params, 1, &taskHandleArray[10]);
    xTaskCreate(GemmTask, "Layer14", 2048, &layer14params, 1, &taskHandleArray[11]);
    xTaskCreate(LogSoftmaxTask, "Layer15", 2048, &layer15params, 1, &taskHandleArray[12]);
    xSemaphoreTake(mutex, portMAX_DELAY);
    vSemaphoreDelete(mutex);
  }

}
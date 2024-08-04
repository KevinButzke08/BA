#pragma once 

namespace FAST_INFERENCE {    
    #define REF_ACCURACY 80.80000000000001

    #define N_CLASSES 15
    #define N_FEATURES 784

    void predict_SimpleMLP15(double const * const x, double * pred);
}
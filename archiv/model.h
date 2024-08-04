#pragma once 

namespace FAST_INFERENCE {    
    #define REF_ACCURACY 87.9

    #define N_CLASSES 10
    #define N_FEATURES 784

    void predict_CnnAction(double const * const x, double * pred);
}
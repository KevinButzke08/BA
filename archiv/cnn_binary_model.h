#pragma once 

namespace FAST_INFERENCE {    
    #define REF_ACCURACY 70.84

    #define N_CLASSES 10
    #define N_FEATURES 784

    void predict_SmallCnnActionBINARY5(double const * const x, double * pred);
}
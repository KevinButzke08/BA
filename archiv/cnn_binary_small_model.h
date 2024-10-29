#pragma once 

namespace FAST_INFERENCE {    
    #define REF_ACCURACY 64.4

    #define N_CLASSES 10
    #define N_FEATURES 784

    void predict_SmallCnnActionBINARY8(double const * const x, double * pred);
}
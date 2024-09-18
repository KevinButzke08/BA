#pragma once 

namespace FAST_INFERENCE {    
    constexpr double REF_ACCURACY = 80.8000000000001;

    constexpr int N_CLASSES = 15;
    #define N_FEATURES 784

    void predict_SimpleMLP152(double const * const x, double * pred);
    //void predict_SimpleMLP15(double const *const x, double *pred);

}
namespace FAST_VARIANT {
    constexpr double REF_ACCURACY =  80.80000000000001;

    constexpr int N_CLASSES =  15;
    #define N_FEATURES 784
    void predict_SimpleMLP15(double const *const x, double *pred);
}
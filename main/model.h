#pragma once 

namespace FAST_INFERENCE {    
    //#define REF_ACCURACY 80.80000000000001

    //#define N_CLASSES 15
    #define N_FEATURES 784
    #define REF_ACCURACY 64.4

    #define N_CLASSES 10
    //#define N_FEATURES 784

    void predict_SmallCnnActionBINARY8(double const * const x, double * pred);
    //void predict_SimpleMLP15(double const * const x, double *pred);
    //void predict_SimpleMLP152(double const * const x, double * pred);
    //void predict_SimpleMLP151(double const *const x, double *pred);

}
namespace FAST_VARIANT {
    constexpr double REF_ACCURACY_Variant = 80.80000000000001;

    constexpr int N_CLASSES_Variant =  15;
    //#define N_FEATURES 784
    void predict_SimpleMLP152(double const *const x, double *pred);
}
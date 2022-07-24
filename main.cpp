#include "trt_tensor.hpp"

#include <cudnn.h>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cassert>


#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }


int test_conv_relu() {
    
    ::srand(::time(0));
    std::cout << "CUDNN_VERSION:" << CUDNN_VERSION << std::endl;
    
    // 设定输入输出tensor的维度参数
    constexpr int batch_size = 4;
    constexpr int channel_in = 3;
    constexpr int height_in = 112;
    constexpr int width_in = 112;

    constexpr int channel_out = 15;
    constexpr int height_out = 112;
    constexpr int width_out = 112;

    constexpr int kernel_h = 1;
    constexpr int kernel_w = 1;

    // 构造相关Tensor
    // input
    TRT::Tensor q_tensor(std::vector<int>{batch_size, channel_in, height_in, width_in});
    // kernel input
    TRT::Tensor kernel_tensor(std::vector<int>{channel_out, channel_in, kernel_h, kernel_w});
    // bias
    TRT::Tensor bias_tensor(std::vector<int>{channel_out});
    TRT::Tensor z_tensor(std::vector<int>{batch_size, channel_out, height_out, width_out});
    // output
    TRT::Tensor out_tensor(std::vector<int>{batch_size, channel_out, height_out, width_out});

    auto qptr_cpu = q_tensor.cpu<float>();

    for(int i = 0; i < q_tensor.numel(); ++i) 
    {
        qptr_cpu[i] = float(rand() % 100000) / 100000;
    }
    q_tensor.save_to_file("q_tensor.npz");

    auto biasptr_cpu = bias_tensor.cpu<float>();
    for(int i = 0; i < bias_tensor.numel(); ++i) 
    {
        biasptr_cpu[i] = float(rand() % 100000) / 100000;
    }
    bias_tensor.save_to_file("bias_tensor.npz");

    auto kernelptr_cpu = kernel_tensor.cpu<float>();
    for(int i = 0; i < kernel_tensor.numel(); ++i) 
    {
        kernelptr_cpu[i] = float(rand() % 100000) / 100000;
    } 
    kernel_tensor.save_to_file("kernel_tensor.npz");


    auto qptr_gpu = q_tensor.to_gpu(true).gpu<float>();
    auto bias_gpu = bias_tensor.to_gpu(true).gpu<float>();
    auto kernel_gpu = kernel_tensor.to_gpu(true).gpu<float>();
    auto outptr_gpu = out_tensor.to_gpu().gpu<float>();


    cudaStream_t stream = out_tensor.get_stream();
    
    // 创建cudnn句柄并设置handle的stream
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));
    checkCUDNN(cudnnSetStream(cudnn, stream));
    // y = act ( alpha1 * conv(x) + alpha2 * z + bias )
    const float alpha1 = 1;
    const float alpha2 = 0;

    // 设置输入Tensor描述符
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                          /*format=*/CUDNN_TENSOR_NCHW,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/batch_size,
                                          /*channels=*/channel_in,
                                          /*image_height=*/height_in,
                                          /*image_width=*/width_in));
    // 设置输出Tensor描述符
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/batch_size,
                                      /*channels=*/channel_out,
                                      /*image_height=*/height_out,
                                      /*image_width=*/width_out));

    // 设置bias描述符
    cudnnTensorDescriptor_t bias_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&bias_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(bias_descriptor,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/1,
                                      /*channels=*/channel_out,
                                      /*image_height=*/1,
                                      /*image_width=*/1));

    // 设置z描述符
    // // y = act ( alpha1 * conv(x) + alpha2 * z + bias ) 这里用不到 
    cudnnTensorDescriptor_t z_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&z_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(z_descriptor,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/batch_size,
                                      /*channels=*/channel_out,
                                      /*image_height=*/height_out,
                                      /*image_width=*/width_out));

    // 设置conv weight的描述
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*format=*/CUDNN_TENSOR_NCHW,
                                          /*out_channels=*/channel_out,
                                          /*in_channels=*/channel_in,
                                          /*kernel_height=*/kernel_h,
                                          /*kernel_width=*/kernel_w));

    // 设置卷积相关参数
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                              /*pad_height=*/0,
                                              /*pad_width=*/0,
                                              /*vertical_stride=*/1,
                                              /*horizontal_stride=*/1,
                                              /*dilation_height=*/1,
                                              /*dilation_width=*/1,
                                              /*mode=*/CUDNN_CROSS_CORRELATION,
                                              /*computeType=*/CUDNN_DATA_FLOAT));

    // 设置激活层相关参数    
    cudnnActivationDescriptor_t activation_descriptor;
    checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
    checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
                                            /*mode=*/CUDNN_ACTIVATION_RELU,
                                            /*reluNanOpt=*/CUDNN_PROPAGATE_NAN,
                                            /*relu_coef=*/0));
    
    // 获取卷积计算算法相关参数和workspace
    int cnt = 0;
    cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn, &cnt);
    std::cout << "cnt: " << cnt << std::endl;
    cudnnConvolutionFwdAlgoPerf_t convolution_algorithm;
    int ret_cnt = 0;
    
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
                                            input_descriptor,
                                            kernel_descriptor,
                                            convolution_descriptor,
                                            output_descriptor,
                                            1,
                                            &ret_cnt,
                                            &convolution_algorithm)); 

    size_t workspace_bytes = 0;       
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                      input_descriptor,
                                                      kernel_descriptor,
                                                      convolution_descriptor,
                                                      output_descriptor,
                                                      convolution_algorithm.algo,
                                                      &workspace_bytes));

    void* d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);

    // 执行卷积运算
    checkCUDNN(cudnnConvolutionBiasActivationForward(
        cudnn, &alpha1, input_descriptor, qptr_gpu, kernel_descriptor, kernel_gpu,
        convolution_descriptor, convolution_algorithm.algo, d_workspace, workspace_bytes,
        &alpha2, z_descriptor, outptr_gpu,
        bias_descriptor, bias_gpu, activation_descriptor, output_descriptor, outptr_gpu));

    out_tensor.to_cpu(true);
    out_tensor.save_to_file("out_tensor.npz");
    
    // 销毁描述符和句柄
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(z_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyTensorDescriptor(bias_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn);
    cudaFree(d_workspace);

    return 0;
}

int main() 
{

    test_conv_relu();

    return 0;
}
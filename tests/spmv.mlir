// RUN: tutorial-opt %s --sparsifier="enable-runtime-library=true" | mlir-runner -e main -entry-point-result=void -shared-libs=../../+_repo_rules+llvm-project/mlir/libmlir_c_runner_utils.so,../../+_repo_rules+llvm-project/mlir/libmlir_runner_utils.so | FileCheck %s

#CSR = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : dense, d1 : compressed)}>
#SparseVector = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed)}>

module {

    // Sparse kernel
    func.func @spmv(%a: tensor<3x3xf32, #CSR>,
                    %b: tensor<1024xf32, #SparseVector>,
                    %c: tensor<1024xf32, #SparseVector>) -> tensor<1024xf32 {
        %result = linalg.dot ins(%a, %b : tensor<3x3xf32, #CSR>, tensor<1024x<f32, #SparseVector)
                             out (%c : tensor<1024xf32, #SparseVector>)
        return %result : tensor<3x3xf32, #CSR>
    }


    // Main driver
    func.func @main() {

    }






}

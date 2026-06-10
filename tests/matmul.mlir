// RUN: tutorial-opt %s --sparsifier="enable-runtime-library=true" | mlir-runner -e main -entry-point-result=void -shared-libs=../../+_repo_rules+llvm-project/mlir/libmlir_c_runner_utils.so,../../+_repo_rules+llvm-project/mlir/libmlir_runner_utils.so | FileCheck %s

#CSR = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : dense, d1 : compressed)}>

module {

  //
  // Sparse kernel.
  //
  func.func @sparse_matmul(%a: tensor<3x3xf32, #CSR>,
                           %b: tensor<3x3xf32, #CSR>)
                           -> tensor<3x3xf32, #CSR> {
    %c = tensor.empty() : tensor<3x3xf32, #CSR>
    %result = linalg.matmul
                ins(%a, %b : tensor<3x3xf32, #CSR>, tensor<3x3xf32, #CSR>)
                outs(%c : tensor<3x3xf32, #CSR>) -> tensor<3x3xf32, #CSR>
    return %result : tensor<3x3xf32, #CSR>
  }


  //
  // Main driver
  //
  func.func @main() {
    %d1 = arith.constant sparse<[[0,0],[1,1],[2,2]], [1.0, 2.0, 3.0]> : tensor<3x3xf32>
    %d2 = arith.constant sparse<[[0,0],[1,1],[2,2]], [4.0, 5.0, 6.0]> : tensor<3x3xf32>

    %s1 = sparse_tensor.convert %d1 : tensor<3x3xf32> to tensor<3x3xf32, #CSR>
    %s2 = sparse_tensor.convert %d2 : tensor<3x3xf32> to tensor<3x3xf32, #CSR>

    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 3
    // CHECK-NEXT: dim = ( 3, 3 )
    // CHECK-NEXT: lvl = ( 3, 3 )
    // CHECK-NEXT: pos[1] : ( 0, 1, 2, 3 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 2 )
    // CHECK-NEXT: values : ( 1, 2, 3 )
    // CHECK-NEXT: ----
    sparse_tensor.print %s1 : tensor<3x3xf32, #CSR>

    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 3
    // CHECK-NEXT: dim = ( 3, 3 )
    // CHECK-NEXT: lvl = ( 3, 3 )
    // CHECK-NEXT: pos[1] : ( 0, 1, 2, 3 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 2 )
    // CHECK-NEXT: values : ( 4, 5, 6 )
    // CHECK-NEXT: ----
    sparse_tensor.print %s2 : tensor<3x3xf32, #CSR>


    %0 = call @sparse_matmul(%s1, %s2) : (tensor<3x3xf32, #CSR>, tensor<3x3xf32, #CSR>) -> tensor<3x3xf32, #CSR>

    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 3
    // CHECK-NEXT: dim = ( 3, 3 )
    // CHECK-NEXT: lvl = ( 3, 3 )
    // CHECK-NEXT: pos[1] : ( 0, 1, 2, 3 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 2 )
    // CHECK-NEXT: values : ( 4, 10, 18 )
    // CHECK-NEXT: ----
    sparse_tensor.print %0 : tensor<3x3xf32, #CSR>


    // Release the resources
    bufferization.dealloc_tensor %0  : tensor<3x3xf32, #CSR>
    bufferization.dealloc_tensor %s1 : tensor<3x3xf32, #CSR>
    bufferization.dealloc_tensor %s2 : tensor<3x3xf32, #CSR>

    return
  }
}

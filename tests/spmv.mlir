// RUN: tutorial-opt %s --linalg-generalize-named-ops --sparsifier="enable-runtime-library=true" | mlir-runner -e main -entry-point-result=void -shared-libs=../../+_repo_rules+llvm-project/mlir/libmlir_c_runner_utils.so,../../+_repo_rules+llvm-project/mlir/libmlir_runner_utils.so | FileCheck %s

#CSR = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : dense, d1 : compressed)}>
#SparseVector = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed)}>

module {

    //
    // Sparse kernel
    //
    func.func @spmv(%a: tensor<4x4xf32, #CSR>, %b: tensor<4xf32, #SparseVector>) -> tensor<4xf32, #SparseVector> {
        %z = arith.constant 0.0 : f32
        %c = tensor.empty() : tensor<4xf32>
        %x = linalg.fill ins(%z : f32) outs(%c : tensor<4xf32>) -> tensor<4xf32>
        %mv = linalg.matvec ins(%a, %b : tensor<4x4xf32, #CSR>, tensor<4xf32, #SparseVector>)
                            outs(%x : tensor<4xf32>) -> tensor<4xf32>
        %0 = sparse_tensor.convert %mv : tensor<4xf32> to tensor<4xf32, #SparseVector>
        return %0 : tensor<4xf32, #SparseVector>
    }

    //
    // Main driver
    //
    func.func @main() {
        %d1 = arith.constant sparse<[[0, 0], [1, 1], [2, 2], [3, 3]], [1.0, 2.0, 3.0, 4.0]> : tensor<4x4xf32>
        %d2 = arith.constant sparse<[[0], [1], [2], [3]], [5.0, 6.0, 7.0, 8.0]> : tensor<4xf32>

        %s1 = sparse_tensor.convert %d1 : tensor<4x4xf32> to tensor<4x4xf32, #CSR>
        %s2 = sparse_tensor.convert %d2 : tensor<4xf32> to tensor<4xf32, #SparseVector>

        //
        // CHECK:      ---- Sparse Tensor ----
        // CHECK-NEXT: nse = 4
        // CHECK-NEXT: dim = ( 4, 4 )
        // CHECK-NEXT: lvl = ( 4, 4 )
        // CHECK-NEXT: pos[1] : ( 0, 1, 2, 3, 4 )
        // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3 )
        // CHECK-NEXT: values : ( 1, 2, 3, 4 )
        // CHECK-NEXT: ----
        //
        //
        // CHECK:      ---- Sparse Tensor ----
        // CHECK-NEXT: nse = 4
        // CHECK-NEXT: dim = ( 4 )
        // CHECK-NEXT: lvl = ( 4 )
        // CHECK-NEXT: pos[0] : ( 0, 4 )
        // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3 )
        // CHECK-NEXT: values : ( 5, 6, 7, 8 )
        // CHECK-NEXT: ----
        //
        sparse_tensor.print %s1 : tensor<4x4xf32, #CSR>
        sparse_tensor.print %s2 : tensor<4xf32, #SparseVector>

        %0 = call @spmv(%s1, %s2) : (tensor<4x4xf32, #CSR>, tensor<4xf32, #SparseVector>) -> tensor<4xf32, #SparseVector>

        //
        // CHECK:      ---- Sparse Tensor ----
        // CHECK-NEXT: nse = 4
        // CHECK-NEXT: dim = ( 4 )
        // CHECK-NEXT: lvl = ( 4 )
        // CHECK-NEXT: pos[0] : ( 0, 4 )
        // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3 )
        // CHECK-NEXT: values : ( 5, 12, 21, 32 )
        // CHECK-NEXT: ----
        //
        sparse_tensor.print %0 : tensor<4xf32, #SparseVector>

        bufferization.dealloc_tensor %s1 : tensor<4x4xf32, #CSR>
        bufferization.dealloc_tensor %s2 : tensor<4xf32, #SparseVector>
        bufferization.dealloc_tensor %0 : tensor<4xf32, #SparseVector>

        return
    }
}

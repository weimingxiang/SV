#pragma once

// @generated by tools/codegen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/chain_matmul_ops.h>

namespace at {


// aten::chain_matmul(Tensor[] matrices) -> Tensor
TORCH_API inline at::Tensor chain_matmul(at::TensorList matrices) {
    return at::_ops::chain_matmul::call(matrices);
}

// aten::chain_matmul.out(Tensor[] matrices, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & chain_matmul_out(at::Tensor & out, at::TensorList matrices) {
    return at::_ops::chain_matmul_out::call(matrices, out);
}

// aten::chain_matmul.out(Tensor[] matrices, *, Tensor(a!) out) -> Tensor(a!)
TORCH_API inline at::Tensor & chain_matmul_outf(at::TensorList matrices, at::Tensor & out) {
    return at::_ops::chain_matmul_out::call(matrices, out);
}

}

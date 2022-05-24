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



#include <ATen/ops/cumprod_backward_ops.h>

namespace at {


// aten::cumprod_backward(Tensor grad, Tensor input, int dim, Tensor output) -> Tensor
TORCH_API inline at::Tensor cumprod_backward(const at::Tensor & grad, const at::Tensor & input, int64_t dim, const at::Tensor & output) {
    return at::_ops::cumprod_backward::call(grad, input, dim, output);
}

}

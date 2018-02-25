# Operation Semantics

The following describes the semantics of operations defined in the
[`ComputationBuilder`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)
interface. Typically, these operations map one-to-one to operations defined in
the RPC interface in
[`xla_data.proto`](https://www.tensorflow.org/code/tensorflow/compiler/xla/xla_data.proto).

A note on nomenclature: the generalized data type XLA deals with is an
N-dimensional array holding elements of some uniform type (such as 32-bit
float). Throughout the documentation, *array* is used to denote an
arbitrary-dimensional array. For convenience, special cases have more specific
and familiar names; for example a *vector* is a 1-dimensional array and a
*matrix* is a 2-dimensional array.

## Broadcast

See also
[`ComputationBuilder::Broadcast`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

Adds dimensions to an array by duplicating the data in the array.

<b> `Broadcast(operand, broadcast_sizes)` </b>

Arguments         | Type                    | Semantics
----------------- | ----------------------- | -------------------------------
`operand`         | `ComputationDataHandle` | The array to duplicate
`broadcast_sizes` | `ArraySlice<int64>`     | The sizes of the new dimensions

The new dimensions are inserted on the left, i.e. if `broadcast_sizes` has
values `{a0, ..., aN}` and the operand shape has dimensions `{b0, ..., bM}` then
the shape of the output has dimensions `{a0, ..., aN, b0, ..., bM}`.

The new dimensions index into copies of the operand, i.e.

```
output[i0, ..., iN, j0, ..., jM] = operand[j0, ..., jM]
```

For example, if `operand` is a scalar `f32` with value `2.0f`, and
`broadcast_sizes` is `{2, 3}`, then the result will be an array with shape
`f32[2, 3]` and all the values in the result will be `2.0f`.

## Call

See also
[`ComputationBuilder::Call`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

Invokes a computation with the given arguments.

<b> `Call(computation, args...)` </b>

| Arguments     | Type                     | Semantics                        |
| ------------- | ------------------------ | -------------------------------- |
| `computation` | `Computation`            | computation of type `T_0, T_1,   |
:               :                          : ..., T_N -> S` with N parameters :
:               :                          : of arbitrary type                :
| `args`        | sequence of N            | N arguments of arbitrary type    |
:               : `ComputationDataHandle`s :                                  :

The arity and types of the `args` must match the parameters of the
`computation`. It is allowed to have no `args`.

## Clamp

See also
[`ComputationBuilder::Clamp`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

Clamps an operand to within the range between a minimum and maximum value.

<b> `Clamp(computation, args...)` </b>

| Arguments     | Type                    | Semantics                        |
| ------------- | ----------------------- | -------------------------------- |
| `computation` | `Computation`           | computation of type `T_0, T_1,   |
:               :                         : ..., T_N -> S` with N parameters :
:               :                         : of arbitrary type                :
| `operand`     | `ComputationDataHandle` | array of type T                  |
| `min`         | `ComputationDataHandle` | array of type T                  |
| `max`         | `ComputationDataHandle` | array of type T                  |

Given an operand and minimum and maximum values, returns the operand if it is in
the range between the minimum and maximum, else returns the minimum value if the
operand is below this range or the maximum value if the operand is above this
range.  That is, `clamp(x, a, b) =  max(min(x, a), b)`.

All three arrays must be the same shape. Alternately, as a restricted form of
[broadcasting](broadcasting.md), `min` and/or `max` can be a scalar of type `T`.

Example with scalar `min` and `max`:

```
let operand: s32[3] = {-1, 5, 9};
let min: s32 = 0;
let max: s32 = 6;
==>
Clamp(operand, min, max) = s32[3]{0, 5, 6};
```

## Collapse

See also
[`ComputationBuilder::Collapse`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)
and the @{tf.reshape} operation.

Collapses dimensions of an array into one dimension.

<b> `Collapse(operand, dimensions)` </b>

| Arguments    | Type                    | Semantics                           |
| ------------ | ----------------------- | ----------------------------------- |
| `operand`    | `ComputationDataHandle` | array of type T                     |
| `dimensions` | `int64` vector          | in-order, consecutive subset of T's |
:              :                         : dimensions.                         :

Collapse replaces the given subset of the operand's dimensions by a single
dimension. The input arguments are an arbitrary array of type T and a
compile-time-constant vector of dimension indices. The dimension indices must be
an in-order (low to high dimension numbers), consecutive subset of T's
dimensions. Thus, {0, 1, 2}, {0, 1}, or {1, 2} are all valid dimension sets, but
{1, 0} or {0, 2} are not. They are replaced by a single new dimension, in the
same position in the dimension sequence as those they replace, with the new
dimension size equal to the product of original dimension sizes. The lowest
dimension number in `dimensions` is the slowest varying dimension (most major)
in the loop nest which collapses these dimension, and the highest dimension
number is fastest varying (most minor). See the @{tf.reshape} operator
if more general collapse ordering is needed.

For example, let v be an array of 24 elements:

```
let v = f32[4x2x3] {{{10, 11, 12},  {15, 16, 17}},
                    {{20, 21, 22},  {25, 26, 27}},
                    {{30, 31, 32},  {35, 36, 37}},
                    {{40, 41, 42},  {45, 46, 47}}};

// Collapse to a single dimension, leaving one dimension.
let v012 = Collapse(v, {0,1,2});
then v012 == f32[24] {10, 11, 12, 15, 16, 17,
                      20, 21, 22, 25, 26, 27,
                      30, 31, 32, 35, 36, 37,
                      40, 41, 42, 45, 46, 47};

// Collapse the two lower dimensions, leaving two dimensions.
let v01 = Collapse(v, {0,1});
then v01 == f32[4x6] {{10, 11, 12, 15, 16, 17},
                      {20, 21, 22, 25, 26, 27},
                      {30, 31, 32, 35, 36, 37},
                      {40, 41, 42, 45, 46, 47}};

// Collapse the two higher dimensions, leaving two dimensions.
let v12 = Collapse(v, {1,2});
then v12 == f32[8x3] {{10, 11, 12},
                      {15, 16, 17},
                      {20, 21, 22},
                      {25, 26, 27},
                      {30, 31, 32},
                      {35, 36, 37},
                      {40, 41, 42},
                      {45, 46, 47}};

```

## Concatenate

See also
[`ComputationBuilder::ConcatInDim`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

Concatenate composes an array from multiple array operands. The array is of the
same rank as each of the input array operands (which must be of the same rank as
each other) and contains the arguments in the order that they were specified.

<b> `Concatenate(operands..., dimension)` </b>

| Arguments   | Type                    | Semantics                            |
| ----------- | ----------------------- | ------------------------------------ |
| `operands`  | sequence of N           | N arrays of type T with dimensions   |
:             : `ComputationDataHandle` : [L0, L1, ...]. Requires N >= 1.      :
| `dimension` | `int64`                 | A value in the interval `[0, N)`     |
:             :                         : that names the dimension to be       :
:             :                         : concatenated between the `operands`. :

With the exception of `dimension` all dimensions must be the same. This is
because XLA does not support "ragged" arrays Also note that rank-0 values
cannot be concatenated (as it's impossible to name the dimension along which the
concatenation occurs).

1-dimensional example:

```
Concat({{2, 3}, {4, 5}, {6, 7}}, 0)
>>> {2, 3, 4, 5, 6, 7}
```

2-dimensional example:

```
let a = {
  {1, 2},
  {3, 4},
  {5, 6},
};
let b = {
  {7, 8},
};
Concat({a, b}, 0)
>>> {
  {1, 2},
  {3, 4},
  {5, 6},
  {7, 8},
}
```

Diagram:
<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/ops_concatenate.png">
</div>

## ConvertElementType

See also
[`ComputationBuilder::ConvertElementType`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

Similar to an element-wise `static_cast` in C++, performs an element-wise
conversion operation from a data shape to a target shape. The dimensions must
match, and the conversion is an element-wise one; e.g. `s32` elements become
`f32` elements via an `s32`-to-`f32` conversion routine.

<b> `ConvertElementType(operand, new_element_type)` </b>

Arguments          | Type                    | Semantics
------------------ | ----------------------- | ---------------------------
`operand`          | `ComputationDataHandle` | array of type T with dims D
`new_element_type` | `PrimitiveType`         | type U

If the dimensions of the operand and the target shape do not match, or an
invalid conversion is requested (e.g. to/from a tuple) an error will be
produced.

A conversion such as `T=s32` to `U=f32` will perform a normalizing int-to-float
conversion routine such as round-to-nearest-even.

> Note: The precise float-to-int and visa-versa conversions are currently
> unspecified, but may become additional arguments to the convert operation in
> the future.  Not all possible conversions have been implemented for all
>targets.

```
let a: s32[3] = {0, 1, 2};
let b: f32[3] = convert(a, f32);
then b == f32[3]{0.0, 1.0, 2.0}
```

## Conv (convolution)

See also
[`ComputationBuilder::Conv`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

As ConvWithGeneralPadding, but the padding is specified in a short-hand way as
either SAME or VALID. SAME padding pads the input (`lhs`) with zeroes so that
the output has the same shape as the input when not taking striding into
account. VALID padding simply means no padding.

## ConvWithGeneralPadding (convolution)

See also
[`ComputationBuilder::ConvWithGeneralPadding`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

Computes a convolution of the kind used in neural networks. Here, a convolution
can be thought of as a n-dimensional window moving across a n-dimensional base
area and a computation is performed for each possible position of the window.

| Arguments        | Type                    | Semantics                     |
| ---------------- | ----------------------- | ----------------------------- |
| `lhs`            | `ComputationDataHandle` | rank n+2 array of inputs      |
| `rhs`            | `ComputationDataHandle` | rank n+2 array of kernel      |
:                  :                         : weights                       :
| `window_strides` | `ArraySlice<int64>`     | n-d array of kernel strides   |
| `padding`        | `ArraySlice<pair<int64, | n-d array of (low, high)      |
:                  : int64>>`                : padding                       :
| `lhs_dilation`   | `ArraySlice<int64>`     | n-d lhs dilation factor array |
| `rhs_dilation`   | `ArraySlice<int64>`     | n-d rhs dilation factor array |

Let n be the number of spatial dimensions. The `lhs` argument is a rank n+2
array describing the base area. This is called the input, even though of course
the rhs is also an input. In a neural network, these are the input activations.
The n+2 dimensions are, in this order:

*   `batch`: Each coordinate in this dimension represents an independent input
    for which convolution is carried out.
*   `z/depth/features`: Each (y,x) position in the base area has a vector
    associated to it, which goes into this dimension.
*   `spatial_dims`: Describes the `n` spatial dimensions that define the base
    area that the window moves across.

The `rhs` argument is a rank n+2 array describing the convolutional
filter/kernel/window. The dimensions are, in this order:

*   `output-z`: The `z` dimension of the output.
*   `input-z`: The size of this dimension should equal the size of the `z`
    dimension in lhs.
*   `spatial_dims`: Describes the `n` spatial dimensions that define the n-d
    window that moves across the base area.

The `window_strides` argument specifies the stride of the convolutional window
in the spatial dimensions. For example, if the stride in a the first spatial
dimension is 3, then the window can only be placed at coordinates where the
first spatial index is divisible by 3.

The `padding` argument specifies the amount of zero padding to be applied to the
base area. The amount of padding can be negative -- the absolute value of
negative padding indicates the number of elements to remove from the specified
dimension before doing the convolution. `padding[0]` specifies the padding for
dimension `y` and `padding[1]` specifies the padding for dimension `x`. Each
pair has the low padding as the first element and the high padding as the second
element. The low padding is applied in the direction of lower indices while the
high padding is applied in the direction of higher indices. For example, if
`padding[1]` is `(2,3)` then there will be a padding by 2 zeroes on the left and
by 3 zeroes on the right in the second spatial dimension. Using padding is
equivalent to inserting those same zero values into the input (`lhs`) before
doing the convolution.

The `lhs_dilation` and `rhs_dilation` arguments specify the dilation factor to
be applied to the lhs and rhs, respectively, in each spatial dimension. If the
dilation factor in a spatial dimension is d, then d-1 holes are implicitly
placed between each of the entries in that dimension, increasing the size of the
array. The holes are filled with a no-op value, which for convolution means
zeroes.

Dilation of the rhs is also called atrous convolution. For more details, see the
@{tf.nn.atrous_conv2d}. Dilation of the lhs is
also called deconvolution.

The output shape has these dimensions, in this order:

*   `batch`: Same size as `batch` on the input (`lhs`).
*   `z`: Same size as `output-z` on the kernel (`rhs`).
*   `spatial_dims`: One value for each valid placement of the convolutional
    window.

The valid placements of the convolutional window are determined by the strides
and the size of the base area after padding.

To describe what a convolution does, consider a 2d convolution, and pick some
fixed `batch`, `z`, `y`, `x` coordinates in the output. Then `(y,x)` is a
position of a corner of the window within the base area (e.g. the upper left
corner, depending on how you interpret the spatial dimensions). We now have a 2d
window, taken from the base area, where each 2d point is associated to a 1d
vector, so we get a 3d box. From the convolutional kernel, since we fixed the
output coordinate `z`, we also have a 3d box. The two boxes have the same
dimensions, so we can take the sum of the element-wise products between the two
boxes (similar to a dot product). That is the output value.

Note that if `output-z` is e.g., 5, then each position of the window produces 5
values in the output into the `z` dimension of the output. These values differ
in what part of the convolutional kernel is used - there is a separate 3d box of
values used for each `output-z` coordinate. So you could think of it as 5
separate convolutions with a different filter for each of them.

Here is pseudo-code for a 2d convolution with padding and striding:

```
for (b, oz, oy, ox) {  // output coordinates
  value = 0;
  for (iz, ky, kx) {  // kernel coordinates and input z
    iy = oy*stride_y + ky - pad_low_y;
    ix = ox*stride_x + kx - pad_low_x;
    if ((iy, ix) inside the base area considered without padding) {
      value += input(b, iz, iy, ix) * kernel(oz, iz, ky, kx);
    }
  }
  output(b, oz, oy, ox) = value;
}
```

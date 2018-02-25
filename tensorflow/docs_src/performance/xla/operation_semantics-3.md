## Reduce

See also
[`ComputationBuilder::Reduce`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

Applies a reduction function to an array.

<b> `Reduce(operand, init_value, computation, dimensions)` </b>

| Arguments     | Type                    | Semantics                        |
| ------------- | ----------------------- | -------------------------------- |
| `operand`     | `ComputationDataHandle` | array of type `T`                |
| `init_value`  | `ComputationDataHandle` | scalar of type `T`               |
| `computation` | `Computation`           | computation of type `T, T -> T`  |
| `dimensions`  | `int64` array           | unordered array of dimensions to |
:               :                         : reduce                           :

Conceptually, this operation reduces one or more dimensions in the input array
into scalars. The rank of the result array is `rank(operand) - len(dimensions)`.
`init_value` is the initial value used for every reduction and may also be
inserted anywhere during computation if the back-end chooses to do so. So in
most cases `init_value` should be an identity of the reduction function (for
example, 0 for addition).

The evaluation order of the reduction function is arbitrary and may be
non-deterministic. Therefore, the reduction function should not be overly
sensitive to reassociation.

Some reduction functions like addition are not strictly associative for floats.
However, if the range of the data is limited, floating-point addition is close
enough to being associative for most practical uses. It is possible to conceive
of some completely non-associative reductions, however, and these will produce
incorrect or unpredictable results in XLA reductions.

As an example, when reducing across the one dimension in a 1D array with values
[10, 11, 12, 13], with reduction function `f` (this is `computation`) then that
could be computed as

`f(10, f(11, f(12, f(init_value, 13)))`

but there are also many other possibilities, e.g.

`f(init_value, f(f(10, f(init_value, 11)), f(f(init_value, 12), f(13,
init_value))))`

The following is a rough pseudo-code example of how reduction could be
implemented, using summation as the reduction computation with an initial value
of 0.

```python
result_shape <- remove all dims in dimensions from operand_shape

# Iterate over all elements in result_shape. The number of r's here is equal
# to the rank of the result
for r0 in range(result_shape[0]), r1 in range(result_shape[1]), ...:
  # Initialize this result element
  result[r0, r1...] <- 0

  # Iterate over all the reduction dimensions
  for d0 in range(dimensions[0]), d1 in range(dimensions[1]), ...:
    # Increment the result element with the value of the operand's element.
    # The index of the operand's element is constructed from all ri's and di's
    # in the right order (by construction ri's and di's together index over the
    # whole operand shape).
    result[r0, r1...] += operand[ri... di]
```

Here's an example of reducing a 2D array (matrix). The shape has rank 2,
dimension 0 of size 2 and dimension 1 of size 3:

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:35%" src="https://www.tensorflow.org/images/ops_2d_matrix.png">
</div>

Results of reducing dimensions 0 or 1 with an "add" function:

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:35%" src="https://www.tensorflow.org/images/ops_reduce_from_2d_matrix.png">
</div>

Note that both reduction results are 1D arrays. The diagram shows one as column
and another as row just for visual convenience.

For a more complex example, here is a 3D array. Its rank is 3, dimension 0 of
size 4, dimension 1 of size 2 and dimension 2 of size 3. For simplicity, the
values 1 to 6 are replicated across dimension 0.

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:35%" src="https://www.tensorflow.org/images/ops_reduce_from_3d_matrix.png">
</div>

Similarly to the 2D example, we can reduce just one dimension. If we reduce
dimension 0, for example, we get a rank-2 array where all values across
dimension 0 were folded into a scalar:

```text
|  4   8  12 |
| 16  20  24 |
```

If we reduce dimension 2, we also get a rank-2 array where all values across
dimension 2 were folded into a scalar:

```text
| 6  15 |
| 6  15 |
| 6  15 |
| 6  15 |
```

Note that the relative order between the remaining dimensions in the input is
preserved in the output, but some dimensions may get assigned new numbers (since
the rank changes).

We can also reduce multiple dimensions. Add-reducing dimensions 0 and 1 produces
the 1D array `| 20 28 36 |`.

Reducing the 3D array over all its dimensions produces the scalar `84`.

## ReducePrecision

See also
[`ComputationBuilder::ReducePrecision`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

Models the effect of converting floating-point values to a lower-precision
format (such as IEEE-FP16) and back to the original format.  The number of
exponent and mantissa bits in the lower-precision format can be specified
arbitrarily, although all bit sizes may not be supported on all hardware
implementations.

<b> `ReducePrecision(operand, mantissa_bits, exponent_bits)` </b>

| Arguments           | Type                    | Semantics                    |
| ------------------- | ----------------------- | ---------------------------- |
| `operand`           | `ComputationDataHandle` | array of floating-point type |
:                     :                         : `T`.                         :
| `exponent_bits`     | `int32`                 | number of exponent bits in   |
:                     :                         : lower-precision format       :
| `mantissa_bits`     | `int32`                 | number of mantissa bits in   |
:                     :                         : lower-precision format       :

The result is an array of type `T`.  The input values are rounded to the nearest
value representable with the given number of mantissa bits (using "ties to even"
semantics), and any values that exceed the range specified by the number of
exponent bits are clamped to positive or negative infinity.  `NaN` values are
retained, although they may be converted to canonical `NaN` values.

The lower-precision format must have at least one exponent bit (in order to
distinguish a zero value from an infinity, since both have a zero mantissa), and
must have a non-negative number of mantissa bits.  The number of exponent or
mantissa bits may exceed the corresponding value for type `T`; the corresponding
portion of the conversion is then simply a no-op.


## ReduceWindow

See also
[`ComputationBuilder::ReduceWindow`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

Applies a reduction function to all elements in each window of the input
multi-dimensional array, producing an output multi-dimensional array with the
same number of elements as the number of valid positions of the window. A
pooling layer can be expressed as a `ReduceWindow`.

<b> `ReduceWindow(operand, init_value, computation, window_dimensions,
window_strides, padding)` </b>

| Arguments           | Type                    | Semantics                    |
| ------------------- | ----------------------- | ---------------------------- |
| `operand`           | `ComputationDataHandle` | N dimensional array          |
:                     :                         : containing elements of type  :
:                     :                         : T. This is the base area on  :
:                     :                         : which the window is placed.  :
| `init_value`        | `ComputationDataHandle` | Starting value for the       |
:                     :                         : reduction. See [Reduce]      :
:                     :                         : (#reduce) for details.       :
| `computation`       | `Computation`           | Reduction function of type   |
:                     :                         : `T, T -> T`, to apply to all :
:                     :                         : elements in each window      :
| `window_dimensions` | `ArraySlice<int64>`     | array of integers for window |
:                     :                         : dimension values             :
| `window_strides`    | `ArraySlice<int64>`     | array of integers for window |
:                     :                         : stride values                :
| `padding`           | `Padding`               | padding type for window      |
:                     :                         : (Padding\:\:kSame or         :
:                     :                         : Padding\:\:kValid)           :

Below code and figure shows an example of using `ReduceWindow`. Input is a
matrix of size [4x6] and both window_dimensions and window_stride_dimensions are
[2x3].

```
// Create a computation for the reduction (maximum).
Computation max;
{
  ComputationBuilder builder(client_, "max");
  auto y = builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "y");
  auto x = builder.Parameter(1, ShapeUtil::MakeShape(F32, {}), "x");
  builder.Max(y, x);
  max = builder.Build().ConsumeValueOrDie();
}

// Create a ReduceWindow computation with the max reduction computation.
ComputationBuilder builder(client_, "reduce_window_2x3");
auto shape = ShapeUtil::MakeShape(F32, {4, 6});
auto input = builder.Parameter(0, shape, "input");
builder.ReduceWindow(
    input, *max,
    /*init_val=*/builder.ConstantLiteral(LiteralUtil::MinValue(F32)),
    /*window_dimensions=*/{2, 3},
    /*window_stride_dimensions=*/{2, 3},
    Padding::kValid);
```

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:35%" src="https://www.tensorflow.org/images/ops_reduce_window.png">
</div>

Stride of 1 in a dimension specifies that the position of a window in the
dimension is 1 element away from its adjacent window. In order to specify that
no windows overlap with each other, window_stride_dimensions should be equal to
window_dimensions. The figure below illustrates the use of two different stride
values. Padding is applied to each dimension of the input and the calculations
are the same as though the input came in with the dimensions it has after
padding.

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:75%" src="https://www.tensorflow.org/images/ops_reduce_window_stride.png">
</div>

The evaluation order of the reduction function is arbitrary and may be
non-deterministic. Therefore, the reduction function should not be overly
sensitive to reassociation. See the discussion about associativity in the
context of [`Reduce`](#reduce) for more details.

## Reshape

See also
[`ComputationBuilder::Reshape`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)
and the [`Collapse`](#collapse) operation.

Reshapes the dimensions of an array into a new configuration.

<b> `Reshape(operand, new_sizes)` </b>
<b> `Reshape(operand, dimensions, new_sizes)` </b>

Arguments    | Type                    | Semantics
------------ | ----------------------- | ---------------------------------------
`operand`    | `ComputationDataHandle` | array of type T
`dimensions` | `int64` vector          | order in which dimensions are collapsed
`new_sizes`  | `int64` vector          | vector of sizes of new dimensions

Conceptually, reshape first flattens an array into a one-dimensional vector of
data values, and then refines this vector into a new shape. The input arguments
are an arbitrary array of type T, a compile-time-constant vector of dimension
indices, and a compile-time-constant vector of dimension sizes for the result.
The values in the `dimension` vector, if given, must be a permutation of all of
T's dimensions; the default if not given is `{0, ..., rank - 1}`. The order of
the dimensions in `dimensions` is from slowest-varying dimension (most major) to
fastest-varying dimension (most minor) in the loop nest which collapses the
input array into a single dimension. The `new_sizes` vector determines the size
of the output array. The value at index 0 in `new_sizes` is the size of
dimension 0, the value at index 1 is the size of dimension 1, and so on. The
product of the `new_size` dimensions must equal the product of the operand's
dimension sizes. When refining the collapsed array into the multidimensional
array defined by `new_sizes`, the dimensions in `new_sizes` are ordered from
slowest varying (most major) and to fastest varying (most minor).

For example, let v be an array of 24 elements:

```
let v = f32[4x2x3] {{{10, 11, 12}, {15, 16, 17}},
                    {{20, 21, 22}, {25, 26, 27}},
                    {{30, 31, 32}, {35, 36, 37}},
                    {{40, 41, 42}, {45, 46, 47}}};

In-order collapse:
let v012_24 = Reshape(v, {0,1,2}, {24});
then v012_24 == f32[24] {10, 11, 12, 15, 16, 17, 20, 21, 22, 25, 26, 27,
                         30, 31, 32, 35, 36, 37, 40, 41, 42, 45, 46, 47};

let v012_83 = Reshape(v, {0,1,2}, {8,3});
then v012_83 == f32[8x3] {{10, 11, 12}, {15, 16, 17},
                          {20, 21, 22}, {25, 26, 27},
                          {30, 31, 32}, {35, 36, 37},
                          {40, 41, 42}, {45, 46, 47}};

Out-of-order collapse:
let v021_24 = Reshape(v, {1,2,0}, {24});
then v012_24 == f32[24]  {10, 20, 30, 40, 11, 21, 31, 41, 12, 22, 32, 42,
                          15, 25, 35, 45, 16, 26, 36, 46, 17, 27, 37, 47};

let v021_83 = Reshape(v, {1,2,0}, {8,3});
then v021_83 == f32[8x3] {{10, 20, 30}, {40, 11, 21},
                          {31, 41, 12}, {22, 32, 42},
                          {15, 25, 35}, {45, 16, 26},
                          {36, 46, 17}, {27, 37, 47}};


let v021_262 = Reshape(v, {1,2,0}, {2,6,2});
then v021_262 == f32[2x6x2] {{{10, 20}, {30, 40},
                              {11, 21}, {31, 41},
                              {12, 22}, {32, 42}},
                             {{15, 25}, {35, 45},
                              {16, 26}, {36, 46},
                              {17, 27}, {37, 47}}};
```

As a special case, reshape can transform a single-element array to a scalar and
vice versa. For example,

```
Reshape(f32[1x1] {{5}}, {0,1}, {}) == 5;
Reshape(5, {}, {1,1}) == f32[1x1] {{5}};
```

## Rev (reverse)

See also
[`ComputationBuilder::Rev`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

<b>`Rev(operand, dimensions)`</b>

Arguments    | Type                    | Semantics
------------ | ----------------------- | ---------------------
`operand`    | `ComputationDataHandle` | array of type T
`dimensions` | `ArraySlice<int64>`     | dimensions to reverse

Reverses the order of elements in the `operand` array along the specified
`dimensions`, generating an output array of the same shape. Each element of the
operand array at a multidimensional index is stored into the output array at a
transformed index. The multidimensional index is transformed by reversing the
index in each dimension to be reversed (i.e., if a dimension of size N is one of
the reversing dimensions, its index i is transformed into N - 1 - i).

One use for the `Rev` operation is to reverse the convolution weight array along
the two window dimensions during the gradient computation in neural networks.

## RngBernoulli

See also
[`ComputationBuilder::RngBernoulli`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

Constructs an output of a given shape with random numbers generated following
the Bernoulli distribution. The parameter needs to be a scalar valued F32
operand while the output shape needs to have elemental type U32.

<b>`RngBernoulli(mean, shape)`</b>

| Arguments | Type                    | Semantics                             |
| --------- | ----------------------- | ------------------------------------- |
| `mean`    | `ComputationDataHandle` | Scalar of type F32 specifying mean of |
:           :                         : generated numbers                     :
| `shape`   | `Shape`                 | Output shape of type U32              |

## RngNormal

See also
[`ComputationBuilder::RngNormal`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

Constructs an output of a given shape with random numbers generated following
the $$N(\mu, \sigma)$$ normal distribution. The parameters `mu` and `sigma`, and
output shape have to have elemental type F32. The parameters furthermore have to
be scalar valued.

<b>`RngNormal(mean, sigma, shape)`</b>

| Arguments | Type                    | Semantics                              |
| --------- | ----------------------- | -------------------------------------- |
| `mu`      | `ComputationDataHandle` | Scalar of type F32 specifying mean of  |
:           :                         : generated numbers                      :
| `sigma`   | `ComputationDataHandle` | Scalar of type F32 specifying standard |
:           :                         : deviation of generated numbers         :
| `shape`   | `Shape`                 | Output shape of type F32               |

## RngUniform

See also
[`ComputationBuilder::RngUniform`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

Constructs an output of a given shape with random numbers generated following
the uniform distribution over the interval $$[a,b)$$. The parameters and output
shape may be either F32, S32 or U32, but the types have to be consistent.
Furthermore, the parameters need to be scalar valued. If $$b <= a$$ the result
is implementation-defined.

<b>`RngUniform(a, b, shape)`</b>

| Arguments | Type                    | Semantics                         |
| --------- | ----------------------- | --------------------------------- |
| `a`       | `ComputationDataHandle` | Scalar of type T specifying lower |
:           :                         : limit of interval                 :
| `b`       | `ComputationDataHandle` | Scalar of type T specifying upper |
:           :                         : limit of interval                 :
| `shape`   | `Shape`                 | Output shape of type T            |

## SelectAndScatter

See also
[`ComputationBuilder::SelectAndScatter`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

This operation can be considered as a composite operation that first computes
`ReduceWindow` on the `operand` array to select an element from each window, and
then scatters the `source` array to the indices of the selected elements to
construct an output array with the same shape as the operand array. The binary
`select` function is used to select an element from each window by applying it
across each window, and it is called with the property that the first
parameter's index vector is lexicographically less than the second parameter's
index vector. The `select` function returns `true` if the first parameter is
selected and returns `false` if the second parameter is selected, and the
function must hold transitivity (i.e., if `select(a, b)` and `select(b, c)` are
`true`, then `select(a, c)` is also `true`) so that the selected element does
not depend on the order of the elements traversed for a given window.

The function `scatter` is applied at each selected index in the output array. It
takes two scalar parameters:

1.  Current value at the selected index in the output array
2.  The scatter value from `source` that applies to the selected index

It combines the two parameters and returns a scalar value that's used to update
the value at the selected index in the output array. Initially, all indices of
the output array are set to `init_value`.

The output array has the same shape as the `operand` array and the `source`
array must have the same shape as the result of applying a `ReduceWindow`
operation on the `operand` array. `SelectAndScatter` can be used to
backpropagate the gradient values for a pooling layer in a neural network.

<b>`SelectAndScatter(operand, select, window_dimensions, window_strides,
padding, source, init_value, scatter)`</b>

| Arguments           | Type                    | Semantics                    |
| ------------------- | ----------------------- | ---------------------------- |
| `operand`           | `ComputationDataHandle` | array of type T over which   |
:                     :                         : the windows slide            :
| `select`            | `Computation`           | binary computation of type   |
:                     :                         : `T, T -> PRED`, to apply to  :
:                     :                         : all elements in each window; :
:                     :                         : returns `true` if the first  :
:                     :                         : parameter is selected and    :
:                     :                         : returns `false` if the       :
:                     :                         : second parameter is selected :
| `window_dimensions` | `ArraySlice<int64>`     | array of integers for window |
:                     :                         : dimension values             :
| `window_strides`    | `ArraySlice<int64>`     | array of integers for window |
:                     :                         : stride values                :
| `padding`           | `Padding`               | padding type for window      |
:                     :                         : (Padding\:\:kSame or         :
:                     :                         : Padding\:\:kValid)           :
| `source`            | `ComputationDataHandle` | array of type T with the     |
:                     :                         : values to scatter            :
| `init_value`        | `ComputationDataHandle` | scalar value of type T for   |
:                     :                         : the initial value of the     :
:                     :                         : output array                 :
| `scatter`           | `Computation`           | binary computation of type   |
:                     :                         : `T, T -> T`, to apply each   :
:                     :                         : scatter source element with  :
:                     :                         : its destination element      :

The figure below shows examples of using `SelectAndScatter`, with the `select`
function computing the maximal value among its parameters. Note that when the
windows overlap, as in the figure (2) below, an index of the `operand` array may
be selected multiple times by different windows. In the figure, the element of
value 9 is selected by both of the top windows (blue and red) and the binary
addition `scatter` function produces the output element of value 8 (2 + 6).

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%"
    src="https://www.tensorflow.org/images/ops_scatter_to_selected_window_element.png">
</div>

The evaluation order of the `scatter` function is arbitrary and may be
non-deterministic. Therefore, the `scatter` function should not be overly
sensitive to reassociation. See the discussion about associativity in the
context of [`Reduce`](#reduce) for more details.

## Select

See also
[`ComputationBuilder::Select`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

Constructs an output array from elements of two input arrays, based on the
values of a predicate array.

<b> `Select(pred, on_true, on_false)` </b>

Arguments  | Type                    | Semantics
---------- | ----------------------- | ------------------
`pred`     | `ComputationDataHandle` | array of type PRED
`on_true`  | `ComputationDataHandle` | array of type T
`on_false` | `ComputationDataHandle` | array of type T

The arrays `on_true` and `on_false` must have the same shape. This is also the
shape of the output array. The array `pred` must have the same dimensionality as
`on_true` and `on_false`, with the `PRED` element type.

For each element `P` of `pred`, the corresponding element of the output array is
taken from `on_true` if the value of `P` is `true`, and from `on_false` if the
value of `P` is `false`. As a restricted form of [broadcasting]
(broadcasting.md), `pred` can be a scalar of type `PRED`. In this case, the
output array is taken wholly from `on_true` if `pred` is `true`, and from
`on_false` if `pred` is `false`.

Example with non-scalar `pred`:

```
let pred: PRED[4] = {true, false, false, true};
let v1: s32[4] = {1, 2, 3, 4};
let v2: s32[4] = {100, 200, 300, 400};
==>
Select(pred, v1, v2) = s32[4]{1, 200, 300, 4};
```

Example with scalar `pred`:

```
let pred: PRED = true;
let v1: s32[4] = {1, 2, 3, 4};
let v2: s32[4] = {100, 200, 300, 400};
==>
Select(pred, v1, v2) = s32[4]{1, 2, 3, 4};
```

Selections between tuples are supported. Tuples are considered to be scalar
types for this purpose. If `on_true` and `on_false` are tuples (which must have
the same shape!) then `pred` has to be a scalar of type `PRED`.

## Slice

See also
[`ComputationBuilder::Slice`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

Slicing extracts a sub-array from the input array. The sub-array is of the same
rank as the input and contains the values inside a bounding box within the input
array where the dimensions and indices of the bounding box are given as
arguments to the slice operation.

<b> `Slice(operand, start_indices, limit_indices)` </b>

| Arguments       | Type                    | Semantics                        |
| --------------- | ----------------------- | -------------------------------- |
| `operand`       | `ComputationDataHandle` | N dimensional array of type T    |
| `start_indices` | `ArraySlice<int64>`     | List of N integers containing    |
:                 :                         : the starting indices of the      :
:                 :                         : slice for each dimension. Values :
:                 :                         : must be greater than or equal to :
:                 :                         : zero.                            :
| `limit_indices` | `ArraySlice<int64>`     | List of N integers containing    |
:                 :                         : the ending indices (exclusive)   :
:                 :                         : for the slice for each           :
:                 :                         : dimension. Each value must be    :
:                 :                         : strictly greater than the        :
:                 :                         : respective `start_indices` value :
:                 :                         : for the dimension and less than  :
:                 :                         : or equal to the size of the      :
:                 :                         : dimension.                       :

1-dimensional example:

```
let a = {0.0, 1.0, 2.0, 3.0, 4.0}
Slice(a, {2}, {4}) produces:
  {2.0, 3.0}
```

2-dimensional example:

```
let b =
 { {0.0,  1.0,  2.0},
   {3.0,  4.0,  5.0},
   {6.0,  7.0,  8.0},
   {9.0, 10.0, 11.0} }

Slice(b, {2, 1}, {4, 3}) produces:
  { { 7.0,  8.0},
    {10.0, 11.0} }
```

## DynamicSlice

See also
[`ComputationBuilder::DynamicSlice`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

DynamicSlice extracts a sub-array from the input array at dynamic
`start_indices`. The size of the slice in each dimension is passed in
`size_indices`, which specify the end point of exclusive slice intervals in each
dimension: [start, start + size). The shape of `start_indices` must be rank ==
1, with dimension size equal to the rank of `operand`.
Note: handling of out-of-bounds slice indices (generated by incorrect runtime
calculation of 'start_indices') is currently implementation-defined. Currently,
slice indices are computed modulo input dimension sizes to prevent out-of-bound
array accesses, but this behavior may change in future implementations.

<b> `DynamicSlice(operand, start_indices, size_indices)` </b>

| Arguments       | Type                    | Semantics                        |
| --------------- | ----------------------- | -------------------------------- |
| `operand`       | `ComputationDataHandle` | N dimensional array of type T    |
| `start_indices` | `ComputationDataHandle` | Rank 1 array of N integers       |
:                 :                         : containing the starting indices  :
:                 :                         : of the slice for each dimension. :
:                 :                         : Value must be greater than or    :
:                 :                         : equal to zero.                   :
| `size_indices`  | `ArraySlice<int64>`     | List of N integers containing    |
:                 :                         : the slice size for each          :
:                 :                         : dimension. Each value must be    :
:                 :                         : strictly greater than zero, and  :
:                 :                         : start + size must be less than   :
:                 :                         : or equal to the size of the      :
:                 :                         : dimension to avoid wrapping      :
:                 :                         : modulo dimension size.           :

1-dimensional example:

```
let a = {0.0, 1.0, 2.0, 3.0, 4.0}
let s = {2}

DynamicSlice(a, s, {2}) produces:
  {2.0, 3.0}
```

2-dimensional example:

```
let b =
 { {0.0,  1.0,  2.0},
   {3.0,  4.0,  5.0},
   {6.0,  7.0,  8.0},
   {9.0, 10.0, 11.0} }
let s = {2, 1}

DynamicSlice(b, s, {2, 2}) produces:
  { { 7.0,  8.0},
    {10.0, 11.0} }
```
## DynamicUpdateSlice

See also
[`ComputationBuilder::DynamicUpdateSlice`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

DynamicUpdateSlice generates a result which is the value of the input array
`operand`, with a slice `update` overwritten at `start_indices`.
The shape of `update` determines the shape of the sub-array of the result which
is updated.
The shape of `start_indices` must be rank == 1, with dimension size equal to
the rank of `operand`.
Note: handling of out-of-bounds slice indices (generated by incorrect runtime
calculation of 'start_indices') is currently implementation-defined. Currently,
slice indices are computed modulo update dimension sizes to prevent out-of-bound
array accesses, but this behavior may change in future implementations.

<b> `DynamicUpdateSlice(operand, update, start_indices)` </b>

| Arguments       | Type                    | Semantics                        |
| --------------- | ----------------------- | -------------------------------- |
| `operand`       | `ComputationDataHandle` | N dimensional array of type T    |
| `update`        | `ComputationDataHandle` | N dimensional array of type T    |
:                 :                         : containing the slice update.     :
:                 :                         : Each dimension of update shape    :
:                 :                         : must be strictly greater than    :
:                 :                         : zero, and start + update must be :
:                 :                         : less than operand size for each  :
:                 :                         : dimension to avoid generating    :
:                 :                         : out-of-bounds update indices.    :
| `start_indices` | `ComputationDataHandle` | Rank 1 array of N integers       |
:                 :                         : containing the starting indices  :
:                 :                         : of the slice for each dimension. :
:                 :                         : Value must be greater than or    :
:                 :                         : equal to zero.                   :

1-dimensional example:

```
let a = {0.0, 1.0, 2.0, 3.0, 4.0}
let u = {5.0, 6.0}
let s = {2}

DynamicUpdateSlice(a, u, s) produces:
  {0.0, 1.0, 5.0, 6.0, 4.0}
```

2-dimensional example:

```
let b =
 { {0.0,  1.0,  2.0},
   {3.0,  4.0,  5.0},
   {6.0,  7.0,  8.0},
   {9.0, 10.0, 11.0} }
let u =
 { {12.0,  13.0},
   {14.0,  15.0},
   {16.0,  17.0} }

let s = {1, 1}

DynamicUpdateSlice(b, u, s) produces:
 { {0.0,  1.0,  2.0},
   {3.0, 12.0, 13.0},
   {6.0, 14.0, 15.0},
   {9.0, 16.0, 17.0} }
```

## Sort

See also
[`ComputationBuilder::Sort`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

Sorts the elements in the operand.

<b>`Sort(operand)`</b>

Arguments | Type                    | Semantics
--------- | ----------------------- | -------------------
`operand` | `ComputationDataHandle` | The operand to sort

## Transpose

See also the @{tf.reshape} operation.

<b>`Transpose(operand)`</b>

Arguments     | Type                    | Semantics
---------     | ----------------------- | -------------------------
`operand`     | `ComputationDataHandle` | The operand to transpose.
`permutation` | `ArraySlice<int64>`     | How to permute the dimensions.


Permutes the operand dimensions with the given permutation, so
`∀ i . 0 ≤ i < rank ⇒ input_dimensions[permutation[i]] = output_dimensions[i]`.

This is the same as Reshape(operand, permutation,
                            Permute(permutation, operand.shape.dimensions)).

## Tuple

See also
[`ComputationBuilder::Tuple`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

A tuple containing a variable number of data handles, each of which has its own
shape.

This is analogous to `std::tuple` in C++. Conceptually:

```
let v: f32[10] = f32[10]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
let s: s32 = 5;
let t: (f32[10], s32) = tuple(v, s);
```

Tuples can be deconstructed (accessed) via the [`GetTupleElement`]
(#gettupleelement) operation.

## While

See also
[`ComputationBuilder::While`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

<b> `While(condition, body, init)` </b>

| Arguments   | Type          | Semantics                                      |
| ----------- | ------------- | ---------------------------------------------- |
| `condition` | `Computation` | Computation of type `T -> PRED` which defines  |
:             :               : the termination condition of the loop.         :
| `body`      | `Computation` | Computation of type `T -> T` which defines the |
:             :               : body of the loop.                              :
| `init`      | `T`           | Initial value for the parameter of `condition` |
:             :               : and `body`.                                    :

Sequentially executes the `body` until the `condition` fails. This is similar to
a typical while loop in many other languages except for the differences and
restrictions listed below.

*   A `While` node returns a value of type `T`, which is the result from the
    last execution of the `body`.
*   The shape of the type `T` is statically determined and must be the same
    across all iterations.
*   `While` nodes are not allowed to be nested. (This restriction may be lifted
    in the future on some targets.)

The T parameters of the computations are initialized with the `init` value in
the first iteration and are automatically updated to the new result from `body`
in each subsequent iteration.

One main use case of the `While` node is to implement the repeated execution of
training in neural networks. Simplified pseudocode is shown below with a graph
that represents the computation. The code can be found in
[`while_test.cc`](https://www.tensorflow.org/code/tensorflow/compiler/xla/tests/while_test.cc).
The type `T` in this example is a `Tuple` consisting of an `int32` for the
iteration count and a `vector[10]` for the accumulator. For 1000 iterations, the
loop keeps adding a constant vector to the accumulator.

```
// Pseudocode for the computation.
init = {0, zero_vector[10]} // Tuple of int32 and float[10].
result = init;
while (result(0) < 1000) {
  iteration = result(0) + 1;
  new_vector = result(1) + constant_vector[10];
  result = {iteration, new_vector};
}
```

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/ops_while.png">
</div>

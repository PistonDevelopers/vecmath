#![crate_name = "vecmath"]
#![deny(missing_docs)]

//! A simple and generic library for vector math.
//!
//! Notice that row major is mathematical standard,
//! while OpenGL uses column major format.
//! This library supports both formats, prefixing functions with 'row_' or 'col_'.
//!
//! For row major affine transforms, use `Matrix2x3` (2D) and `Matrix3x4` (3D).
//! For column major affine transforms, use `Matrix3x2` (2D) and `Matrix4x3` (3D).
//!
//! If you are using `Matrix3` or `Matrix4`, 
//! then you need to pick either row or column major.
//!
//! Notice that there are two kinds of transforms: Positions and vectors.
//! The vector transforms ignores the translate component.
//! For example, `row_mat2x3_transform_pos2` transforms a position.
//! `row_mat2x3_transform_vec2` transforms a vector.

extern crate num;

use num::{Float, NumCast, ToPrimitive};

pub mod consts;

/// A 2D vector.
pub type Vector2<T> = [T; 2];

/// A 3D vector.
pub type Vector3<T> = [T; 3];

/// A 4D vector.
pub type Vector4<T> = [T; 4];

/// A 2x3 matrix.
///
/// To multiply two matrices use `row_mat2x3_mul`.
pub type Matrix2x3<T> = [[T; 3]; 2];

/// A 3x2 matrix.
///
/// To multiply two matrices use `col_mat3x2_mul`.
pub type Matrix3x2<T> = [[T; 2]; 3];

/// A 3x3 matrix.
///
/// To multiply two matrices use `row_mat3_mul` or `col_mat3_mul`.
pub type Matrix3<T> = [[T; 3]; 3];

/// A 3x4 matrix.
///
/// To multiply two matrices use `row_mat3x4_mul`.
pub type Matrix3x4<T> = [[T; 4]; 3];

/// A 4x3 matrix.
///
/// To multiply two matrices use `col_mat4x3_mul`.
///
/// This format can also store vertices of a quad.
pub type Matrix4x3<T> = [[T; 3]; 4];

/// A 4x4 matrix.
///
/// To multiply two matrices use `row_mat4_mul` or `col_mat4_mul`.
pub type Matrix4<T> = [[T; 4]; 4];

/// Computes column vector in column matrix product.
///
/// The semantics of the order is the same as for row matrices.
#[inline(always)]
pub fn col_mat3x2_mul_col<T: Float + Copy>(
    a: Matrix3x2<T>,
    b: Matrix3x2<T>,
    i: usize
) -> Vector2<T> {
    [
        vec3_dot_vec2(col_mat3x2_row(a, 0), b[i]),
        vec3_dot_vec2(col_mat3x2_row(a, 1), b[i])
    ]
}

/// Computes column vector in column matrix product.
///
/// The semantics of the order is the same as for row matrices.
#[inline(always)]
pub fn col_mat3_mul_col<T: Float + Copy>(
    a: Matrix3<T>,
    b: Matrix3<T>,
    i: usize
) -> Vector3<T> {
    [
        vec3_dot(col_mat3_row(a, 0), b[i]),
        vec3_dot(col_mat3_row(a, 1), b[i]),
        vec3_dot(col_mat3_row(a, 2), b[i])
    ]
}

/// Computes column vector in column matrix product.
///
/// The semantics of the order is the same as for row matrices.
#[inline(always)]
pub fn col_mat4x3_mul_col<T: Float + Copy>(
    a: Matrix4x3<T>,
    b: Matrix4x3<T>,
    i: usize
) -> Vector3<T> {
    [
        vec4_dot_vec3(col_mat4x3_row(a, 0), b[i]),
        vec4_dot_vec3(col_mat4x3_row(a, 1), b[i]),
        vec4_dot_vec3(col_mat4x3_row(a, 2), b[i])
    ]
}

/// Computes column vector in column matrix product.
///
/// The semantics of the order is the same as for row matrices.
#[inline(always)]
pub fn col_mat4_mul_col<T: Float + Copy>(
    a: Matrix4<T>, 
    b: Matrix4<T>,
    i: usize
) -> Vector4<T> {
    [
        vec4_dot(col_mat4_row(a, 0), b[i]),
        vec4_dot(col_mat4_row(a, 1), b[i]),
        vec4_dot(col_mat4_row(a, 2), b[i]),
        vec4_dot(col_mat4_row(a, 3), b[i])
    ]
}

/// Computes row vector in row matrix product.
#[inline(always)]
pub fn row_mat2x3_mul_row<T: Float + Copy>(
    a: Matrix2x3<T>, 
    b: Matrix2x3<T>,
    i: usize
) -> Vector3<T> {
    [
        vec3_dot_vec2(a[i], row_mat2x3_col(b, 0)),
        vec3_dot_vec2(a[i], row_mat2x3_col(b, 1)),
        vec3_dot_pos2(a[i], row_mat2x3_col(b, 2))
    ]
}

/// Computes row vector in row matrix product.
#[inline(always)]
pub fn row_mat3_mul_row<T: Float + Copy>(
    a: Matrix3<T>, 
    b: Matrix3<T>,
    i: usize
) -> Vector3<T> {
    [
        vec3_dot(a[i], row_mat3_col(b, 0)),
        vec3_dot(a[i], row_mat3_col(b, 1)),
        vec3_dot(a[i], row_mat3_col(b, 2)),
    ]
}

/// Computes row vector in row matrix product.
#[inline(always)]
pub fn row_mat3x4_mul_row<T: Float + Copy>(
    a: Matrix3x4<T>, 
    b: Matrix3x4<T>,
    i: usize
) -> Vector4<T> {
    [
        vec4_dot_vec3(a[i], row_mat3x4_col(b, 0)),
        vec4_dot_vec3(a[i], row_mat3x4_col(b, 1)),
        vec4_dot_vec3(a[i], row_mat3x4_col(b, 2)),
        vec4_dot_pos3(a[i], row_mat3x4_col(b, 3))
    ]
}

/// Computes row vector in row matrix product.
#[inline(always)]
pub fn row_mat4_mul_row<T: Float + Copy>(
    a: Matrix4<T>, 
    b: Matrix4<T>,
    i: usize
) -> Vector4<T> {
    [
        vec4_dot(a[i], row_mat4_col(b, 0)),
        vec4_dot(a[i], row_mat4_col(b, 1)),
        vec4_dot(a[i], row_mat4_col(b, 2)),
        vec4_dot(a[i], row_mat4_col(b, 3))
    ]
}

/// Multiplies two matrices.
#[inline(always)]
pub fn col_mat3x2_mul<T: Float + Copy>(
    a: Matrix3x2<T>,
    b: Matrix3x2<T>
) -> Matrix3x2<T> {
    [
        col_mat3x2_mul_col(a, b, 0),
        col_mat3x2_mul_col(a, b, 1),
        col_mat3x2_mul_col(a, b, 2)
    ]
}

/// Multiplies two matrices.
#[inline(always)]
pub fn col_mat3_mul<T: Float + Copy>(
    a: Matrix3<T>,
    b: Matrix3<T>
) -> Matrix3<T> {
    [
        col_mat3_mul_col(a, b, 0),
        col_mat3_mul_col(a, b, 1),
        col_mat3_mul_col(a, b, 2)
    ]
}

/// Multiplies two matrices.
#[inline(always)]
pub fn col_mat4x3_mul<T: Float + Copy>(
    a: Matrix4x3<T>,
    b: Matrix4x3<T>
) -> Matrix4x3<T> {
    [
        col_mat4x3_mul_col(a, b, 0),
        col_mat4x3_mul_col(a, b, 1),
        col_mat4x3_mul_col(a, b, 2),
        col_mat4x3_mul_col(a, b, 3)
    ]
}

/// Multiplies two matrices.
#[inline(always)]
pub fn col_mat4_mul<T: Float + Copy>(
    a: Matrix4<T>,
    b: Matrix4<T>
) -> Matrix4<T> {
    [
        col_mat4_mul_col(a, b, 0),
        col_mat4_mul_col(a, b, 1),
        col_mat4_mul_col(a, b, 2),
        col_mat4_mul_col(a, b, 3)
    ]
}

/// Multiplies two matrices.
#[inline(always)]
pub fn row_mat2x3_mul<T: Float + Copy>(
    a: Matrix2x3<T>, 
    b: Matrix2x3<T>
) -> Matrix2x3<T> {
    [
        row_mat2x3_mul_row(a, b, 0),
        row_mat2x3_mul_row(a, b, 1),
    ]
}

/// Multiplies two matrices.
#[inline(always)]
pub fn row_mat3_mul<T: Float + Copy>(
    a: Matrix3<T>,
    b: Matrix3<T>
) -> Matrix3<T> {
    [
        row_mat3_mul_row(a, b, 0),
        row_mat3_mul_row(a, b, 1),
        row_mat3_mul_row(a, b, 2)
    ]
}

/// Multiplies two matrices.
#[inline(always)]
pub fn row_mat3x4_mul<T: Float + Copy>(
    a: Matrix3x4<T>, 
    b: Matrix3x4<T>
) -> Matrix3x4<T> {
    [
        row_mat3x4_mul_row(a, b, 0),
        row_mat3x4_mul_row(a, b, 1),
        row_mat3x4_mul_row(a, b, 2)
    ]
}

/// Multiplies two matrices.
#[inline(always)]
pub fn row_mat4_mul<T: Float + Copy>(
    a: Matrix4<T>,
    b: Matrix4<T>
) -> Matrix4<T> {
    [
        row_mat4_mul_row(a, b, 0),
        row_mat4_mul_row(a, b, 1),
        row_mat4_mul_row(a, b, 2),
        row_mat4_mul_row(a, b, 3)
    ]
}

#[test]
fn test_row_mat2x3_mul() {
    let a: Matrix2x3<f64> = mat2x3_id();
    let b = a;
    let _ = row_mat2x3_mul(a, b);
}

#[test]
fn test_row_mat3x4_mul() {
    let a: Matrix3x4<f64> = mat3x4_id();
    let b = a;
    let _ = row_mat3x4_mul(a, b);
}

/// Constructs identity matrix.
#[inline(always)]
pub fn mat2x3_id<T: Float + Copy>() -> Matrix2x3<T> {
    let one = T::one();
    let zero = T::zero();
    [
        [one, zero, zero],
        [zero, one, zero]
    ]
}

/// Constructs identity matrix.
#[inline(always)]
pub fn mat3x2_id<T: Float + Copy>() -> Matrix3x2<T> {
    let one = T::one();
    let zero = T::zero();
    [
        [one, zero],
        [zero, one],
        [zero, zero]
    ]
}

/// Constructs identity matrix.
#[inline(always)]
pub fn mat3_id<T: Float + Copy>() -> Matrix3<T> {
    let one = T::one();
    let zero = T::zero();
    [
        [one, zero, zero],
        [zero, one, zero],
        [zero, zero, one]
    ]
}

/// Constructs identity matrix.
#[inline(always)]
pub fn mat3x4_id<T: Float + Copy>() -> Matrix3x4<T> {
    let one = T::one();
    let zero = T::zero();
    [
        [one, zero, zero, zero],
        [zero, one, zero, zero],
        [zero, zero, one, zero]
    ]
}

/// Constructs identity matrix.
#[inline(always)]
pub fn mat4x3_id<T: Float + Copy>() -> Matrix4x3<T> {
    let one = T::one();
    let zero = T::zero();
    [
        [one, zero, zero],
        [zero, one, zero],
        [zero, zero, one],
        [zero, zero, zero]
    ]
}

/// Constructs identity matrix.
#[inline(always)]
pub fn mat4_id<T: Float + Copy>() -> Matrix4<T> {
    let one = T::one();
    let zero = T::zero();
    [
        [one, zero, zero, zero],
        [zero, one, zero, zero],
        [zero, zero, one, zero],
        [zero, zero, zero, one]
    ]
}

/// Converts to another vector type.
#[inline(always)]
pub fn vec2_cast<T: ToPrimitive + Copy, U: NumCast>(
    a: Vector2<T>
) -> Vector2<U> {
    [
        NumCast::from(a[0]).unwrap(),
        NumCast::from(a[1]).unwrap() 
    ]
}

/// Converts to another vector type.
#[inline(always)]
pub fn vec3_cast<T: ToPrimitive + Copy, U: NumCast>(
    a: Vector3<T>
) -> Vector3<U> {
    [
        NumCast::from(a[0]).unwrap(),
        NumCast::from(a[1]).unwrap(), 
        NumCast::from(a[2]).unwrap()
    ]
}

/// Converts to another vector type.
#[inline(always)]
pub fn vec4_cast<T: ToPrimitive + Copy, U: NumCast>(
    a: Vector4<T>
) -> Vector4<U> {
    [
        NumCast::from(a[0]).unwrap(),
        NumCast::from(a[1]).unwrap(), 
        NumCast::from(a[2]).unwrap(),
        NumCast::from(a[3]).unwrap()
    ]
}

/// Converts to another matrix type.
#[inline(always)]
pub fn mat2x3_cast<T: ToPrimitive + Copy, U: NumCast>(
    mat: Matrix2x3<T>
) -> Matrix2x3<U> {
    [
        vec3_cast(mat[0]),
        vec3_cast(mat[1])
    ]
}

/// Converts to another matrix type.
#[inline(always)]
pub fn mat3x2_cast<T: ToPrimitive + Copy, U: NumCast>(
    mat: Matrix3x2<T>
) -> Matrix3x2<U> {
    [
        vec2_cast(mat[0]),
        vec2_cast(mat[1]),
        vec2_cast(mat[2])
    ]
}

/// Converts to another matrix type.
#[inline(always)]
pub fn mat3_cast<T: ToPrimitive + Copy, U: NumCast>(
    mat: Matrix3<T>
) -> Matrix3<U> {
    [
        vec3_cast(mat[0]),
        vec3_cast(mat[1]),
        vec3_cast(mat[2])
    ]
}

/// Converts to another matrix type.
#[inline(always)]
pub fn mat3x4_cast<T: ToPrimitive + Copy, U: NumCast>(
    m: Matrix3x4<T>
) -> Matrix3x4<U> {
    [
        vec4_cast(m[0]),
        vec4_cast(m[1]),
        vec4_cast(m[2])
    ]
}

/// Converts to another matrix type.
#[inline(always)]
pub fn mat4x3_cast<T: ToPrimitive + Copy, U: NumCast>(
    m: Matrix4x3<T>
) -> Matrix4x3<U> {
    [
        vec3_cast(m[0]),
        vec3_cast(m[1]),
        vec3_cast(m[2]),
        vec3_cast(m[3])
    ]
}

/// Converts to another matrix type.
#[inline(always)]
pub fn mat4_cast<T: ToPrimitive + Copy, U: NumCast>(
    m: Matrix4<T>
) -> Matrix4<U> {
    [
        vec4_cast(m[0]),
        vec4_cast(m[1]),
        vec4_cast(m[2]),
        vec4_cast(m[3])
    ]
}

/// Subtracts 'b' from 'a'.
#[inline(always)]
pub fn vec2_sub<T: Float>(a: Vector2<T>, b: Vector2<T>) -> Vector2<T> {
    [
        a[0] - b[0],
        a[1] - b[1],
    ]
}

/// Subtracts 'b' from 'a'.
#[inline(always)]
pub fn vec3_sub<T: Float>(a: Vector3<T>, b: Vector3<T>) -> Vector3<T> {
    [
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2],
    ]
}

/// Subtracts 'b' from 'a'.
#[inline(always)]
pub fn vec4_sub<T: Float>(a: Vector4<T>, b: Vector4<T>) -> Vector4<T> {
    [
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2],
        a[3] - b[3]
    ]
}

/// Subtracts 'b' from 'a'.
#[inline(always)]
pub fn mat2x3_sub<T: Float + Copy>(a: Matrix2x3<T>, b: Matrix2x3<T>) -> Matrix2x3<T> {
    [
        vec3_sub(a[0], b[0]),
        vec3_sub(a[1], b[1])
    ]
}

/// Subtracts 'b' from 'a'.
#[inline(always)]
pub fn mat3x2_sub<T: Float + Copy>(a: Matrix3x2<T>, b: Matrix3x2<T>) -> Matrix3x2<T> {
    [
        vec2_sub(a[0], b[0]),
        vec2_sub(a[1], b[1]),
        vec2_sub(a[2], b[2])
    ]
}

/// Subtracts 'b' from 'a'.
#[inline(always)]
pub fn mat3_sub<T: Float + Copy>(a: Matrix3<T>, b: Matrix3<T>) -> Matrix3<T> {
    [
        vec3_sub(a[0], b[0]),
        vec3_sub(a[1], b[1]),
        vec3_sub(a[2], b[2])
    ]
}

/// Subtracts 'b' from 'a'.
#[inline(always)]
pub fn mat3x4_sub<T: Float + Copy>(a: Matrix3x4<T>, b: Matrix3x4<T>) -> Matrix3x4<T> {
    [
        vec4_sub(a[0], b[0]),
        vec4_sub(a[1], b[1]),
        vec4_sub(a[2], b[2])
    ]
}

/// Subtracts 'b' from 'a'.
#[inline(always)]
pub fn mat4x3_sub<T: Float + Copy>(a: Matrix4x3<T>, b: Matrix4x3<T>) -> Matrix4x3<T> {
    [
        vec3_sub(a[0], b[0]),
        vec3_sub(a[1], b[1]),
        vec3_sub(a[2], b[2]),
        vec3_sub(a[3], b[3])
    ]
}

/// Subtracts 'b' from 'a'.
#[inline(always)]
pub fn mat4_sub<T: Float + Copy>(a: Matrix4<T>, b: Matrix4<T>) -> Matrix4<T> {
    [
        vec4_sub(a[0], b[0]),
        vec4_sub(a[1], b[1]),
        vec4_sub(a[2], b[2]),
        vec4_sub(a[3], b[3])
    ]
}

/// Adds two vectors.
#[inline(always)]
pub fn vec2_add<T: Float>(a: Vector2<T>, b: Vector2<T>) -> Vector2<T> {
    [
        a[0] + b[0],
        a[1] + b[1],
    ]
}

/// Adds two vectors.
#[inline(always)]
pub fn vec3_add<T: Float>(a: Vector3<T>, b: Vector3<T>) -> Vector3<T> {
    [
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2]
    ]
}

/// Adds two vectors.
#[inline(always)]
pub fn vec4_add<T: Float>(a: Vector4<T>, b: Vector4<T>) -> Vector4<T> {
    [
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3]
    ]
}

/// Adds two matrices.
#[inline(always)]
pub fn mat2x3_add<T: Float + Copy>(a: Matrix2x3<T>, b: Matrix2x3<T>) -> Matrix2x3<T> {
    [
        vec3_add(a[0], b[0]),
        vec3_add(a[1], b[1])
    ]
}

/// Adds two matrices.
#[inline(always)]
pub fn mat3x2_add<T: Float + Copy>(a: Matrix3x2<T>, b: Matrix3x2<T>) -> Matrix3x2<T> {
    [
        vec2_add(a[0], b[0]),
        vec2_add(a[1], b[1]),
        vec2_add(a[2], b[2])
    ]
}

/// Adds two matrices.
#[inline(always)]
pub fn mat3_add<T: Float + Copy>(a: Matrix3<T>, b: Matrix3<T>) -> Matrix3<T> {
    [
        vec3_add(a[0], b[0]),
        vec3_add(a[1], b[1]),
        vec3_add(a[2], b[2])
    ]
}

/// Adds two matrices.
#[inline(always)]
pub fn mat3x4_add<T: Float + Copy>(a: Matrix3x4<T>, b: Matrix3x4<T>) -> Matrix3x4<T> {
    [
        vec4_add(a[0], b[0]),
        vec4_add(a[1], b[1]),
        vec4_add(a[2], b[2])
    ]
}

/// Adds two matrices.
#[inline(always)]
pub fn mat4x3_add<T: Float + Copy>(a: Matrix4x3<T>, b: Matrix4x3<T>) -> Matrix4x3<T> {
    [
        vec3_add(a[0], b[0]),
        vec3_add(a[1], b[1]),
        vec3_add(a[2], b[2]),
        vec3_add(a[3], b[3])
    ]
}

/// Adds two matrices.
#[inline(always)]
pub fn mat4_add<T: Float + Copy>(a: Matrix4<T>, b: Matrix4<T>) -> Matrix4<T> {
    [
        vec4_add(a[0], b[0]),
        vec4_add(a[1], b[1]),
        vec4_add(a[2], b[2]),
        vec4_add(a[3], b[3])
    ]
}

/// Multiplies two vectors component wise.
#[inline(always)]
pub fn vec2_mul<T: Float>(a: Vector2<T>, b: Vector2<T>) -> Vector2<T> {
    [a[0] * b[0], a[1] * b[1]]
}

/// Multiplies two vectors component wise.
#[inline(always)]
pub fn vec3_mul<T: Float>(a: Vector3<T>, b: Vector3<T>) -> Vector3<T> {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

/// Multiplies two vectors component wise.
#[inline(always)]
pub fn vec4_mul<T: Float>(a: Vector4<T>, b: Vector4<T>) -> Vector4<T> {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
}

/// Computes the dot product.
#[inline(always)]
pub fn vec2_dot<T: Float>(a: Vector2<T>, b: Vector2<T>) -> T {
    a[0] * b[0] + a[1] * b[1]
}

/// Computes the dot product.
#[inline(always)]
pub fn vec3_dot<T: Float>(a: Vector3<T>, b: Vector3<T>) -> T {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Computes the dot product.
#[inline(always)]
pub fn vec4_dot<T: Float>(a: Vector4<T>, b: Vector4<T>) -> T {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// Computes the square length of a vector.
#[inline(always)]
pub fn vec2_square_len<T: Float>(a: Vector2<T>) -> T {
    a[0] * a[0] + a[1] * a[1]
}

/// Computes the square length of a vector.
#[inline(always)]
pub fn vec3_square_len<T: Float>(a: Vector3<T>) -> T {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
}

/// Computes the square length of a vector.
#[inline(always)]
pub fn vec4_square_len<T: Float>(a: Vector4<T>) -> T {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3]
}

/// Computes the cross product.
#[inline(always)]
pub fn vec2_cross<T: Float>(a: Vector2<T>, b: Vector2<T>) -> T {
    a[0] * b[1] - a[1] * b[0]
}

/// Computes the cross product.
#[inline(always)]
pub fn vec3_cross<T: Float>(a: Vector3<T>, b: Vector3<T>) -> Vector3<T> {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]
}

/// Multiplies the vector with a scalar.
#[inline(always)]
pub fn vec2_scale<T: Float>(a: Vector2<T>, b: T) -> Vector2<T> {
    [
        a[0] * b,
        a[1] * b
    ]
}

/// Multiplies the vector with a scalar.
#[inline(always)]
pub fn vec3_scale<T: Float>(a: Vector3<T>, b: T) -> Vector3<T> {
    [
        a[0] * b,
        a[1] * b,
        a[2] * b
    ]
}

/// Multiplies the vector with a scalar.
#[inline(always)]
pub fn vec4_scale<T: Float>(a: Vector4<T>, b: T) -> Vector4<T> {
    [
        a[0] * b,
        a[1] * b,
        a[2] * b,
        a[3] * b
    ]
}

/// Negates the vector.
#[inline(always)]
pub fn vec2_neg<T: Float>(a: Vector2<T>) -> Vector2<T> {
    [-a[0], -a[1]]
}

/// Negates the vector.
#[inline(always)]
pub fn vec3_neg<T: Float>(a: Vector3<T>) -> Vector3<T> {
    [-a[0], -a[1], -a[2]]
}

/// Negates the vector.
#[inline(always)]
pub fn vec4_neg<T: Float>(a: Vector4<T>) -> Vector4<T> {
    [-a[0], -a[1], -a[2], -a[3]]
}

/// Computes the length of vector.
#[inline(always)]
pub fn vec2_len<T: Float>(a: Vector2<T>) -> T {
    vec2_square_len(a).sqrt()
}

/// Computes the length of vector.
#[inline(always)]
pub fn vec3_len<T: Float>(a: Vector3<T>) -> T {
    vec3_square_len(a).sqrt()
}

/// Computes the length of vector.
#[inline(always)]
pub fn vec4_len<T: Float>(a: Vector4<T>) -> T {
    vec4_square_len(a).sqrt()
}

/// Computes the inverse length of a vector.
#[inline(always)]
pub fn vec2_inv_len<T: Float>(a: Vector2<T>) -> T {
    let one = T::one();
    one / vec2_len(a)
}

/// Computes the inverse length of a vector.
#[inline(always)]
pub fn vec3_inv_len<T: Float>(a: Vector3<T>) -> T {
    let one = T::one();
    one / vec3_len(a)
}

/// Computes the inverse length of a vector.
#[inline(always)]
pub fn vec4_inv_len<T: Float>(a: Vector4<T>) -> T {
    let one = T::one();
    one / vec4_len(a)
}

/// Computes the normalized.
#[inline(always)]
pub fn vec2_normalized<T: Float>(a: Vector2<T>) -> Vector2<T> {
    vec2_scale(a, vec2_inv_len(a))
}

/// Computes the normalized.
#[inline(always)]
pub fn vec3_normalized<T: Float>(a: Vector3<T>) -> Vector3<T> {
    vec3_scale(a, vec3_inv_len(a))
}

/// Computes the normalized.
#[inline(always)]
pub fn vec4_normalized<T: Float>(a: Vector4<T>) -> Vector4<T> {
    vec4_scale(a, vec4_inv_len(a))
}

/// Computes the normalized difference between two vectors.
///
/// This is often used to get direction from 'b' to 'a'.
#[inline(always)]
pub fn vec2_normalized_sub<T: Float>(
    a: Vector2<T>, 
    b: Vector2<T>
) -> Vector2<T> {
    vec2_normalized(vec2_sub(a, b))
}

/// Computes the normalized difference between two vectors.
///
/// This is often used to get direction from 'b' to 'a'.
#[inline(always)]
pub fn vec3_normalized_sub<T: Float>(
    a: Vector3<T>, 
    b: Vector3<T>
) -> Vector3<T> {
    vec3_normalized(vec3_sub(a, b))
}

/// Computes the normalized difference between two vectors.
///
/// This is often used to get direction from 'b' to 'a'.
#[inline(always)]
pub fn vec4_normalized_sub<T: Float>(
    a: Vector4<T>, 
    b: Vector4<T>
) -> Vector4<T> {
    vec4_normalized(vec4_sub(a, b))
}

/// Computes transformed vector component.
///
/// This is used when transforming vectors through matrices.
#[inline(always)]
pub fn vec3_dot_vec2<T: Float>(a: Vector3<T>, b: Vector2<T>) -> T {
    a[0] * b[0] + a[1] * b[1]
}

/// Computes transformed vector component.
///
/// This is used when transforming vectors through matrices.
#[inline(always)]
pub fn vec4_dot_vec3<T: Float>(a: Vector4<T>, b: Vector3<T>) -> T {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Computes transformed position component.
///
/// This is used when transforming points through matrices.
#[inline(always)]
pub fn vec3_dot_pos2<T: Float + Copy>(a: Vector3<T>, b: Vector2<T>) -> T {
    vec3_dot_vec2(a, b) + a[2]
}

/// Computes transformed position component.
///
/// This is used when transforming points through matrices.
#[inline(always)]
pub fn vec4_dot_pos3<T: Float + Copy>(a: Vector4<T>, b: Vector3<T>) -> T {
    vec4_dot_vec3(a, b) + a[3]
}

/// Returns a column vector of a row matrix.
#[inline(always)]
pub fn row_mat2x3_col<T: Copy>(mat: Matrix2x3<T>, i: usize) -> Vector2<T> {
    [mat[0][i], mat[1][i]]
}

/// Returns a row vector of a column matrix.
#[inline(always)]
pub fn col_mat2x3_row<T: Copy>(mat: Matrix2x3<T>, i: usize) -> Vector2<T> {
    row_mat2x3_col(mat, i)
}

/// Returns a column vector of a row matrix.
#[inline(always)]
pub fn row_mat3x2_col<T: Copy>(a: Matrix3x2<T>, i: usize) -> Vector3<T> {
    [a[0][i], a[1][i], a[2][i]]
}

/// Returns a row vector of a column matrix.
#[inline(always)]
pub fn col_mat3x2_row<T: Copy>(a: Matrix3x2<T>, i: usize) -> Vector3<T> {
    row_mat3x2_col(a, i)
}

/// Returns a column vector of a row matrix.
#[inline(always)]
pub fn row_mat3_col<T: Copy>(a: Matrix3<T>, i: usize) -> Vector3<T> {
    [a[0][i], a[1][i], a[2][i]]
}

/// Returns a row vector of a column matrix.
#[inline(always)]
pub fn col_mat3_row<T: Copy>(a: Matrix3<T>, i: usize) -> Vector3<T> {
    row_mat3_col(a, i)
}

/// Returns a column vector of a row matrix.
#[inline(always)]
pub fn row_mat3x4_col<T: Copy>(mat: Matrix3x4<T>, i: usize) -> Vector3<T> {
    [mat[0][i], mat[1][i], mat[2][i]]
}

/// Returns a row vector of a column matrix.
#[inline(always)]
pub fn col_mat3x4_row<T: Copy>(mat: Matrix3x4<T>, i: usize) -> Vector3<T> {
    row_mat3x4_col(mat, i)
}

/// Returns a column vector of a row matrix.
#[inline(always)]
pub fn row_mat4x3_col<T: Copy>(a: Matrix4x3<T>, i: usize) -> Vector4<T> {
    [a[0][i], a[1][i], a[2][i], a[3][i]]
}

/// Returns a column vector of a row matrix.
#[inline(always)]
pub fn col_mat4x3_row<T: Copy>(a: Matrix4x3<T>, i: usize) -> Vector4<T> {
    row_mat4x3_col(a, i)
}

/// Returns a column vector of a row matrix.
#[inline(always)]
pub fn row_mat4_col<T: Copy>(a: Matrix4<T>, i: usize) -> Vector4<T> {
    [a[0][i], a[1][i], a[2][i], a[3][i]]
}

/// Returns a row vector of a column matrix.
#[inline(always)]
pub fn col_mat4_row<T: Copy>(a: Matrix4<T>, i: usize) -> Vector4<T> {
    row_mat4_col(a, i)
}

/// Constructs the transpose of a matrix.
#[inline(always)]
pub fn mat2x3_transposed<T: Copy>(a: Matrix2x3<T>) -> Matrix3x2<T> {
    [
        row_mat2x3_col(a, 0),
        row_mat2x3_col(a, 1),
        row_mat2x3_col(a, 2)
    ]
}

/// Constructs the transpose of a matrix.
#[inline(always)]
pub fn mat3x2_transposed<T: Copy>(a: Matrix3x2<T>) -> Matrix2x3<T> {
    [
        row_mat3x2_col(a, 0),
        row_mat3x2_col(a, 1)
    ]
}

/// Constructs the transpose of a matrix.
#[inline(always)]
pub fn mat3_transposed<T: Copy>(a: Matrix3<T>) -> Matrix3<T> {
    [
        row_mat3_col(a, 0),
        row_mat3_col(a, 1),
        row_mat3_col(a, 2)
    ]
}

/// Constructs the transpose of a matrix.
#[inline(always)]
pub fn mat3x4_transposed<T: Copy>(a: Matrix3x4<T>) -> Matrix4x3<T> {
    [
        row_mat3x4_col(a, 0),
        row_mat3x4_col(a, 1),
        row_mat3x4_col(a, 2),
        row_mat3x4_col(a, 3)
    ]
}

/// Constructs the transpose of a matrix.
#[inline(always)]
pub fn mat4x3_transposed<T: Copy>(a: Matrix4x3<T>) -> Matrix3x4<T> {
    [
        row_mat4x3_col(a, 0),
        row_mat4x3_col(a, 1),
        row_mat4x3_col(a, 2)
    ]
}

/// Constructs the transpose of a matrix.
#[inline(always)]
pub fn mat4_transposed<T: Copy>(a: Matrix4<T>) -> Matrix4<T> {
    [
        row_mat4_col(a, 0),
        row_mat4_col(a, 1),
        row_mat4_col(a, 2),
        row_mat4_col(a, 3)
    ]
}

/// Transforms a 3D vector through a matrix.
#[inline(always)]
pub fn col_mat3_transform<T: Float + Copy>(
    mat: Matrix3<T>,
    a: Vector3<T>
) -> Vector3<T> {
    [
        vec3_dot(col_mat3_row(mat, 0), a),
        vec3_dot(col_mat3_row(mat, 1), a),
        vec3_dot(col_mat3_row(mat, 2), a)
    ]
}

/// Transforms a 4D vector through a matrix.
#[inline(always)]
pub fn col_mat4_transform<T: Float + Copy>(
    mat: Matrix4<T>,
    a: Vector4<T>
) -> Vector4<T> {
    [
        vec4_dot(col_mat4_row(mat, 0), a),
        vec4_dot(col_mat4_row(mat, 1), a),
        vec4_dot(col_mat4_row(mat, 2), a),
        vec4_dot(col_mat4_row(mat, 3), a)
    ]
}

/// Transforms a 3D vector through a matrix.
#[inline(always)]
pub fn row_mat3_transform<T: Float + Copy>(
    mat: Matrix3<T>,
    a: Vector3<T>
) -> Vector3<T> {
    [
        vec3_dot(mat[0], a),
        vec3_dot(mat[1], a),
        vec3_dot(mat[2], a)
    ]
}

/// Transforms a 4D vector through a matrix.
#[inline(always)]
pub fn row_mat4_transform<T: Float + Copy>(
    mat: Matrix4<T>,
    a: Vector4<T>
) -> Vector4<T> {
    [
        vec4_dot(mat[0], a),
        vec4_dot(mat[1], a),
        vec4_dot(mat[2], a),
        vec4_dot(mat[3], a)
    ]
}

/// Transforms a 2D position through matrix.
#[inline(always)]
pub fn row_mat2x3_transform_pos2<T: Float + Copy>(
    mat: Matrix2x3<T>, 
    a: Vector2<T>
) -> Vector2<T> {
    [
        vec3_dot_pos2(mat[0], a),
        vec3_dot_pos2(mat[1], a)
    ]
}

/// Transforms a 2D position through matrix.
#[inline(always)]
pub fn col_mat3x2_transform_pos2<T: Float + Copy>(
    mat: Matrix3x2<T>,
    a: Vector2<T>
) -> Vector2<T> {
    [
        vec3_dot_pos2(col_mat3x2_row(mat, 0), a),
        vec3_dot_pos2(col_mat3x2_row(mat, 1), a)
    ]
}

/// Transforms a 2D position through row matrix.
#[inline(always)]
pub fn row_mat3_transform_pos2<T: Float + Copy>(
    mat: Matrix3<T>,
    a: Vector2<T>
) -> Vector2<T> {
    [
        vec3_dot_pos2(mat[0], a),
        vec3_dot_pos2(mat[1], a)
    ]
}

/// Transforms a 2D position through column matrix.
#[inline(always)]
pub fn col_mat3_transform_pos2<T: Float + Copy>(
    mat: Matrix3<T>,
    a: Vector2<T>
) -> Vector2<T> {
    [
        vec3_dot_pos2(col_mat3_row(mat, 0), a),
        vec3_dot_pos2(col_mat3_row(mat, 1), a)
    ]
}

/// Transforms a 3D position through matrix.
#[inline(always)]
pub fn row_mat3x4_transform_pos3<T: Float + Copy>(
    mat: Matrix3x4<T>, 
    a: Vector3<T>
) -> Vector3<T> {
    [
        vec4_dot_pos3(mat[0], a),
        vec4_dot_pos3(mat[1], a),
        vec4_dot_pos3(mat[2], a),
    ]
}

/// Transforms a 3D position through matrix.
#[inline(always)]
pub fn col_mat4x3_transform_pos3<T: Float + Copy>(
    mat: Matrix4x3<T>,
    a: Vector3<T>
) -> Vector3<T> {
    [
        vec4_dot_pos3(col_mat4x3_row(mat, 0), a),
        vec4_dot_pos3(col_mat4x3_row(mat, 1), a),
        vec4_dot_pos3(col_mat4x3_row(mat, 2), a)
    ]
}

/// Transforms a 2D vector through matrix.
#[inline(always)]
pub fn row_mat2x3_transform_vec2<T: Float + Copy>(
    mat: Matrix2x3<T>, 
    a: Vector2<T>
) -> Vector2<T> {
    [
        vec3_dot_vec2(mat[0], a),
        vec3_dot_vec2(mat[1], a)
    ]
}

/// Transforms a 2D vector through matrix.
#[inline(always)]
pub fn col_mat3x2_transform_vec2<T: Float + Copy>(
    mat: Matrix3x2<T>,
    a: Vector2<T>
) -> Vector2<T> {
    [
        vec3_dot_vec2(col_mat3x2_row(mat, 0), a),
        vec3_dot_vec2(col_mat3x2_row(mat, 1), a)
    ]
}

/// Transforms a 2D vector through row matrix.
#[inline(always)]
pub fn row_mat3_transform_vec2<T: Float + Copy>(
    mat: Matrix3<T>,
    a: Vector2<T>
) -> Vector2<T> {
    [
        vec3_dot_vec2(mat[0], a),
        vec3_dot_vec2(mat[1], a)
    ]
}

/// Transforms a 2D vector through column matrix.
#[inline(always)]
pub fn col_mat3_transform_vec2<T: Float + Copy>(
    mat: Matrix3<T>,
    a: Vector2<T>
) -> Vector2<T> {
    [
        vec3_dot_vec2(col_mat3_row(mat, 0), a),
        vec3_dot_vec2(col_mat3_row(mat, 1), a)
    ]
}

/// Transforms a 3D vector through matrix.
#[inline(always)]
pub fn row_mat3x4_transform_vec3<T: Float + Copy>(
    mat: Matrix3x4<T>, 
    a: Vector3<T>
) -> Vector3<T> {
    [
        vec4_dot_vec3(mat[0], a),
        vec4_dot_vec3(mat[1], a),
        vec4_dot_vec3(mat[2], a)
    ]
}

/// Transforms a 3D vector through matrix.
#[inline(always)]
pub fn col_mat4x3_transform_vec3<T: Float + Copy>(
    mat: Matrix4x3<T>,
    a: Vector3<T>
) -> Vector3<T> {
    [
        vec4_dot_vec3(col_mat4x3_row(mat, 0), a),
        vec4_dot_vec3(col_mat4x3_row(mat, 1), a),
        vec4_dot_vec3(col_mat4x3_row(mat, 2), a)
    ]
}

/// Computes the determinant of a matrix.
pub fn mat2x3_det<T: Float>(mat: Matrix2x3<T>) -> T {
      mat[0][0] * mat[1][1]
    - mat[0][1] * mat[1][0]
}

/// Computes the determinant of a matrix.
pub fn mat3x2_det<T: Float>(mat: Matrix3x2<T>) -> T {
      mat[0][0] * mat[1][1]
    - mat[0][1] * mat[1][0]
}

/// Computes the determinant of a matrix.
pub fn mat3_det<T: Float>(mat: Matrix3<T>) -> T {
      mat[0][0] * mat[1][1] * mat[2][2]
    + mat[0][1] * mat[1][2] * mat[2][0]
    + mat[0][2] * mat[1][0] * mat[2][1]
    - mat[0][0] * mat[1][2] * mat[2][1]
    - mat[0][1] * mat[1][0] * mat[2][2]
    - mat[0][2] * mat[1][1] * mat[2][0]
}

/// Computes the determinant of a matrix.
pub fn mat3x4_det<T: Float>(mat: Matrix3x4<T>) -> T {
      mat[0][0] * mat[1][1] * mat[2][2]
    + mat[0][1] * mat[1][2] * mat[2][0]
    + mat[0][2] * mat[1][0] * mat[2][1]
    - mat[0][0] * mat[1][2] * mat[2][1]
    - mat[0][1] * mat[1][0] * mat[2][2]
    - mat[0][2] * mat[1][1] * mat[2][0]
}

/// Computes the determinant of a matrix.
pub fn mat4x3_det<T: Float>(mat: Matrix4x3<T>) -> T {
      mat[0][0] * mat[1][1] * mat[2][2]
    + mat[0][1] * mat[1][2] * mat[2][0]
    + mat[0][2] * mat[1][0] * mat[2][1]
    - mat[0][0] * mat[1][2] * mat[2][1]
    - mat[0][1] * mat[1][0] * mat[2][2]
    - mat[0][2] * mat[1][1] * mat[2][0]
}

/// Computes the determinant of a 4x4 matrix.
pub fn mat4_det<T: Float>(mat: Matrix4<T>) -> T {
      mat[0][0] * mat[1][1] * mat[2][2] * mat[3][3] 
    + mat[0][0] * mat[1][2] * mat[2][3] * mat[3][1]
    + mat[0][0] * mat[1][3] * mat[2][1] * mat[3][2]

    + mat[0][1] * mat[1][0] * mat[2][3] * mat[3][2]
    + mat[0][1] * mat[1][2] * mat[2][0] * mat[3][3]
    + mat[0][1] * mat[1][3] * mat[2][2] * mat[3][0]

    + mat[0][2] * mat[1][0] * mat[2][1] * mat[3][3]
    + mat[0][2] * mat[1][1] * mat[2][3] * mat[3][0]
    + mat[0][2] * mat[1][3] * mat[2][0] * mat[3][1]

    + mat[0][3] * mat[1][0] * mat[2][2] * mat[3][1]
    + mat[0][3] * mat[1][1] * mat[2][0] * mat[3][2]
    + mat[0][3] * mat[1][2] * mat[2][1] * mat[3][0]

    - mat[0][0] * mat[1][1] * mat[2][3] * mat[3][2]
    - mat[0][0] * mat[1][2] * mat[2][1] * mat[3][3]
    - mat[0][0] * mat[1][3] * mat[2][2] * mat[3][1]

    - mat[0][1] * mat[1][0] * mat[2][2] * mat[3][3]
    - mat[0][1] * mat[1][2] * mat[2][3] * mat[3][0]
    - mat[0][1] * mat[1][3] * mat[2][0] * mat[3][2]

    - mat[0][2] * mat[1][0] * mat[2][3] * mat[3][1]
    - mat[0][2] * mat[1][1] * mat[2][0] * mat[3][3]
    - mat[0][2] * mat[1][3] * mat[2][1] * mat[3][0]

    - mat[0][3] * mat[1][0] * mat[2][1] * mat[3][2]
    - mat[0][3] * mat[1][1] * mat[2][2] * mat[3][0]
    - mat[0][3] * mat[1][2] * mat[2][0] * mat[3][1]
}

/// Computes inverse determinant of a 2x3 matrix.
#[inline(always)]
pub fn mat2x3_inv_det<T: Float>(mat: Matrix2x3<T>) -> T {
    let one = T::one();
    one / mat2x3_det(mat)
}

/// Computes inverse determinant of a 3x2 matrix.
#[inline(always)]
pub fn mat3x2_inv_det<T: Float>(mat: Matrix3x2<T>) -> T {
    let one = T::one();
    one / mat3x2_det(mat)
}

/// Computes inverse determinant of a 3x3 matrix.
#[inline(always)]
pub fn mat3_inv_det<T: Float>(mat: Matrix3<T>) -> T {
    let one = T::one();
    one / mat3_det(mat)
}

/// Computes inverse determinant of a 3x4 matrix.
#[inline(always)]
pub fn mat3x4_inv_det<T: Float>(mat: Matrix3x4<T>) -> T {
    let one = T::one();
    one / mat3x4_det(mat)
}

/// Computes inverse determinant of a 4x3 matrix.
#[inline(always)]
pub fn mat4x3_inv_det<T: Float>(mat: Matrix4x3<T>) -> T {
    let one = T::one();
    one / mat4x3_det(mat)
}

/// Computes the inverse determinant of a 4x4 matrix.
#[inline(always)]
pub fn mat4_inv_det<T: Float>(mat: Matrix4<T>) -> T {
    let one = T::one();
    one / mat4_det(mat)
}

/// Computes the inverse of a 2x3 matrix.
pub fn mat2x3_inv<T: Float + Copy>(mat: Matrix2x3<T>) -> Matrix2x3<T> {
    let inv_det = mat2x3_inv_det(mat);

    [
        [
              mat[1][1] * inv_det,
            - mat[0][1] * inv_det,
            (
              mat[0][1] * mat[1][2]
            - mat[0][2] * mat[1][1]
            ) * inv_det
        ],
        [
            - mat[1][0] * inv_det,
              mat[0][0] * inv_det,
            (
              mat[0][2] * mat[1][0]
            - mat[0][0] * mat[1][2]
            ) * inv_det,
        ]
    ]
}

/// Computes the inverse of a 3x2 matrix.
pub fn mat3x2_inv<T: Float + Copy>(mat: Matrix3x2<T>) -> Matrix3x2<T> {
    let inv_det = mat3x2_inv_det(mat);

    [
        [
            mat[1][1] * inv_det,
          - mat[0][1] * inv_det
        ],
        [
          - mat[1][0] * inv_det,
            mat[0][0] * inv_det
        ],
        [
            (
                  mat[1][0] * mat[2][1]
                - mat[1][1] * mat[2][0]
            ) * inv_det,
            (
                  mat[0][1] * mat[2][0]
                - mat[0][0] * mat[2][1]
            ) * inv_det
        ]
    ]
}

/// Computes the inverse of a 3x3 matrix.
pub fn mat3_inv<T: Float + Copy>(mat: Matrix3<T>) -> Matrix3<T> {
    let inv_det = mat3_inv_det(mat);

    [
        [   (
                  mat[1][1] * mat[2][2]
                - mat[1][2] * mat[2][1]
            ) * inv_det,
            (
                  mat[0][2] * mat[2][1]
                - mat[0][1] * mat[2][2]
            ) * inv_det,
            (
                  mat[0][1] * mat[1][2]
                - mat[0][2] * mat[1][1]
            ) * inv_det
        ],
        [
            (
                  mat[1][2] * mat[2][0]
                - mat[1][0] * mat[2][2]
            ) * inv_det,
            (
                  mat[0][0] * mat[2][2]
                - mat[0][2] * mat[2][0]
            ) * inv_det,
            (
                  mat[0][2] * mat[1][0]
                - mat[0][0] * mat[1][2]
            ) * inv_det
        ],
        [
            (
                  mat[1][0] * mat[2][1]
                - mat[1][1] * mat[2][0]
            ) * inv_det,
            (
                  mat[0][1] * mat[2][0]
                - mat[0][0] * mat[2][1]
            ) * inv_det,
            (
                  mat[0][0] * mat[1][1]
                - mat[0][1] * mat[1][0]
            ) * inv_det
        ]
    ]
}

/// Computes the inverse of a 3x4 matrix.
pub fn mat3x4_inv<T: Float + Copy>(mat: Matrix3x4<T>) -> Matrix3x4<T> {
    let inv_det = mat3x4_inv_det(mat);

    [
        [   (
              mat[1][1] * mat[2][2]
            - mat[1][2] * mat[2][1]
            ) * inv_det,
            (
              mat[0][2] * mat[2][1]
            - mat[0][1] * mat[2][2]
            ) * inv_det,
            (
              mat[0][1] * mat[1][2]
            - mat[0][2] * mat[1][1]
            ) * inv_det,
            (
              mat[0][1] * mat[1][3] * mat[2][2]
            + mat[0][2] * mat[1][1] * mat[2][3]
            + mat[0][3] * mat[1][2] * mat[2][1]
            - mat[0][1] * mat[1][2] * mat[2][3]
            - mat[0][2] * mat[1][3] * mat[2][1]
            - mat[0][3] * mat[1][1] * mat[2][2]
            ) * inv_det
        ],
        [
            (
              mat[1][2] * mat[2][0]
            - mat[1][0] * mat[2][2]
            ) * inv_det,
            (
              mat[0][0] * mat[2][2]
            - mat[0][2] * mat[2][0]
            ) * inv_det,
            (
              mat[0][2] * mat[1][0]
            - mat[0][0] * mat[1][2]
            ) * inv_det,
            (
              mat[0][0] * mat[1][2] * mat[2][3]
            + mat[0][2] * mat[1][3] * mat[2][0]
            + mat[0][3] * mat[1][0] * mat[2][2]
            - mat[0][0] * mat[1][3] * mat[2][2]
            - mat[0][2] * mat[1][0] * mat[2][3]
            - mat[0][3] * mat[1][2] * mat[2][0]
            ) * inv_det
        ],
        [
            (
              mat[1][0] * mat[2][1]
            - mat[1][1] * mat[2][0]
            ) * inv_det,
            (
              mat[0][1] * mat[2][0]
            - mat[0][0] * mat[2][1]
            ) * inv_det,
            (
              mat[0][0] * mat[1][1]
            - mat[0][1] * mat[1][0]
            ) * inv_det,
            (
              mat[0][0] * mat[1][3] * mat[2][1]
            + mat[0][1] * mat[1][0] * mat[2][3]
            + mat[0][3] * mat[1][1] * mat[2][0]
            - mat[0][0] * mat[1][1] * mat[2][3]
            - mat[0][1] * mat[1][3] * mat[2][0]
            - mat[0][3] * mat[1][0] * mat[2][1]
            ) * inv_det
        ]
    ]
}

/// Computes the inverse of a 4x3 matrix.
pub fn mat4x3_inv<T: Float + Copy>(mat: Matrix4x3<T>) -> Matrix4x3<T> {
    let inv_det = mat4x3_inv_det(mat);

    [
        [   (
                  mat[1][1] * mat[2][2]
                - mat[1][2] * mat[2][1]
            ) * inv_det,
            (
                  mat[0][2] * mat[2][1]
                - mat[0][1] * mat[2][2]
            ) * inv_det,
            (
                  mat[0][1] * mat[1][2]
                - mat[0][2] * mat[1][1]
            ) * inv_det
        ],
        [
            (
                  mat[1][2] * mat[2][0]
                - mat[1][0] * mat[2][2]
            ) * inv_det,
            (
                  mat[0][0] * mat[2][2]
                - mat[0][2] * mat[2][0]
            ) * inv_det,
            (
                  mat[0][2] * mat[1][0]
                - mat[0][0] * mat[1][2]
            ) * inv_det
        ],
        [
            (
                  mat[1][0] * mat[2][1]
                - mat[1][1] * mat[2][0]
            ) * inv_det,
            (
                  mat[0][1] * mat[2][0]
                - mat[0][0] * mat[2][1]
            ) * inv_det,
            (
                  mat[0][0] * mat[1][1]
                - mat[0][1] * mat[1][0]
            ) * inv_det
        ],
        [
            (
                mat[1][0] * mat[2][2] * mat[3][1]
                + mat[1][1] * mat[2][0] * mat[3][2]
                + mat[1][2] * mat[2][1] * mat[3][0]
                - mat[1][0] * mat[2][1] * mat[3][2]
                - mat[1][1] * mat[2][2] * mat[3][0]
                - mat[1][2] * mat[2][0] * mat[3][1]
            ) * inv_det,
            (
                mat[0][0] * mat[2][1] * mat[3][2]
                + mat[0][1] * mat[2][2] * mat[3][0]
                + mat[0][2] * mat[2][0] * mat[3][1]
                - mat[0][0] * mat[2][2] * mat[3][1]
                - mat[0][1] * mat[2][0] * mat[3][2]
                - mat[0][2] * mat[2][1] * mat[3][0]
            ) * inv_det,
            (
                mat[0][0] * mat[1][2] * mat[3][1]
                + mat[0][1] * mat[1][0] * mat[3][2]
                + mat[0][2] * mat[1][1] * mat[3][0]
                - mat[0][0] * mat[1][1] * mat[3][2]
                - mat[0][1] * mat[1][2] * mat[3][0]
                - mat[0][2] * mat[1][0] * mat[3][1]
            ) * inv_det
        ]
    ]
}

/// Computes the inverse of a 4x4 matrix.
pub fn mat4_inv<T: Float + Copy>(mat: Matrix4<T>) -> Matrix4<T> {
    let inv_det = mat4_inv_det(mat);

    [
        [   (
                mat[1][1] * mat[2][2] * mat[3][3]
                + mat[1][2] * mat[2][3] * mat[3][1]
                + mat[1][3] * mat[2][1] * mat[3][2]
                - mat[1][1] * mat[2][3] * mat[3][2]
                - mat[1][2] * mat[2][1] * mat[3][3]
                - mat[1][3] * mat[2][2] * mat[3][1]
            ) * inv_det,
            (
                mat[0][1] * mat[2][3] * mat[3][2]
                + mat[0][2] * mat[2][1] * mat[3][3]
                + mat[0][3] * mat[2][2] * mat[3][1]
                - mat[0][1] * mat[2][2] * mat[3][3]
                - mat[0][2] * mat[2][3] * mat[3][1]
                - mat[0][3] * mat[2][1] * mat[3][2]
            ) * inv_det,
            (
                mat[0][1] * mat[1][2] * mat[3][3]
                + mat[0][2] * mat[1][3] * mat[3][1]
                + mat[0][3] * mat[1][1] * mat[3][2]
                - mat[0][1] * mat[1][3] * mat[3][2]
                - mat[0][2] * mat[1][1] * mat[3][3]
                - mat[0][3] * mat[1][2] * mat[3][1]
            ) * inv_det,
            (
                mat[0][1] * mat[1][3] * mat[2][2]
                + mat[0][2] * mat[1][1] * mat[2][3]
                + mat[0][3] * mat[1][2] * mat[2][1]
                - mat[0][1] * mat[1][2] * mat[2][3]
                - mat[0][2] * mat[1][3] * mat[2][1]
                - mat[0][3] * mat[1][1] * mat[2][2]
            ) * inv_det
        ],
        [
            (
                mat[1][0] * mat[2][3] * mat[3][2]
                + mat[1][2] * mat[2][0] * mat[3][3]
                + mat[1][3] * mat[2][2] * mat[3][0]
                - mat[1][0] * mat[2][2] * mat[3][3]
                - mat[1][2] * mat[2][3] * mat[3][0]
                - mat[1][3] * mat[2][0] * mat[3][2]
            ) * inv_det,
            (
                mat[0][0] * mat[2][2] * mat[3][3]
                + mat[0][2] * mat[2][3] * mat[3][0]
                + mat[0][3] * mat[2][0] * mat[3][2]
                - mat[0][0] * mat[2][3] * mat[3][2]
                - mat[0][2] * mat[2][0] * mat[3][3]
                - mat[0][3] * mat[2][2] * mat[3][0]
            ) * inv_det,
            (
                mat[0][0] * mat[1][3] * mat[3][2]
                + mat[0][2] * mat[1][0] * mat[3][3]
                + mat[0][3] * mat[1][2] * mat[3][0]
                - mat[0][0] * mat[1][2] * mat[3][3]
                - mat[0][2] * mat[1][3] * mat[3][0]
                - mat[0][3] * mat[1][0] * mat[3][2]
            ) * inv_det,
            (
                mat[0][0] * mat[1][2] * mat[2][3]
                + mat[0][2] * mat[1][3] * mat[2][0]
                + mat[0][3] * mat[1][0] * mat[2][2]
                - mat[0][0] * mat[1][3] * mat[2][2]
                - mat[0][2] * mat[1][0] * mat[2][3]
                - mat[0][3] * mat[1][2] * mat[2][0]
            ) * inv_det
        ],
        [
            (
                mat[1][0] * mat[2][1] * mat[3][3]
                + mat[1][1] * mat[2][3] * mat[3][0]
                + mat[1][3] * mat[2][0] * mat[3][1]
                - mat[1][0] * mat[2][3] * mat[3][1]
                - mat[1][1] * mat[2][0] * mat[3][3]
                - mat[1][3] * mat[2][1] * mat[3][0]
            ) * inv_det,
            (
                mat[0][0] * mat[2][3] * mat[3][1]
                + mat[0][1] * mat[2][0] * mat[3][3]
                + mat[0][3] * mat[2][1] * mat[3][0]
                - mat[0][0] * mat[2][1] * mat[3][3]
                - mat[0][1] * mat[2][3] * mat[3][0]
                - mat[0][3] * mat[2][0] * mat[3][1]
            ) * inv_det,
            (
                mat[0][0] * mat[1][1] * mat[3][3]
                + mat[0][1] * mat[1][3] * mat[3][0]
                + mat[0][3] * mat[1][0] * mat[3][1]
                - mat[0][0] * mat[1][3] * mat[3][1]
                - mat[0][1] * mat[1][0] * mat[3][3]
                - mat[0][3] * mat[1][1] * mat[3][0]
            ) * inv_det,
            (
                mat[0][0] * mat[1][3] * mat[2][1]
                + mat[0][1] * mat[1][0] * mat[2][3]
                + mat[0][3] * mat[1][1] * mat[2][0]
                - mat[0][0] * mat[1][1] * mat[2][3]
                - mat[0][1] * mat[1][3] * mat[2][0]
                - mat[0][3] * mat[1][0] * mat[2][1]
            ) * inv_det
        ],
        [
            (
                mat[1][0] * mat[2][2] * mat[3][1]
                + mat[1][1] * mat[2][0] * mat[3][2]
                + mat[1][2] * mat[2][1] * mat[3][0]
                - mat[1][0] * mat[2][1] * mat[3][2]
                - mat[1][1] * mat[2][2] * mat[3][0]
                - mat[1][2] * mat[2][0] * mat[3][1]
            ) * inv_det,
            (
                mat[0][0] * mat[2][1] * mat[3][2]
                + mat[0][1] * mat[2][2] * mat[3][0]
                + mat[0][2] * mat[2][0] * mat[3][1]
                - mat[0][0] * mat[2][2] * mat[3][1]
                - mat[0][1] * mat[2][0] * mat[3][2]
                - mat[0][2] * mat[2][1] * mat[3][0]
            ) * inv_det,
            (
                mat[0][0] * mat[1][2] * mat[3][1]
                + mat[0][1] * mat[1][0] * mat[3][2]
                + mat[0][2] * mat[1][1] * mat[3][0]
                - mat[0][0] * mat[1][1] * mat[3][2]
                - mat[0][1] * mat[1][2] * mat[3][0]
                - mat[0][2] * mat[1][0] * mat[3][1]
            ) * inv_det,
            (
                mat[0][0] * mat[1][1] * mat[2][2]
                + mat[0][1] * mat[1][2] * mat[2][0]
                + mat[0][2] * mat[1][0] * mat[2][1]
                - mat[0][0] * mat[1][2] * mat[2][1]
                - mat[0][1] * mat[1][0] * mat[2][2]
                - mat[0][2] * mat[1][1] * mat[2][0]
            ) * inv_det
        ]
    ]
}


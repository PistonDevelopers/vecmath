#![crate_name = "vecmath"]
#![deny(missing_doc)]

//! A simple and generic library for vector math.

use std::num::{One};

/// A 3D vector.
pub type Vector3<T> = [T, ..3];

/// A 4D vector.
pub type Vector4<T> = [T, ..4];

/// A matrix in row major format.
///
/// Notice that row major is mathematical standard,
/// while OpenGL uses column major format.
/// To use matrices with OpenGL, convert to base vectors,
/// which uses vectors per column.
pub type Matrix3x4<T> = [[T, ..4], ..3];

/// A matrix with 4 rows and 4 columns.
pub type Matrix4<T> = [[T, ..4], ..4];

/// A matrix in column major format.
///
/// This format can also store vertices of a quad.
pub type Base4x3<T> = [[T, ..3], ..4];

/// Subtracts 'b' from 'a'.
#[inline(always)]
pub fn vec3_sub<T: Num>(a: Vector3<T>, b: Vector3<T>) -> Vector3<T> {
    [
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2],
    ]
}

/// Adds two vectors.
#[inline(always)]
pub fn vec3_add<T: Num>(a: Vector3<T>, b: Vector3<T>) -> Vector3<T> {
    [
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2]
    ]
}

/// Computes the dot product.
#[inline(always)]
pub fn vec3_dot<T: Num>(a: Vector3<T>) -> T {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
}

/// Computes the cross product.
#[inline(always)]
pub fn vec3_cross<T: Num>(a: Vector3<T>, b: Vector3<T>) -> Vector3<T> {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]
}

/// Multiplies the vector with a scalar.
#[inline(always)]
pub fn vec3_mul<T: Num>(a: Vector3<T>, b: T) -> Vector3<T> {
    [
        a[0] * b,
        a[1] * b,
        a[2] * b
    ]
}

/// Computes the length of vector.
#[inline(always)]
pub fn vec3_len<T: Float>(a: Vector3<T>) -> T {
    vec3_dot(a).sqrt()
}

/// Computes the inverse length of a vector.
#[inline(always)]
pub fn vec3_inv_len<T: Float>(a: Vector3<T>) -> T {
    let one: T = One::one();
    one / vec3_len(a)
}

/// Computes the normalized.
#[inline(always)]
pub fn vec3_normalized<T: Float>(a: Vector3<T>) -> Vector3<T> {
    vec3_mul(a, vec3_inv_len(a))
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

/// Computes transformed vector component.
///
/// This is used when transforming vectors through matrices.
#[inline(always)]
pub fn vec4_dot_vec<T: Num>(a: Vector4<T>, b: Vector3<T>) -> T {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Computes transformed position component.
///
/// This is used when transforming points through matrices.
#[inline(always)]
pub fn vec4_dot_pos<T: Num + Copy>(a: Vector4<T>, b: Vector3<T>) -> T {
    vec4_dot_vec(a, b) + a[3]
}

/// Returns a row vector from a base vector column matrix.
#[inline(always)]
pub fn base4x3_row<T: Copy>(base: Base4x3<T>, i: uint) -> Vector4<T> {
    [base[0][i], base[1][i], base[2][i], base[3][i]]
}

/// Constructs a 3x4 matrix from base vectors.
#[inline(always)]
pub fn base4x3_mat<T: Copy>(base: Base4x3<T>) -> Matrix3x4<T> {
    [
        base4x3_row(base, 0),
        base4x3_row(base, 1),
        base4x3_row(base, 2)
    ]
}

/// Gets a column of a 3x4 matrix.
#[inline(always)]
pub fn mat3x4_col<T: Copy>(mat: Matrix3x4<T>, i: uint) -> Vector3<T> {
    [mat[0][i], mat[1][i], mat[2][i]]
}

/// Transforms a 3D position through matrix.
#[inline(always)]
pub fn mat3x4_transform_pos<T: Num + Copy>(
    mat: Matrix3x4<T>, 
    a: Vector3<T>
) -> Vector3<T> {
    [
        vec4_dot_pos(mat[0], a),
        vec4_dot_pos(mat[1], a),
        vec4_dot_pos(mat[2], a),
    ]
}

/// Transforms a 3D vector through matrix.
#[inline(always)]
pub fn mat3x_transform_vec<T: Num + Copy>(
    mat: Matrix3x4<T>, a: Vector3<T>
) -> Vector3<T> {
    [
        vec4_dot_vec(mat[0], a),
        vec4_dot_vec(mat[1], a),
        vec4_dot_vec(mat[2], a)
    ]
}

/// Computes the determinant of a 4x4 matrix.
pub fn mat4_det<T: Num>(mat: Matrix4<T>) -> T {
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

/// Computes the inverse determinant of a 4x4 matrix.
#[inline(always)]
pub fn mat4_inv_det<T: Num>(mat: Matrix4<T>) -> T {
    let one: T = One::one();
    one / mat4_det(mat)
}

/// Computes the inverse of a 4x4 matrix.
pub fn mat4_inv<T: Num + Copy>(mat: Matrix4<T>) -> Matrix4<T> {
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

/// Computes the determinant of a matrix.
pub fn mat3x4_det<T: Num>(mat: Matrix3x4<T>) -> T {
      mat[0][0] * mat[1][1] * mat[2][2]
    + mat[0][1] * mat[1][2] * mat[2][0]
    + mat[0][2] * mat[1][0] * mat[2][1]
    - mat[0][0] * mat[1][2] * mat[2][1]
    - mat[0][1] * mat[1][0] * mat[2][2]
    - mat[0][2] * mat[1][1] * mat[2][0]
}

/// Computes inverse determinant of a 3x4 matrix.
#[inline(always)]
pub fn mat3x4_inv_det<T: Num>(mat: Matrix3x4<T>) -> T {
    let one: T = One::one();
    one / mat3x4_det(mat)
}

/// Computes the inverse of a 3x4 matrix.
pub fn mat3x4_inv<T: Num + Copy>(mat: Matrix3x4<T>) -> Matrix3x4<T> {
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

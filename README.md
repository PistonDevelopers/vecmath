# vecmath [![Build Status](https://travis-ci.org/PistonDevelopers/vecmath.svg?branch=master)](https://travis-ci.org/PistonDevelopers/vecmath)

A simple and type agnostic Rust library for vector math designed for reexporting

[How to contribute](https://github.com/PistonDevelopers/piston/blob/master/CONTRIBUTING.md)

## Motivation

It is hard to agree on the "the best" way of designing a linear algebra or game math library.  
Sometimes you only need a few functions, like dot or cross product and don't want to deal with traits.  
Perhaps you know the math, but the type system gets in your way?  
This library is designed to be simple, generic and easy to build abstractions on top of.  

If this is not what you looking for, here are some alternatives:

* Game math - [cgmath-rs](https://github.com/bjz/cgmath-rs)
* Linear algebra + game math - [nalgebra](http://nalgebra.org/)

### Goals

* No traits
* No type declarations that gives dependencies
* Global functions only to make it easier to build on top of it
* Fixed arrays
* Row vs column major matrices agnostic
* Generics

### Non-Goals

* Game math or linear algebra
* Traits or module system that people can disagree on
* Quaternions
* Dual numbers
* Matrix setup

## Usage

A good convention is to reexport the library in a `vecmath` module in your library/executable.  
By reexporting you can add new functions and send pull requests without breaking the code.  
Besides, it is nicer to put custom math functions under the same name space,  
even if those never will be added to the original vector math library.  

### Edit your 'Cargo.toml' file

Open up 'Cargo.toml' in a text editor and add the following:

```
[dependencies.vecmath]
version = "1.0.0"
```

### Step 1

Add the following to 'lib.rs':

```Rust
extern crate vecmath;

mod math; // Use 'pub mod' if you want it to be visible outside library.
```

### Step 2

Create a new file 'math.rs' in your 'src/' directory.
Open 'math.rs' in a text editor and type:

```Rust
pub use vecmath::row_mat2x3_mul as multiply;

pub type Matrix2d = vecmath::Matrix2x3<f64>;

// etc.
```

You can add your own custom functions and rename the existing ones for your usage.

## Naming conventions

All methods are prefixed with a short name version.  

Examples:

`mat3x4_` a 3x4 matrix.

`mat4_` a 4x4 matrix.

`vec3_` a vector with 3 components.

`col_mat4x3_` the matrix is treated as a matrix with column major

## Generic conventions

For simplicity, all methods should take a generic parameter with specific operations, e.g. `<T: Add<T, Output = T>>` for addition.  
In cases where extra methods are required, `traits::Float` should be used instead.  
All arguments are passed by value, so `Copy` can be added as requirement when needed.  
Inlining will remove the overhead.  

This increases readability and is good enough for the usage of this library.

vecmath
=======

A simple and type agnostic library for vector math designed for reexporting

## Motivation

It is hard to agree on the "the best" way of designing a linear algebra or game math library.  
Sometimes you only need a few functions, like dot or cross product and don't want to deal with traits.  
This library is designed to be simple, generic and easy to build abstractions on top of.  

### Goals

* Global functions only to make it easier to build on top of it
* Fixed arrays
* Column major matrices for less confusion
* Base vector matrices with row major, to use with OpenGL and make matrix construction easier
* Generics

### Non-Goals

* A complete library for game or linear algebra
* Traits or module system that people can disagree on

## Usage

A good convention is to reexport the library in a `vecmath` module in your library/executable.  
By reexporting you can add new functions and send pull requests without breaking the code.  
Besides, it is nicer to put custom math functions under the same name space,  
even if those never will be added to the original vector math library.  

### Edit your 'Cargo.toml' file

If you do not have one, you can use [Rust-Empty](https://github.com/bvssvni/rust-empty) and generate one with `make cargo-lib` for libraries or `make cargo-exe` for executables.

Open up 'Cargo.toml' in a text editor and add the following:

```
[dependencies.vecmath]

git = "https://github.com/PistonDevelopers/vecmath"
```

### Step 1

Add the following to 'lib.rs':

```Rust
#![feature(globs)]

exter crate vecmath;

mod vecmath; // Use 'pub mod' if you want it to be visible outside library.
```

### Step 2

Create a new file 'vecmath.rs' in your 'src/' directory.
Open 'vecmath.rs' in a text editor and type:

```Rust
pub use vecmath::*;
```

You can add your own custom functions, but please follow the same naming conventions.

## Naming conventions

All methods are prefixed with a short name version.  

Examples:

`mat3x4_` a matrix with 3 rows and 4 columns.

`mat4_` a matrix with 4 rows and 4 columns.

`base4x3_` a row major base vector matrix with 4 columns and 3 rows.

`vec3_` a vector with 3 components.

## Generic conventions

For simplicity, all methods should take a generic parameter `<T: Num>`.  

This increases readability and is good enough for the usage of this library.

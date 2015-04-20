//! Various useful traits

use std::ops::*;

/// Convenience trait for floats.
pub trait Float:
    Radians + One + Zero
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Div<Self, Output = Self>
    + Rem<Self, Output = Self>
    + Neg<Output = Self> {}

impl<T> Float for T where
    T: Radians + One + Zero
    + Add<T, Output = T>
    + Mul<T, Output = T>
    + Sub<T, Output = T>
    + Div<T, Output = T>
    + Rem<T, Output = T>
    + Neg<Output = T> {}

/// Useful constants for radians.
pub trait Radians {
    /// Returns radians corresponding to 90 degrees.
    fn _90() -> Self;

    /// Returns radians corresponding to 180 degrees.
    fn _180() -> Self;

    /// Returns radians corresponding to 360 degrees.
    fn _360() -> Self;

    /// Convert a value to radians from degrees.
    /// Equivalent to ```value * (π / 180)```.
    fn deg_to_rad(self) -> Self;

    /// Convert a value to degrees from radians.
    /// Equivalent to ```value * (180 / π)```.
    fn rad_to_deg(self) -> Self;
}

impl Radians for f32 {
    #[inline(always)]
    fn _90() -> f32 {
        ::std::f32::consts::FRAC_PI_2
    }

    #[inline(always)]
    fn _180() -> f32 {
        ::std::f32::consts::PI
    }

    #[inline(always)]
    fn _360() -> f32 {
        <Self as Radians>::_180() * 2.0
    }

    #[inline(always)]
    fn deg_to_rad(self) -> Self {
        self * (::std::f32::consts::PI / 180.0f32)
    }

    #[inline(always)]
    fn rad_to_deg(self) -> Self {
        self * (180.0f32 / ::std::f32::consts::PI)
    }
}

impl Radians for f64 {
    #[inline(always)]
    fn _90() -> f64 {
        ::std::f64::consts::FRAC_PI_2
    }

    #[inline(always)]
    fn _180() -> f64 {
        ::std::f64::consts::PI
    }

    #[inline(always)]
    fn _360() -> f64 {
        <Self as Radians>::_180() * 2.0
    }

    #[inline(always)]
    fn deg_to_rad(self) -> Self {
        self * (::std::f64::consts::PI / 180.0f64)
    }

    #[inline(always)]
    fn rad_to_deg(self) -> Self {
        self * (180.0f64 / ::std::f64::consts::PI)
    }
}

/// Number 1.
pub trait One {
    /// Returns 1.
    fn one() -> Self;
}

/// Number 0.
pub trait Zero {
    /// Returns 0.
    fn zero() -> Self;
}

impl One for f64 {
    #[inline(always)]
    fn one() -> f64 { 1.0 }
}

impl One for f32 {
    #[inline(always)]
    fn one() -> f32 { 1.0 }
}

impl Zero for f64 {
    #[inline(always)]
    fn zero() -> f64 { 0.0 }
}

impl Zero for f32 {
    #[inline(always)]
    fn zero() -> f32 { 0.0 }
}

/// Square root.
pub trait Sqrt {
    /// Returns square root.
    fn sqrt(self) -> Self;
}

impl Sqrt for f32 {
    #[inline(always)]
    fn sqrt(self) -> f32 { self.sqrt() }
}

impl Sqrt for f64 {
    #[inline(always)]
    fn sqrt(self) -> f64 { self.sqrt() }
}

/// Casts into another type.
pub trait Cast<T> {
    /// Casts into other type.
    fn cast(self) -> T;
}

impl Cast<f32> for f64 {
    fn cast(self) -> f32 { self as f32 }
}

impl Cast<f64> for f32 {
    fn cast(self) -> f64 { self as f64 }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_f32_sqrt() {
        let a = 4.0f32;
        let b = <f32 as Sqrt>::sqrt(a);
        assert_eq!(b, 2.0f32);
    }

    #[test]
    fn test_f64_sqrt() {
        let a = 4.0f64;
        let b = <f64 as Sqrt>::sqrt(a);
        assert_eq!(b, 2.0f64);
    }

    #[test]
    fn test_f32_deg_to_rad() {
        let degrees = 23.0f32;
        let radians = degrees.deg_to_rad();
        assert!(f32::abs_sub(radians, 0.401425) > ::std::f32::EPSILON);
    }

    #[test]
    fn test_f64_deg_to_rad() {
        let degrees = 60.0f64;
        let radians = degrees.deg_to_rad();
        assert!(f64::abs_sub(radians, 1.047197)  > ::std::f64::EPSILON);
    }
}

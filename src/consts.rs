//! Various useful constants

use std::num::FromPrimitive;

/// Useful constants for radians.
pub trait Radians : FromPrimitive {
    /// Returns radians corresponding to 90 degrees.
    fn _90() -> Self;

    /// Returns radians corresponding to 180 degrees.
    fn _180() -> Self;

    /// Returns radians corresponding to 360 degrees.
    fn _360() -> Self;

    /// Convert a value to radians, assuming the initial units
    /// are in degrees. Equivelent to:
    /// ```ignore
    ///     value * (π / 180)
    /// ```
    fn to_rad(&self) -> Self;
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
        ::std::f32::consts::PI_2
    }

    #[inline(always)]
    fn to_rad(&self) -> Self {
        *self * (::std::f32::consts::PI / 180.0f32)
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
        ::std::f64::consts::PI_2
    }

    #[inline(always)]
    fn to_rad(&self) -> Self {
        *self * (::std::f64::consts::PI / 180.0f64)
    }
}

#[cfg(test)]
mod test {
    use super::{Radians};
    use std::num::Float;

    #[test]
    fn test_f32_deg_to_rad() {
        let degrees = 23.0f32;
        let radians = degrees.to_rad();
        assert!(Float::abs_sub(radians, 0.401425) > ::std::f32::EPSILON);
    }

    #[test]
    fn test_f64_deg_to_rad() {
        let degrees = 60.0f64;
        let radians = degrees.to_rad();
        assert!(Float::abs_sub(radians, 1.047197)  > ::std::f64::EPSILON);
    }
}


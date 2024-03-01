pub extern crate rustfft;

// export rustfft to phastft
use std::f64::consts::PI;

use rand::{distributions::Uniform, prelude::*};
use rustfft::num_traits::Float;

/// Asserts that two f64 numbers are approximately equal.
///
/// # Panics
///
/// Panics if `actual` and `expected` are too far from each other
#[allow(dead_code)]
#[track_caller]
pub fn assert_f64_closeness(actual: f64, expected: f64, epsilon: f64) {
    if (actual - expected).abs() >= epsilon {
        panic!(
            "Assertion failed: {actual} too far from expected value {expected} (with epsilon {epsilon})",
        );
    }
}

/// Asserts that two f32 numbers are approximately equal.
///
/// # Panics
///
/// Panics if `actual` and `expected` are too far from each other
#[allow(dead_code)]
#[track_caller]
pub fn assert_f32_closeness(actual: f32, expected: f32, epsilon: f32) {
    if (actual - expected).abs() >= epsilon {
        panic!(
            "Assertion failed: {actual} too far from expected value {expected} (with epsilon {epsilon})",
        );
    }
}

/// Generate a random, complex, signal in the provided buffers
///
/// # Panics
///
/// Panics if `reals.len() != imags.len()`
pub fn gen_random_signal_f32(reals: &mut [f32], imags: &mut [f32]) {
    assert!(reals.len() == imags.len() && !reals.is_empty());
    let mut rng = thread_rng();
    let between = Uniform::from(0.0..1.0);
    let angle_dist = Uniform::from(0.0..2.0 * PI);
    let num_amps = reals.len();

    let mut probs: Vec<_> = (0..num_amps).map(|_| between.sample(&mut rng)).collect();

    let total: _ = probs.iter().sum();
    let total_recip = total.recip();

    probs.iter_mut().for_each(|p| *p *= total_recip);

    let angles = (0..num_amps).map(|_| angle_dist.sample(&mut rng));

    probs
        .iter()
        .zip(angles)
        .enumerate()
        .for_each(|(i, (p, a))| {
            let p_sqrt = p.sqrt();
            let (sin_a, cos_a) = a.sin_cos();
            let re = p_sqrt * cos_a;
            let im = p_sqrt * sin_a;
            reals[i] = re;
            imags[i] = im;
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_random_signal() {
        let big_n = 1 << 25;
        let mut reals: Vec<_> = vec![0.0; big_n];
        let mut imags: Vec<_> = vec![0.0; big_n];

        gen_random_signal(&mut reals, &mut imags);

        let sum = reals
            .iter()
            .zip(imags.iter())
            .map(|(re, im)| re.powi(2) + im.powi(2))
            .sum();

        assert_f64_closeness(sum, 1.0, 1e-6);
    }
}

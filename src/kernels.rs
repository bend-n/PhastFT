use std::any::TypeId;
use std::simd::{f32x16, f64x8};

use num_traits::Float;

trait FftChunkSIMD {
    type F;
    fn fft_chunk_n_simd(
        reals: &mut [Self::F],
        imags: &mut [Self::F],
        twiddles_re: &[Self::F],
        twiddles_im: &[Self::F],
        dist: usize,
    );
}

impl FftChunkSIMD for f64 {
    type F = f64;

    fn fft_chunk_n_simd(
        reals: &mut [Self::F],
        imags: &mut [Self::F],
        twiddles_re: &[Self::F],
        twiddles_im: &[Self::F],
        dist: usize,
    ) {
        const N: usize = 8;
        let chunk_size = dist << 1;
        assert!(chunk_size >= 16);

        reals
            .chunks_exact_mut(chunk_size)
            .zip(imags.chunks_exact_mut(chunk_size))
            .for_each(|(reals_chunk, imags_chunk)| {
                let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
                let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

                reals_s0
                    .chunks_exact_mut(N)
                    .zip(reals_s1.chunks_exact_mut(N))
                    .zip(imags_s0.chunks_exact_mut(N))
                    .zip(imags_s1.chunks_exact_mut(N))
                    .zip(twiddles_re.chunks_exact(N))
                    .zip(twiddles_im.chunks_exact(N))
                    .for_each(|(((((re_s0, re_s1), im_s0), im_s1), w_re), w_im)| {
                        let real_c0 = f64x8::from_slice(re_s0);
                        let real_c1 = f64x8::from_slice(re_s1);
                        let imag_c0 = f64x8::from_slice(im_s0);
                        let imag_c1 = f64x8::from_slice(im_s1);
                        let tw_re = f64x8::from_slice(w_re);
                        let tw_im = f64x8::from_slice(w_im);

                        re_s0.copy_from_slice((real_c0 + real_c1).as_array());
                        im_s0.copy_from_slice((imag_c0 + imag_c1).as_array());
                        let v_re = real_c0 - real_c1;
                        let v_im = imag_c0 - imag_c1;
                        re_s1.copy_from_slice((v_re * tw_re - v_im * tw_im).as_array());
                        im_s1.copy_from_slice((v_re * tw_im + v_im * tw_re).as_array());
                    });
            });
    }
}

impl FftChunkSIMD for f32 {
    type F = f32;

    fn fft_chunk_n_simd(
        reals: &mut [Self::F],
        imags: &mut [Self::F],
        twiddles_re: &[Self::F],
        twiddles_im: &[Self::F],
        dist: usize,
    ) {
        const N: usize = 8;
        let chunk_size = dist << 1;
        assert!(chunk_size >= 16);

        reals
            .chunks_exact_mut(chunk_size)
            .zip(imags.chunks_exact_mut(chunk_size))
            .for_each(|(reals_chunk, imags_chunk)| {
                let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
                let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

                reals_s0
                    .chunks_exact_mut(N)
                    .zip(reals_s1.chunks_exact_mut(N))
                    .zip(imags_s0.chunks_exact_mut(N))
                    .zip(imags_s1.chunks_exact_mut(N))
                    .zip(twiddles_re.chunks_exact(N))
                    .zip(twiddles_im.chunks_exact(N))
                    .for_each(|(((((re_s0, re_s1), im_s0), im_s1), w_re), w_im)| {
                        let real_c0 = f32x16::from_slice(re_s0);
                        let real_c1 = f32x16::from_slice(re_s1);
                        let imag_c0 = f32x16::from_slice(im_s0);
                        let imag_c1 = f32x16::from_slice(im_s1);
                        let tw_re = f32x16::from_slice(w_re);
                        let tw_im = f32x16::from_slice(w_im);

                        re_s0.copy_from_slice((real_c0 + real_c1).as_array());
                        im_s0.copy_from_slice((imag_c0 + imag_c1).as_array());
                        let v_re = real_c0 - real_c1;
                        let v_im = imag_c0 - imag_c1;
                        re_s1.copy_from_slice((v_re * tw_re - v_im * tw_im).as_array());
                        im_s1.copy_from_slice((v_re * tw_im + v_im * tw_re).as_array());
                    });
            });
    }
}

#[inline(always)]
/// Return `true` if `A` and `B` are the same type
fn same_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

pub(crate) fn fft_chunk_n_simd<T: FftChunkSIMD>(
    reals: &mut [T],
    imags: &mut [T],
    twiddles_re: &[T],
    twiddles_im: &[T],
    dist: usize,
) {
    let chunk_size = dist << 1;
    assert!(chunk_size >= 16);
    fft_chunk_n_simd(reals, imags, twiddles_re, twiddles_im, dist);
}

pub(crate) fn fft_chunk_n<T: Float>(
    reals: &mut [T],
    imags: &mut [T],
    twiddles_re: &[T],
    twiddles_im: &[T],
    dist: usize,
) {
    let chunk_size = dist << 1;

    reals
        .chunks_exact_mut(chunk_size)
        .zip(imags.chunks_exact_mut(chunk_size))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

            reals_s0
                .iter_mut()
                .zip(reals_s1.iter_mut())
                .zip(imags_s0.iter_mut())
                .zip(imags_s1.iter_mut())
                .zip(twiddles_re.iter())
                .zip(twiddles_im.iter())
                .for_each(|(((((re_s0, re_s1), im_s0), im_s1), w_re), w_im)| {
                    let real_c0 = *re_s0;
                    let real_c1 = *re_s1;
                    let imag_c0 = *im_s0;
                    let imag_c1 = *im_s1;

                    *re_s0 = real_c0 + real_c1;
                    *im_s0 = imag_c0 + imag_c1;
                    let v_re = real_c0 - real_c1;
                    let v_im = imag_c0 - imag_c1;
                    *re_s1 = v_re * *w_re - v_im * *w_im;
                    *im_s1 = v_re * *w_im + v_im * *w_re;
                });
        });
}

/// `chunk_size == 4`, so hard code twiddle factors
pub(crate) fn fft_chunk_4<T: Float>(reals: &mut [T], imags: &mut [T]) {
    let dist = 2;
    let chunk_size = dist << 1;

    reals
        .chunks_exact_mut(chunk_size)
        .zip(imags.chunks_exact_mut(chunk_size))
        .for_each(|(reals_chunk, imags_chunk)| {
            let (reals_s0, reals_s1) = reals_chunk.split_at_mut(dist);
            let (imags_s0, imags_s1) = imags_chunk.split_at_mut(dist);

            let real_c0 = reals_s0[0];
            let real_c1 = reals_s1[0];
            let imag_c0 = imags_s0[0];
            let imag_c1 = imags_s1[0];

            reals_s0[0] = real_c0 + real_c1;
            imags_s0[0] = imag_c0 + imag_c1;
            reals_s1[0] = real_c0 - real_c1;
            imags_s1[0] = imag_c0 - imag_c1;

            let real_c0 = reals_s0[1];
            let real_c1 = reals_s1[1];
            let imag_c0 = imags_s0[1];
            let imag_c1 = imags_s1[1];

            reals_s0[1] = real_c0 + real_c1;
            imags_s0[1] = imag_c0 + imag_c1;
            reals_s1[1] = imag_c0 - imag_c1;
            imags_s1[1] = -(real_c0 - real_c1);
        });
}

/// `chunk_size == 2`, so skip phase
pub(crate) fn fft_chunk_2<T: Float>(reals: &mut [T], imags: &mut [T]) {
    reals
        .chunks_exact_mut(2)
        .zip(imags.chunks_exact_mut(2))
        .for_each(|(reals_chunk, imags_chunk)| {
            let z0_re = reals_chunk[0];
            let z0_im = imags_chunk[0];
            let z1_re = reals_chunk[1];
            let z1_im = imags_chunk[1];

            reals_chunk[0] = z0_re + z1_re;
            imags_chunk[0] = z0_im + z1_im;
            reals_chunk[1] = z0_re - z1_re;
            imags_chunk[1] = z0_im - z1_im;
        });
}

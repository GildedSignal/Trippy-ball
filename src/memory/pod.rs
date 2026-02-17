use std::mem::size_of;

pub unsafe trait Pod: Copy {}

macro_rules! impl_pod_primitives {
    ($($ty:ty),* $(,)?) => {
        $(unsafe impl Pod for $ty {})*
    };
}

impl_pod_primitives!(u8, i8, u16, i16, u32, i32, u64, i64, usize, isize, f32, f64);

unsafe impl<T: Pod, const N: usize> Pod for [T; N] {}

#[inline]
pub fn zeroed<T: Pod>() -> T {
    unsafe { std::mem::zeroed() }
}

#[inline]
pub fn read_unaligned<T: Pod>(bytes: &[u8]) -> T {
    assert!(
        bytes.len() >= size_of::<T>(),
        "insufficient bytes for unaligned read"
    );
    unsafe { (bytes.as_ptr() as *const T).read_unaligned() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_pod<T: Pod>() {}

    #[test]
    fn zeroed_primitives_and_arrays_are_zero() {
        let value: u32 = zeroed();
        let array: [u32; 4] = zeroed();
        assert_eq!(value, 0);
        assert_eq!(array, [0, 0, 0, 0]);
    }

    #[test]
    fn read_unaligned_reads_from_non_aligned_start() {
        let bytes = [0xAAu8, 0x78, 0x56, 0x34, 0x12, 0xCC];
        let value = read_unaligned::<u32>(&bytes[1..]);
        assert_eq!(value, 0x1234_5678);
    }

    #[test]
    fn pod_trait_supports_fixed_size_arrays() {
        assert_pod::<[u32; 8]>();
        assert_pod::<[f32; 3]>();
    }

    #[test]
    #[should_panic(expected = "insufficient bytes for unaligned read")]
    fn read_unaligned_panics_when_input_too_short() {
        let _ = read_unaligned::<u32>(&[1, 2, 3]);
    }
}

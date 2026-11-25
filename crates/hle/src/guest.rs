//! Reading and writing guest strings (NUL-terminated UTF-16 and ANSI).

/// Longest string we walk before giving up (guards against missing NULs).
pub const MAX_CHARS: usize = 1 << 16;

/// Read a NUL-terminated UTF-16 string. `None` for a null pointer.
///
/// # Safety
/// `p` must be null or point at a NUL-terminated UTF-16 string.
pub unsafe fn wstr(p: *const u16) -> Option<String> {
    if p.is_null() {
        return None;
    }
    let mut buf = Vec::new();
    // SAFETY: caller guarantees a terminated string; capped walk.
    unsafe {
        while buf.len() < MAX_CHARS {
            let c = p.add(buf.len()).read_unaligned();
            if c == 0 {
                break;
            }
            buf.push(c);
        }
    }
    Some(String::from_utf16_lossy(&buf))
}

/// Read a NUL-terminated ANSI string (treated as UTF-8, lossy).
///
/// # Safety
/// `p` must be null or point at a NUL-terminated byte string.
pub unsafe fn cstr(p: *const u8) -> Option<String> {
    if p.is_null() {
        return None;
    }
    let mut buf = Vec::new();
    // SAFETY: as above.
    unsafe {
        while buf.len() < MAX_CHARS {
            let c = p.add(buf.len()).read();
            if c == 0 {
                break;
            }
            buf.push(c);
        }
    }
    Some(String::from_utf8_lossy(&buf).into_owned())
}

/// Copy `s` as UTF-16 + NUL into a buffer of `cap` u16s, truncating.
/// Returns the number of u16s written, excluding the NUL.
///
/// # Safety
/// `dst` must be valid for `cap` u16 writes.
pub unsafe fn write_wstr(dst: *mut u16, cap: usize, s: &str) -> usize {
    if cap == 0 || dst.is_null() {
        return 0;
    }
    let w: Vec<u16> = s.encode_utf16().take(cap - 1).collect();
    // SAFETY: w.len() + 1 <= cap.
    unsafe {
        std::ptr::copy_nonoverlapping(w.as_ptr(), dst, w.len());
        dst.add(w.len()).write_unaligned(0);
    }
    w.len()
}

/// UTF-16 + NUL, for building guest-side test data.
pub fn to_wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(Some(0)).collect()
}

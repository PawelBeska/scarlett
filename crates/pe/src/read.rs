use crate::{Error, Result};

/// Bounds-checked little-endian reader over a byte slice.
#[derive(Clone, Copy)]
pub(crate) struct Reader<'a> {
    data: &'a [u8],
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Reader { data }
    }

    pub fn bytes(&self, offset: usize, len: usize, what: &'static str) -> Result<&'a [u8]> {
        offset
            .checked_add(len)
            .and_then(|end| self.data.get(offset..end))
            .ok_or(Error::Truncated { what, offset })
    }

    fn array<const N: usize>(&self, offset: usize, what: &'static str) -> Result<[u8; N]> {
        Ok(self.bytes(offset, N, what)?.try_into().unwrap())
    }

    pub fn u16(&self, offset: usize, what: &'static str) -> Result<u16> {
        self.array(offset, what).map(u16::from_le_bytes)
    }

    pub fn u32(&self, offset: usize, what: &'static str) -> Result<u32> {
        self.array(offset, what).map(u32::from_le_bytes)
    }

    pub fn u64(&self, offset: usize, what: &'static str) -> Result<u64> {
        self.array(offset, what).map(u64::from_le_bytes)
    }

    /// NUL-terminated string, capped at 4 KiB.
    pub fn cstr(&self, offset: usize, what: &'static str) -> Result<String> {
        let tail = self
            .data
            .get(offset..)
            .ok_or(Error::Truncated { what, offset })?;
        let tail = &tail[..tail.len().min(4096)];
        let end = tail
            .iter()
            .position(|&b| b == 0)
            .ok_or(Error::Truncated { what, offset })?;
        Ok(String::from_utf8_lossy(&tail[..end]).into_owned())
    }
}

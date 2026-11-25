//! GPU register file, split by the spaces the `SET_*_REG` packets address,
//! with copy-on-write snapshots so every draw can keep the state it ran with.
//!
//! Register numbers are dword indices (byte offset / 4), as PM4 uses them.
//! Named offsets are GFX10 values from AMD's public register descriptions
//! (as in Mesa's `sid.h`); only the ones the command processor itself reads
//! are listed here.

use std::sync::Arc;

pub const CONFIG_BASE: u32 = 0x2000;
pub const CONFIG_SIZE: u32 = 0x0C00;
pub const SH_BASE: u32 = 0x2C00;
pub const SH_SIZE: u32 = 0x0400;
pub const CONTEXT_BASE: u32 = 0xA000;
pub const CONTEXT_SIZE: u32 = 0x1000;
pub const UCONFIG_BASE: u32 = 0xC000;
pub const UCONFIG_SIZE: u32 = 0x4000;

// SH: graphics shader stages (pixel, vertex/NGG) and compute.
pub const SPI_SHADER_PGM_LO_PS: u32 = 0x2C08;
pub const SPI_SHADER_PGM_HI_PS: u32 = 0x2C09;
pub const SPI_SHADER_PGM_RSRC1_PS: u32 = 0x2C0A;
pub const SPI_SHADER_PGM_RSRC2_PS: u32 = 0x2C0B;
pub const SPI_SHADER_USER_DATA_PS_0: u32 = 0x2C0C;
pub const SPI_SHADER_PGM_LO_VS: u32 = 0x2C48;
pub const SPI_SHADER_PGM_HI_VS: u32 = 0x2C49;
pub const SPI_SHADER_PGM_RSRC1_VS: u32 = 0x2C4A;
pub const SPI_SHADER_PGM_RSRC2_VS: u32 = 0x2C4B;
pub const SPI_SHADER_USER_DATA_VS_0: u32 = 0x2C4C;
pub const COMPUTE_DISPATCH_INITIATOR: u32 = 0x2E00;
pub const COMPUTE_DIM_X: u32 = 0x2E01;
pub const COMPUTE_START_X: u32 = 0x2E04;
pub const COMPUTE_START_Y: u32 = 0x2E05;
pub const COMPUTE_START_Z: u32 = 0x2E06;
pub const COMPUTE_NUM_THREAD_X: u32 = 0x2E07;
pub const COMPUTE_NUM_THREAD_Y: u32 = 0x2E08;
pub const COMPUTE_NUM_THREAD_Z: u32 = 0x2E09;
pub const COMPUTE_PGM_LO: u32 = 0x2E0C;
pub const COMPUTE_PGM_HI: u32 = 0x2E0D;
pub const COMPUTE_PGM_RSRC1: u32 = 0x2E12;
pub const COMPUTE_PGM_RSRC2: u32 = 0x2E13;
pub const COMPUTE_USER_DATA_0: u32 = 0x2E40;

// CONTEXT
pub const CB_TARGET_MASK: u32 = 0xA08E;
pub const PA_CL_VPORT_XSCALE: u32 = 0xA10F;
pub const DB_DEPTH_CONTROL: u32 = 0xA200;
pub const PA_SU_SC_MODE_CNTL: u32 = 0xA205;
pub const CB_COLOR0_BASE: u32 = 0xA318;

// UCONFIG
pub const VGT_PRIMITIVE_TYPE: u32 = 0xC242;
pub const VGT_INDEX_TYPE: u32 = 0xC243;
pub const VGT_NUM_INSTANCES: u32 = 0xC24D;

/// `VGT_PRIMITIVE_TYPE` values (`DI_PT_*`).
pub mod prim {
    pub const POINTLIST: u32 = 0x01;
    pub const LINELIST: u32 = 0x02;
    pub const LINESTRIP: u32 = 0x03;
    pub const TRILIST: u32 = 0x04;
    pub const TRIFAN: u32 = 0x05;
    pub const TRISTRIP: u32 = 0x06;
    pub const RECTLIST: u32 = 0x11;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Space {
    Config,
    Sh,
    Context,
    Uconfig,
}

impl Space {
    pub fn base(self) -> u32 {
        match self {
            Space::Config => CONFIG_BASE,
            Space::Sh => SH_BASE,
            Space::Context => CONTEXT_BASE,
            Space::Uconfig => UCONFIG_BASE,
        }
    }

    pub fn of(reg: u32) -> Option<(Space, usize)> {
        let s = match reg {
            CONFIG_BASE..0x2C00 => Space::Config,
            SH_BASE..0x3000 => Space::Sh,
            CONTEXT_BASE..0xB000 => Space::Context,
            UCONFIG_BASE..0x10000 => Space::Uconfig,
            _ => return None,
        };
        Some((s, (reg - s.base()) as usize))
    }
}

/// All four spaces. Cloning is four `Arc` bumps; a write after a clone
/// copies only the space it touches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Regs {
    config: Arc<Vec<u32>>,
    sh: Arc<Vec<u32>>,
    context: Arc<Vec<u32>>,
    uconfig: Arc<Vec<u32>>,
}

impl Default for Regs {
    fn default() -> Self {
        Regs {
            config: Arc::new(vec![0; CONFIG_SIZE as usize]),
            sh: Arc::new(vec![0; SH_SIZE as usize]),
            context: Arc::new(vec![0; CONTEXT_SIZE as usize]),
            uconfig: Arc::new(vec![0; UCONFIG_SIZE as usize]),
        }
    }
}

impl Regs {
    fn space(&self, s: Space) -> &Arc<Vec<u32>> {
        match s {
            Space::Config => &self.config,
            Space::Sh => &self.sh,
            Space::Context => &self.context,
            Space::Uconfig => &self.uconfig,
        }
    }

    fn space_mut(&mut self, s: Space) -> &mut Vec<u32> {
        Arc::make_mut(match s {
            Space::Config => &mut self.config,
            Space::Sh => &mut self.sh,
            Space::Context => &mut self.context,
            Space::Uconfig => &mut self.uconfig,
        })
    }

    /// Register by absolute dword index; 0 outside the known spaces.
    pub fn get(&self, reg: u32) -> u32 {
        Space::of(reg).map_or(0, |(s, i)| self.space(s)[i])
    }

    /// 64-bit value from a `LO`/`HI` register pair.
    pub fn get64(&self, lo: u32) -> u64 {
        self.get(lo) as u64 | (self.get(lo + 1) as u64) << 32
    }

    /// Write by absolute index. Returns false (and ignores the write) for
    /// registers outside the known spaces.
    pub fn set(&mut self, reg: u32, v: u32) -> bool {
        match Space::of(reg) {
            Some((s, i)) => {
                self.space_mut(s)[i] = v;
                true
            }
            None => false,
        }
    }

    /// Consecutive writes from `reg`; stops at the end of the space.
    pub fn set_range(&mut self, reg: u32, values: &[u32]) -> usize {
        let Some((s, i)) = Space::of(reg) else {
            return 0;
        };
        let space = self.space_mut(s);
        let n = values.len().min(space.len() - i);
        space[i..i + n].copy_from_slice(&values[..n]);
        n
    }

    pub fn clear_context(&mut self) {
        self.context = Arc::new(vec![0; CONTEXT_SIZE as usize]);
    }

    /// Do two snapshots share the storage of `space` (no copy happened)?
    pub fn shares(&self, other: &Regs, s: Space) -> bool {
        Arc::ptr_eq(self.space(s), other.space(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spaces_and_copy_on_write() {
        let mut r = Regs::default();
        assert!(r.set(CB_COLOR0_BASE, 0x1234));
        assert!(!r.set(0x1000, 1), "outside every space");
        let snap = r.clone();
        r.set(SPI_SHADER_PGM_LO_PS, 7);
        assert_eq!(snap.get(SPI_SHADER_PGM_LO_PS), 0, "snapshot unaffected");
        assert!(
            r.shares(&snap, Space::Context),
            "untouched space not copied"
        );
        assert!(!r.shares(&snap, Space::Sh));
        assert_eq!(
            r.set_range(0xAFFE, &[1, 2, 3]),
            2,
            "clipped at the end of CONTEXT"
        );
        r.set(COMPUTE_PGM_LO, 0x10);
        r.set(COMPUTE_PGM_HI, 0x2);
        assert_eq!(r.get64(COMPUTE_PGM_LO), 0x2_0000_0010);
    }
}

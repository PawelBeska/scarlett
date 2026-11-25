//! COM-style objects implemented in Rust and handed to guest code.
//!
//! Console APIs hand out interfaces rather than flat functions: the GDK
//! runtime's `QueryApiImpl` returns per-feature classes, D3D12 returns
//! devices and command lists. To the guest such an object is a pointer to
//! a pointer to a vtable; each vtable slot is an ordinary HLE thunk, so a
//! method call `call [rax+8*n]` goes through the same trampoline as an
//! import, with `this` in rcx.
//!
//! Guest-visible layout of every object made here:
//!
//! ```text
//! this+0   vtable pointer
//! this+8   reference count (u32)
//! this+12  class index (u32)
//! this+16  id into the host-side state table (u64)
//! ```
//!
//! Slots 0-2 are always `IUnknown` (`QueryInterface`, `AddRef`,
//! `Release`); a class lists its own methods from slot 3 on.
//! `QueryInterface` succeeds for `IUnknown` and every interface id the
//! class declares, all served by the same vtable.

use crate::{Call, Handler};
use scarlett_pe::ImportName;
use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, RwLock};

pub const S_OK: u64 = 0;
pub const E_NOINTERFACE: u32 = 0x8000_4002;
pub const E_POINTER: u32 = 0x8000_4003;
pub const E_NOTIMPL: u32 = 0x8000_4001;

/// A GUID in its in-memory (little-endian `Data1..3`) layout.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Guid(pub [u8; 16]);

pub const IID_IUNKNOWN: Guid = Guid::from_fields(
    0x0000_0000,
    0x0000,
    0x0000,
    [0xC0, 0, 0, 0, 0, 0, 0, 0x46],
);

impl Guid {
    pub const fn from_fields(d1: u32, d2: u16, d3: u16, d4: [u8; 8]) -> Self {
        let a = d1.to_le_bytes();
        let b = d2.to_le_bytes();
        let c = d3.to_le_bytes();
        Guid([
            a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], d4[0], d4[1], d4[2], d4[3], d4[4],
            d4[5], d4[6], d4[7],
        ])
    }

    /// `"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"`, braces optional.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().trim_start_matches('{').trim_end_matches('}');
        let p: Vec<&str> = s.split('-').collect();
        if p.len() != 5 || p.iter().map(|x| x.len()).ne([8, 4, 4, 4, 12]) {
            return None;
        }
        let d1 = u32::from_str_radix(p[0], 16).ok()?;
        let d2 = u16::from_str_radix(p[1], 16).ok()?;
        let d3 = u16::from_str_radix(p[2], 16).ok()?;
        let tail = format!("{}{}", p[3], p[4]);
        let mut d4 = [0u8; 8];
        for (i, b) in d4.iter_mut().enumerate() {
            *b = u8::from_str_radix(&tail[i * 2..i * 2 + 2], 16).ok()?;
        }
        Some(Self::from_fields(d1, d2, d3, d4))
    }

    /// Read a GUID from guest memory. `None` for null.
    ///
    /// # Safety
    /// `p` must be null or point at 16 readable bytes.
    pub unsafe fn read(p: u64) -> Option<Self> {
        if p == 0 {
            return None;
        }
        // SAFETY: caller contract.
        Some(Guid(unsafe { (p as *const [u8; 16]).read_unaligned() }))
    }
}

impl std::fmt::Display for Guid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let b = &self.0;
        let d1 = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let d2 = u16::from_le_bytes([b[4], b[5]]);
        let d3 = u16::from_le_bytes([b[6], b[7]]);
        write!(f, "{d1:08x}-{d2:04x}-{d3:04x}-{:02x}{:02x}-", b[8], b[9])?;
        for x in &b[10..] {
            write!(f, "{x:02x}")?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for Guid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{{self}}}")
    }
}

/// A class: its name (for logs and thunk names), the interface ids it
/// answers to besides `IUnknown`, and its methods from slot 3 on.
pub struct ComClass {
    pub name: &'static str,
    pub iids: Vec<Guid>,
    pub methods: Vec<(&'static str, Handler)>,
}

struct Registered {
    class: ComClass,
    vtable: u64,
}

#[repr(C)]
#[allow(dead_code)] // `class` and `state` are part of the guest-visible layout
struct Header {
    vtable: u64,
    refs: AtomicU32,
    class: u32,
    state: u64,
}

pub type State = Arc<dyn Any + Send + Sync>;

#[derive(Default)]
struct Table {
    classes: Vec<Registered>,
    by_name: HashMap<&'static str, u32>,
    /// Live objects: guest address → state.
    objects: HashMap<u64, State>,
}

static TABLE: LazyLock<RwLock<Table>> = LazyLock::new(Default::default);
static NEXT_STATE: AtomicU64 = AtomicU64::new(1);

/// Register `class` (once per name; later calls return the first index).
pub fn register_class(class: ComClass) -> u32 {
    if let Some(&i) = TABLE.read().unwrap().by_name.get(class.name) {
        return i;
    }
    let mut slots: Vec<u64> = Vec::with_capacity(3 + class.methods.len());
    let dll = format!("com:{}", class.name);
    let thunk = |name: &str, h: Handler| {
        crate::module::thunk_for(
            &dll,
            &ImportName::Name {
                hint: 0,
                name: name.to_string(),
            },
            h,
        )
    };
    slots.push(thunk("QueryInterface", query_interface));
    slots.push(thunk("AddRef", add_ref));
    slots.push(thunk("Release", release));
    for (i, (name, h)) in class.methods.iter().enumerate() {
        slots.push(thunk(&format!("{name}#{}", i + 3), *h));
    }
    let vtable = Box::leak(slots.into_boxed_slice()).as_ptr() as u64;
    let mut t = TABLE.write().unwrap();
    if let Some(&i) = t.by_name.get(class.name) {
        return i;
    }
    let idx = t.classes.len() as u32;
    t.by_name.insert(class.name, idx);
    t.classes.push(Registered { class, vtable });
    idx
}

/// Interface ids `class` answers to (besides `IUnknown`).
pub fn class_iids(class: u32) -> Vec<Guid> {
    TABLE.read().unwrap().classes[class as usize].class.iids.clone()
}

pub fn class_name(class: u32) -> &'static str {
    TABLE.read().unwrap().classes[class as usize].class.name
}

pub fn class_index(name: &str) -> Option<u32> {
    TABLE.read().unwrap().by_name.get(name).copied()
}

/// A new object of class `class` with one reference, owning `state`.
pub fn new_object(class: u32, state: State) -> u64 {
    let id = NEXT_STATE.fetch_add(1, Ordering::Relaxed);
    let mut t = TABLE.write().unwrap();
    let vtable = t.classes[class as usize].vtable;
    let this = Box::into_raw(Box::new(Header {
        vtable,
        refs: AtomicU32::new(1),
        class,
        state: id,
    })) as u64;
    t.objects.insert(this, state);
    this
}

/// `this` if it is a live object made by [`new_object`]. Checked against
/// the object table before `this` is ever read, so a stale or foreign
/// pointer is refused without touching its memory.
fn header(this: u64) -> Option<&'static Header> {
    if !TABLE.read().unwrap().objects.contains_key(&this) {
        return None;
    }
    // SAFETY: live object allocated by new_object; freed only after its
    // removal from `objects`.
    Some(unsafe { &*(this as *const Header) })
}

/// Host-side state of `this`, downcast to `T`.
pub fn state<T: Any + Send + Sync>(this: u64) -> Option<Arc<T>> {
    let s = TABLE.read().unwrap().objects.get(&this)?.clone();
    s.downcast::<T>().ok()
}

pub fn class_of(this: u64) -> Option<&'static str> {
    let h = header(this)?;
    Some(TABLE.read().unwrap().classes[h.class as usize].class.name)
}

pub fn ref_count(this: u64) -> u32 {
    header(this).map_or(0, |h| h.refs.load(Ordering::Relaxed))
}

/// Write `value` to the guest out-pointer `out` if it is non-null.
pub fn write_out(out: u64, value: u64) {
    if out != 0 {
        // SAFETY: guest out-parameter.
        unsafe { (out as *mut u64).write_unaligned(value) };
    }
}

fn query_interface(c: &Call) -> u64 {
    let (this, iid, out) = (c.arg(0), c.arg(1), c.arg(2));
    if out == 0 {
        return E_POINTER as u64;
    }
    let Some(h) = header(this) else {
        write_out(out, 0);
        return E_POINTER as u64;
    };
    // SAFETY: guest passes a REFIID.
    let Some(iid) = (unsafe { Guid::read(iid) }) else {
        write_out(out, 0);
        return E_POINTER as u64;
    };
    let t = TABLE.read().unwrap();
    let class = &t.classes[h.class as usize].class;
    if iid == IID_IUNKNOWN || class.iids.contains(&iid) {
        h.refs.fetch_add(1, Ordering::Relaxed);
        write_out(out, this);
        S_OK
    } else {
        log::debug!("{}::QueryInterface({iid}): no such interface", class.name);
        write_out(out, 0);
        E_NOINTERFACE as u64
    }
}

fn add_ref(c: &Call) -> u64 {
    header(c.arg(0)).map_or(0, |h| h.refs.fetch_add(1, Ordering::Relaxed) + 1) as u64
}

fn release(c: &Call) -> u64 {
    let this = c.arg(0);
    let Some(h) = header(this) else { return 0 };
    let left = h.refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if left == 0 {
        TABLE.write().unwrap().objects.remove(&this);
        // SAFETY: allocated by new_object; no longer in the object table.
        drop(unsafe { Box::from_raw(this as *mut Header) });
    }
    left as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guid_round_trip() {
        let g = Guid::parse("{00000000-0000-0000-C000-000000000046}").unwrap();
        assert_eq!(g, IID_IUNKNOWN);
        assert_eq!(g.to_string(), "00000000-0000-0000-c000-000000000046");
        let g = Guid::parse("12345678-9abc-def0-1122-334455667788").unwrap();
        assert_eq!(&g.0[..4], &[0x78, 0x56, 0x34, 0x12]);
        assert_eq!(&g.0[8..], &[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
        assert!(Guid::parse("1234").is_none());
    }

    fn noop(_: &Call) -> u64 {
        0
    }

    #[test]
    fn query_interface_and_release() {
        let other = Guid::parse("aaaaaaaa-0000-0000-0000-000000000000").unwrap();
        let cls = register_class(ComClass {
            name: "TestQI",
            iids: vec![other],
            methods: vec![("Nop", noop)],
        });
        let this = new_object(cls, Arc::new(7u32));
        assert_eq!(class_of(this), Some("TestQI"));
        assert_eq!(*state::<u32>(this).unwrap(), 7);
        let call = |args: [u64; 4]| unsafe { Call::new(args, std::ptr::null()) };

        let mut out = 0u64;
        let out_p = &mut out as *mut u64 as u64;
        let iid = |g: &Guid| g.0.as_ptr() as u64;
        assert_eq!(query_interface(&call([this, iid(&other), out_p, 0])), S_OK);
        assert_eq!((out, ref_count(this)), (this, 2));
        let missing = Guid::parse("bbbbbbbb-0000-0000-0000-000000000000").unwrap();
        assert_eq!(
            query_interface(&call([this, iid(&missing), out_p, 0])),
            E_NOINTERFACE as u64
        );
        assert_eq!(out, 0);

        assert_eq!(release(&call([this, 0, 0, 0])), 1);
        assert_eq!(release(&call([this, 0, 0, 0])), 0);
        assert_eq!(class_of(this), None, "freed object is no longer live");
        // A stale pointer is refused without being read.
        assert_eq!(release(&call([this, 0, 0, 0])), 0);
        assert_eq!(add_ref(&call([0x10, 0, 0, 0])), 0);
    }
}

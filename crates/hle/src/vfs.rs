//! Guest paths → host paths.
//!
//! The guest sees Windows paths (`G:\Data\level1.pak`, `\\?\T:\tmp`,
//! relative names). Each drive letter or Xbox-style device name
//! (`game:`) can be mounted on a host directory. Resolution:
//!
//! * `\\?\` / `\\.\` prefixes are dropped, `/` is treated as `\`;
//! * relative and root-relative paths use the guest current directory;
//! * `.` and `..` are folded, and `..` never climbs above the mount root
//!   (Windows clamps at the drive root, and that also keeps the guest
//!   inside the directory it was given);
//! * each component is matched case-insensitively against the host
//!   directory, since Windows file names are case-insensitive and Linux is
//!   not. Components that do not exist yet keep the guest's spelling.

use std::path::PathBuf;
use std::sync::RwLock;

pub const ERROR_PATH_NOT_FOUND: u32 = 3;

struct State {
    /// (lower-case device name without ':', host root)
    mounts: Vec<(String, PathBuf)>,
    /// Guest current directory, e.g. `G:\`.
    cwd: String,
}

static STATE: RwLock<State> = RwLock::new(State {
    mounts: Vec::new(),
    cwd: String::new(),
});

fn device_key(device: &str) -> String {
    device.trim_end_matches(':').to_lowercase()
}

/// Mount `host` as guest device `device` (`"G"`, `"g:"`, `"game:"`).
/// Re-mounting a device replaces it.
pub fn mount(device: &str, host: impl Into<PathBuf>) {
    let key = device_key(device);
    let mut s = STATE.write().unwrap();
    s.mounts.retain(|(k, _)| *k != key);
    s.mounts.push((key, host.into()));
}

pub fn unmount_all() {
    STATE.write().unwrap().mounts.clear();
}

/// Set the guest current directory (must be absolute, e.g. `G:\`).
pub fn set_current_dir(guest: &str) {
    STATE.write().unwrap().cwd = guest.replace('/', "\\");
}

pub fn current_dir() -> String {
    STATE.read().unwrap().cwd.clone()
}

/// Split a guest path into (device, components), applying cwd and folding
/// `.`/`..`. `None` if it names no device.
pub fn normalize(guest: &str) -> Option<(String, Vec<String>)> {
    let mut p = guest.replace('/', "\\");
    for prefix in ["\\\\?\\", "\\\\.\\", "\\??\\"] {
        if let Some(rest) = p.strip_prefix(prefix) {
            p = rest.to_string();
        }
    }
    let cwd = current_dir();
    let full = match p.find(':') {
        // `X:` or `game:` followed by `\` or nothing: absolute.
        Some(i) if !p[..i].contains('\\') && !p[i + 1..].starts_with(|c| c != '\\') => p,
        // `X:foo` (drive-relative) is rare; treat as `X:\foo`.
        Some(i) if !p[..i].contains('\\') => format!("{}\\{}", &p[..=i], &p[i + 1..]),
        _ if p.starts_with('\\') => {
            let dev = cwd.split(':').next()?;
            format!("{dev}:{p}")
        }
        _ => format!("{}\\{}", cwd.trim_end_matches('\\'), p),
    };
    let (dev, rest) = full.split_once(':')?;
    if dev.is_empty() {
        return None;
    }
    let mut parts: Vec<String> = Vec::new();
    for c in rest.split('\\') {
        match c {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            c => parts.push(c.trim_end_matches([' ', '.']).to_string()),
        }
    }
    Some((device_key(dev), parts))
}

/// Resolve a guest path to a host path, or a Win32 error code.
pub fn resolve(guest: &str) -> Result<PathBuf, u32> {
    let (dev, parts) = normalize(guest).ok_or(ERROR_PATH_NOT_FOUND)?;
    let root = {
        let s = STATE.read().unwrap();
        s.mounts
            .iter()
            .find(|(k, _)| *k == dev)
            .map(|(_, h)| h.clone())
            .ok_or(ERROR_PATH_NOT_FOUND)?
    };
    let mut host = root;
    for c in parts {
        let exact = host.join(&c);
        if exact.exists() {
            host = exact;
            continue;
        }
        let lower = c.to_lowercase();
        let found = std::fs::read_dir(&host).ok().and_then(|rd| {
            rd.flatten()
                .map(|e| e.file_name())
                .find(|n| n.to_string_lossy().to_lowercase() == lower)
        });
        host = match found {
            Some(n) => host.join(n),
            None => exact,
        };
    }
    Ok(host)
}

/// Windows wildcard match (`*`, `?`), case-insensitive. `*.*` matches
/// names without a dot too, as on Windows.
pub fn wildcard(pattern: &str, name: &str) -> bool {
    let pattern = if pattern == "*.*" { "*" } else { pattern };
    let p: Vec<char> = pattern.to_lowercase().chars().collect();
    let n: Vec<char> = name.to_lowercase().chars().collect();
    let (mut pi, mut ni) = (0, 0);
    let (mut star, mut mark) = (None, 0);
    while ni < n.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == n[ni]) {
            pi += 1;
            ni += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ni;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ni = mark;
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|&c| c == '*')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_paths() {
        set_current_dir("G:\\data");
        let n = |s: &str| normalize(s).map(|(d, p)| format!("{d}:{}", p.join("/")));
        assert_eq!(n("G:\\A\\b.txt").unwrap(), "g:A/b.txt");
        assert_eq!(n("\\\\?\\t:\\x\\..\\y").unwrap(), "t:y");
        assert_eq!(n("level\\1.pak").unwrap(), "g:data/level/1.pak");
        assert_eq!(n("\\root.bin").unwrap(), "g:root.bin");
        assert_eq!(
            n("g:/../../etc/passwd").unwrap(),
            "g:etc/passwd",
            "clamped at root"
        );
        assert_eq!(n("game:\\x").unwrap(), "game:x");
        assert_eq!(n("C:").unwrap(), "c:");
    }

    #[test]
    fn wildcards() {
        assert!(wildcard("*.TXT", "readme.txt"));
        assert!(wildcard("a?c*", "abcdef"));
        assert!(!wildcard("*.txt", "readme.md"));
        assert!(wildcard("*.*", "Makefile"));
        assert!(wildcard("*", ""));
    }
}

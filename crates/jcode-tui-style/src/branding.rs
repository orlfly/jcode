//! Application display name, so a downstream product built on this TUI can
//! rebrand the name and labels it shows without forking the UI.
//!
//! Defaults to `jcode`, i.e. the stock binary is unaffected unless
//! `JCODE_APP_NAME` is set. Products that embed this TUI (for example Gcode,
//! which sets `JCODE_APP_NAME=gcode` at startup) get their own name in the
//! header, window titles, notifications and onboarding copy.
//!
//! Scope: this is the *application being run* only. Vendor and service names
//! (the hosted Jcode subscription, its support address, remote-host version
//! hints) are deliberately not rebranded here: renaming them would mislabel
//! someone else's paid service.

use std::sync::OnceLock;

/// The stock application name.
pub const DEFAULT_APP_NAME: &str = "jcode";

/// Environment variable a downstream product sets to rebrand the UI.
pub const APP_NAME_ENV: &str = "JCODE_APP_NAME";

static APP_NAME: OnceLock<String> = OnceLock::new();

/// Read the configured name. Pure: used by [`app_name`] and by tests.
fn resolve_app_name() -> String {
    match std::env::var(APP_NAME_ENV) {
        Ok(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                DEFAULT_APP_NAME.to_string()
            } else {
                trimmed.to_string()
            }
        }
        Err(_) => DEFAULT_APP_NAME.to_string(),
    }
}

/// The application display name, e.g. `jcode` or `gcode`.
///
/// Cached for the process lifetime: the name is decided at startup and must not
/// change under a running UI.
pub fn app_name() -> &'static str {
    APP_NAME.get_or_init(resolve_app_name)
}

/// Title-cased name for prose, e.g. `Jcode` or `Gcode`.
pub fn app_name_title() -> String {
    titlecase(app_name())
}

fn titlecase(name: &str) -> String {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => name.to_string(),
    }
}

/// Rebrand an audited display string that mentions the stock name in prose.
pub fn rebrand_ui_text(text: &str) -> String {
    rebrand_ui_text_as(text, app_name())
}

/// [`rebrand_ui_text`] with an explicit name, so the rule is testable as a pure
/// function.
///
/// Only whole-word occurrences of `jcode` / `Jcode` are replaced. Identifiers
/// stay intact on purpose: `/usr/bin/jcode`, `JCODE_HOME`, `jcode_home` and
/// `jcode-updater` are paths, variables and package names, not prose.
pub fn rebrand_ui_text_as(text: &str, name: &str) -> String {
    if name == DEFAULT_APP_NAME || name.is_empty() {
        return text.to_string();
    }
    let title = titlecase(name);
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;
    while i < text.len() {
        let rest = &text[i..];
        let needle = if rest.starts_with("Jcode") {
            Some(("Jcode", title.as_str()))
        } else if rest.starts_with("jcode") {
            Some(("jcode", name))
        } else {
            None
        };
        match needle {
            Some((stock, replacement)) if is_whole_word(text, i, stock.len()) => {
                out.push_str(replacement);
                i += stock.len();
            }
            Some((stock, _)) => {
                // Part of a longer token: copy it verbatim.
                out.push_str(stock);
                i += stock.len();
            }
            None => {
                // Copy one character (not one byte: the text may be non-ASCII).
                let len = rest.chars().next().map(char::len_utf8).unwrap_or(1);
                out.push_str(&rest[..len]);
                i += len;
            }
        }
    }
    out
}

/// True when `text[at..at+len]` stands alone as prose rather than being part of
/// a bigger token.
///
/// Paths (`/usr/bin/jcode`), file names (`jcode.json`), package names
/// (`jcode-updater`), environment variables (`jcode_home`) and URL fragments are
/// all joined by one of these characters, so any of them on either side means
/// "identifier, leave it alone".
fn is_whole_word(text: &str, at: usize, len: usize) -> bool {
    let joins = |c: char| {
        c.is_ascii_alphanumeric()
            || matches!(c, '_' | '-' | '.' | '/' | '\\' | '@' | ':' | '+')
    };
    let before_ok = text[..at]
        .chars()
        .next_back()
        .map(|c| !joins(c))
        .unwrap_or(true);
    let after_ok = text[at + len..]
        .chars()
        .next()
        .map(|c| !joins(c))
        .unwrap_or(true);
    before_ok && after_ok
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test only: it mutates a process-global environment variable, and
    /// `app_name()` caches its result for the process.
    #[test]
    fn app_name_defaults_to_jcode_and_follows_the_env_var() {
        assert_eq!(DEFAULT_APP_NAME, "jcode");
        assert_eq!(APP_NAME_ENV, "JCODE_APP_NAME");

        // SAFETY: this test is the only writer of this variable, and no other
        // test in this crate reads `app_name()` concurrently.
        unsafe {
            std::env::remove_var(APP_NAME_ENV);
            assert_eq!(resolve_app_name(), DEFAULT_APP_NAME, "unset means stock name");

            std::env::set_var(APP_NAME_ENV, "   ");
            assert_eq!(resolve_app_name(), DEFAULT_APP_NAME, "blank means stock name");

            std::env::set_var(APP_NAME_ENV, " gcode ");
            assert_eq!(resolve_app_name(), "gcode", "value is trimmed");
            assert_eq!(app_name(), "gcode", "the cached accessor agrees");
            assert_eq!(app_name_title(), "Gcode");

            std::env::remove_var(APP_NAME_ENV);
        }
        assert_eq!(resolve_app_name(), DEFAULT_APP_NAME);
    }

    #[test]
    fn rebranding_replaces_prose_but_never_identifiers() {
        let r = |text: &str| rebrand_ui_text_as(text, "gcode");

        // Prose, both casings and word positions.
        assert_eq!(r("jcode collects anonymous usage statistics"), "gcode collects anonymous usage statistics");
        assert_eq!(r("Jcode is a terminal coding agent"), "Gcode is a terminal coding agent");
        assert_eq!(r("run jcode --resume"), "run gcode --resume");
        assert_eq!(r("jcode · new session"), "gcode · new session");
        assert_eq!(r("Update Jcode on the remote host."), "Update Gcode on the remote host.");
        assert_eq!(r("Tell Jcode where to connect."), "Tell Gcode where to connect.");

        // Identifiers, paths, variables and package names stay untouched.
        assert_eq!(r("/usr/bin/jcode"), "/usr/bin/jcode");
        assert_eq!(r("JCODE_HOME is set"), "JCODE_HOME is set");
        assert_eq!(r("jcode_home"), "jcode_home");
        assert_eq!(r("jcode-updater"), "jcode-updater");
        assert_eq!(r("jcode.json"), "jcode.json");
        assert_eq!(r("jcode/foo bar"), "jcode/foo bar");
        assert_eq!(r("myjcode"), "myjcode");
        assert_eq!(r("https://example.com/jcode"), "https://example.com/jcode");
        assert_eq!(r("auth@jcode"), "auth@jcode");

        // Non-ASCII text must survive byte-wise scanning.
        assert_eq!(r("中文 jcode 中文"), "中文 gcode 中文");

        // The stock name is a no-op, as is an empty name.
        assert_eq!(rebrand_ui_text_as("jcode everything", DEFAULT_APP_NAME), "jcode everything");
        assert_eq!(rebrand_ui_text_as("jcode everything", ""), "jcode everything");
    }

    #[test]
    fn the_cached_accessor_is_a_no_op_for_the_stock_name() {
        // Reads the cache; no other test here sets the variable to a custom
        // value at the same time, and this asserts the stock-name path.
        let name = app_name();
        if name == DEFAULT_APP_NAME {
            assert_eq!(rebrand_ui_text("jcode as usual"), "jcode as usual");
        } else {
            assert_eq!(rebrand_ui_text("jcode as usual"), format!("{name} as usual"));
        }
    }
}

//! Application-name helpers for the UI.
//!
//! Thin re-export of the shared branding module so UI code can write
//! `brand::app_name()` instead of reaching into the style crate. Downstream
//! products that embed this TUI set `JCODE_APP_NAME` before startup; see
//! `jcode_tui_style::branding` for the rules (and for what deliberately is not
//! rebranded, such as vendor and service names).

pub(crate) use jcode_tui_style::branding::{app_name, app_name_title};

//! Built-in protocol handlers.
//!
//! Each submodule is gated on a feature flag so users can keep the dependency
//! surface minimal. Use [`register_defaults`] to install all enabled handlers
//! into a [`crate::handler::HandlerRegistry`] in one call.

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;

use crate::handler::HandlerRegistry;

/// Register every built-in handler currently enabled by feature flags.
pub fn register_defaults(registry: &HandlerRegistry) {
    // `registry` is unused when no built-in features are enabled.
    let _ = registry;
    #[cfg(feature = "openai")]
    openai::register(registry);
    #[cfg(feature = "anthropic")]
    anthropic::register(registry);
}

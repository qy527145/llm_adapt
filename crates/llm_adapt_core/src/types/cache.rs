//! 提示词缓存标记。
//!
//! 定义在 IR 层（[`crate::protocol::ir`]）—— 它是 wire 概念，不是角色感知层的概念。
//! 这里只做再导出，保持 `llm_adapt_core::CacheMarker` 这条既有路径可用。

pub use crate::protocol::ir::{CacheMarker, CacheTtl};

//! 工具执行结果的正文。
//!
//! 定义在 IR 层（[`crate::protocol::ir`]）—— 三种形态直接对应 wire 上的写法，
//! 归一会造成无谓的 wire 差异。这里只做再导出。

pub use crate::protocol::ir::{ToolResultBlock, ToolResultContent};

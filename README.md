# llm_adapt

A production-grade Rust anti-corruption layer for large-language-model APIs.
Application code talks to one unified type system — vendor handlers translate
to and from each provider's wire format.

> **Phase 1 status.** The core library (`llm_adapt_core`) and the CLI debug
> tool (`llm-adapt`) are functional. Built-in OpenAI- and Anthropic-compatible
> handlers ship out of the box, including streaming, tool use, vision, and
> prompt-cache accounting. The interactive TUI and Web management panel are
> stubbed and land in phase 2.

## Workspace layout

```
llm_adapt/
├── crates/
│   ├── llm_adapt_core/     # Library: types, traits, handlers, HTTP layer
│   └── llm_adapt_cli/      # Binary: `llm-adapt` (CLI today; TUI + Web next)
└── crates/llm_adapt_core/examples/
                            # Runnable examples
```

Dependencies are unidirectional: `llm_adapt_cli` depends on `llm_adapt_core`,
nothing depends back.

## Core concepts

| Concept | Trait / type | Where |
|---|---|---|
| Role-aware history | `Conversation` (system + Turn[User\|Assistant]) | `llm_adapt_core::types::conversation` |
| Unified request | `ChatRequest` | `llm_adapt_core::types::request` |
| Unified response | `ChatResponse` (carries `AssistantMessage`) | `llm_adapt_core::types::response` |
| Streaming events | `StreamChunk` | `llm_adapt_core::types::stream` |
| Cache markers (5m / 1h) | `CacheMarker`, `CacheTtl` | `llm_adapt_core::types::cache` |
| Usage with TTL-split cache writes | `Usage`, `CacheWriteUsage` | `llm_adapt_core::types::usage` |
| Translate request → HTTP | `RequestHandler` | `llm_adapt_core::handler` |
| Translate body → response | `NonStreamResponseHandler` | `llm_adapt_core::handler` |
| Translate byte stream → chunks | `StreamResponseHandler` | `llm_adapt_core::handler` |
| Plug a protocol in | `HandlerRegistry::register_protocol` | `llm_adapt_core::handler::registry` |
| Send the HTTP | `HttpExecutor` | `llm_adapt_core::http` |
| Convenience facade | `LLMClient` | `llm_adapt_core::client` |

The conversation model is *role-aware*: `UserBlock` only accepts text / image /
tool-result content, `AssistantBlock` only accepts text / thinking / tool-call.
The compiler rejects illegal compositions outright.

All three handler traits are **object-safe** so they can be stored as
`Arc<dyn ...>` inside the registry and swapped at runtime.

## Library quick start

```rust
use llm_adapt_core::{ChatRequest, ClientConfig, Conversation, LLMClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LLMClient::new(ClientConfig::new(
        "https://api.openai.com",
        std::env::var("OPENAI_API_KEY")?,
    ))?;

    let request = ChatRequest::openai("gpt-4o-mini", Conversation::single_user("Hello!"));
    let resp = client.chat(&request).await?;

    println!("{}", resp.text());
    Ok(())
}
```

### Prompt caching (Anthropic 5m / 1h)

```rust
use llm_adapt_core::{
    CacheMarker, ChatRequest, Conversation, SystemPrompt, Turn, UserMessage,
};

let conv = Conversation {
    system: Some(SystemPrompt::text("long stable instructions...")
        .with_cache(CacheMarker::ephemeral_1h())),
    turns:  vec![Turn::User(UserMessage::text("first user turn"))],
};
let req = ChatRequest::anthropic("claude-3-5-sonnet-20241022", conv);
// After `client.chat(&req).await?`, `resp.usage.cache.write.ephemeral_1h`
// tells you how many tokens went into the 1-hour cache tier.
```

Run the bundled examples:

```bash
cargo run -p llm_adapt_core --example basic_openai
cargo run -p llm_adapt_core --example basic_anthropic
cargo run -p llm_adapt_core --example streaming
cargo run -p llm_adapt_core --example tool_calling
cargo run -p llm_adapt_core --example custom_handler   # no key required
```

## CLI

The `llm-adapt` binary is the debug entry point. Everything it does runs
through the same core library and shares a single config file
(`~/.llm-adapt/config.toml`, override with `LLM_ADAPT_HOME`).

```bash
# Set up an OpenAI profile (the api_key may be a literal value or `env:VAR_NAME`).
llm-adapt config set openai api_key env:OPENAI_API_KEY
llm-adapt config set openai default_model gpt-4o-mini
llm-adapt config use openai

# See what protocols are registered.
llm-adapt handlers list
llm-adapt handlers capabilities

# Render a request without sending it — secrets are masked.
llm-adapt preview --prompt "Hello!" --format curl

# Send it for real.
llm-adapt call --prompt "Hello!"
llm-adapt call --prompt "Count to five." --stream

# Anthropic with explicit 1-hour cache on the system prompt.
llm-adapt call --prompt "Hi" --system "long stable instructions" --cache-system 1h

# Anything machine-readable via --json.
llm-adapt config list --json
llm-adapt handlers capabilities --json
```

`llm-adapt tui` and `llm-adapt web` currently print a notice; their
implementations arrive in phase 2.

### CLI command reference

| Command | What it does |
|---|---|
| `config list` / `show` / `path` | Inspect on-disk profiles |
| `config set <profile> <key> <value>` | Edit a single field |
| `config use <name>` | Switch the active profile |
| `config remove <name>` | Delete a profile |
| `config import <file>` / `export [--out file]` | Bulk read/write |
| `preview --prompt … [--format curl\|json]` | Render request only |
| `call --prompt … [--stream] [--system …]` | Send request |
| `handlers list` | List registered protocol handlers |
| `handlers capabilities` | Show per-model capability metadata |
| `tui` | (phase 2) interactive TUI |
| `web [--host …] [--port …]` | (phase 2) web management panel |

All commands take `--json` for machine-readable output.

## Feature flags

`llm_adapt_core` has two opt-out feature flags:

| Feature | Effect | Default |
|---|---|---|
| `openai`    | Registers `openai_compat` handlers (OpenAI / OpenAI-compatible providers) | on |
| `anthropic` | Registers `anthropic_v2` handlers (Anthropic Messages API) | on |

Build with `--no-default-features` to ship only your own custom handlers.

## Designing a custom handler

```rust
use bytes::Bytes;
use llm_adapt_core::{
    ChatRequest, ChatResponse, ClientConfig, HttpRequest, HttpMethod, LLMError,
    NonStreamResponseHandler, RequestHandler, HandlerRegistry,
};

struct MyVendorRequest;
impl RequestHandler for MyVendorRequest {
    fn build_request(&self, req: &ChatRequest, cfg: &ClientConfig)
        -> Result<HttpRequest, LLMError>
    { /* ... shape a vendor-specific JSON body ... */ todo!() }
}

struct MyVendorParse;
impl NonStreamResponseHandler for MyVendorParse {
    fn parse_response(&self, body: Bytes) -> Result<ChatResponse, LLMError>
    { /* ... unify the vendor response ... */ todo!() }
}

// `StreamResponseHandler` is similar — see `crates/llm_adapt_core/src/handlers/openai.rs`.

let registry = HandlerRegistry::new();
registry.register_protocol(
    "my_vendor_v1",
    MyVendorRequest,
    MyVendorParse,
    /* stream handler */ llm_adapt_core::handlers::openai::OpenAIStreamHandler, // placeholder
);
```

Existing built-in handlers under
[`crates/llm_adapt_core/src/handlers/`](crates/llm_adapt_core/src/handlers/)
are the canonical reference implementations.

## Roadmap

* **Phase 1 (this release)** — core types, Handler traits, registry, HTTP
  layer with retry + hooks, OpenAI/Anthropic built-ins, CLI with
  `config/preview/call/handlers`, runnable examples.
* **Phase 2** — ratatui TUI, axum Web backend, pre-built SPA bundled via
  `rust-embed`, request history & template store shared across all three
  surfaces.

## License

MIT — see [LICENSE](LICENSE).

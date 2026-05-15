//! Provider integrations for LangGraph.
//!
//! This crate provides concrete implementations of `BaseChatModel`
//! for popular LLM providers.
//!
//! # Supported Providers
//! - **OpenAI** — GPT-4o, GPT-4, o1, DeepSeek, and OpenAI-compatible endpoints
//! - **Anthropic** — Claude models (Claude 4, Claude 3.5, etc.)
//!
//! # Example
//! ```rust,no_run
//! use std::sync::Arc;
//! use langgraph_providers::openai::{OpenAIModel, OpenAIModelConfig};
//! use langgraph_prebuilt::{create_react_agent, ReActAgentConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let model = OpenAIModel::new(OpenAIModelConfig {
//!     model: "gpt-4o".to_string(),
//!     api_key: std::env::var("OPENAI_API_KEY")?,
//!     ..Default::default()
//! });
//!
//! let agent = create_react_agent(
//!     Arc::new(model),
//!     vec![], // tools
//!     Some(ReActAgentConfig {
//!         system_prompt: Some("You are a helpful assistant.".to_string()),
//!         ..Default::default()
//!     }),
//! )?;
//! # Ok(())
//! # }
//! ```

mod common;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;

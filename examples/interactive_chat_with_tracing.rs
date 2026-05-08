use std::io::{self, Write};
use std::sync::Arc;
use serde_json::{json, Value as JsonValue};
use dotenvy::dotenv;
use langgraph::prelude::*;
use langgraph_checkpoint::checkpoint::memory::InMemorySaver;
use langgraph_checkpoint::config::RunnableConfigExt;
use langgraph_derive::{tool, StateGraph};
use langgraph_prebuilt::{
    prepare_tools, stream_llm, stream_and_print, tools_condition, BaseChatModel, Message,
    ToolNode,
};
use langgraph_providers::openai::{OpenAIModel, OpenAIModelConfig};
use serde::{Deserialize, Serialize};

// Tracing imports
use langgraph_tracing::{
    EventBus, InMemoryTracingStore, TraceStatus, TracingChatModel, 
    TracingGraphObserver,
};

fn load_openai_config() -> (String, Option<String>, String) {
    dotenv().ok();
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env or environment");
    let api_base = std::env::var("OPENAI_API_BASE").ok();
    let model_name = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
    (api_key, api_base, model_name)
}

// -------------------------------------------------------
// Tools
// -------------------------------------------------------

#[tool("multiply", "Multiply two integers a and b")]
fn multiply(a: i64, b: i64) -> Result<i64, String> {
    a.checked_mul(b).ok_or_else(|| "Multiplication overflow".to_string())
}

#[tool("add", "Add two integers a and b")]
fn add(a: i64, b: i64) -> Result<i64, String> {
    a.checked_add(b).ok_or_else(|| "Addition overflow".to_string())
}

#[tool("get_weather", "Get the current weather for a location")]
fn get_weather(location: String) -> Result<String, String> {
    Ok(format!("Weather for {}: sunny, 22°C", location))
}

// -------------------------------------------------------
// State
// -------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default, StateGraph)]
struct GraphState {
    #[channel(messages)]
    messages: Vec<Message>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("  Interactive Chat with Real-time Tracing");
    println!("========================================");
    println!("  1. Starting Tracing Server...");
    
    // Initialize tracing components
    let store = Arc::new(InMemoryTracingStore::new());
    let event_bus = EventBus::new();

    // Start tracing server in background
    let server_store = store.clone();
    let server_bus = event_bus.clone();
    tokio::spawn(async move {
        langgraph_tracing::server::start(
            "127.0.0.1:3333",
            server_store,
            server_bus,
            Some("crates/langgraph-tracing/frontend/dist"),
        )
        .await
        .unwrap();
    });

    println!("  2. Tracing UI available at http://127.0.0.1:3333");
    println!("  3. Type 'quit' to exit.\n");

    // Prepare tools
    let prepared = prepare_tools(vec![
        Arc::new(Multiply::new()),
        Arc::new(Add::new()),
        Arc::new(GetWeather::new()),
    ]);

    // Create base model
    let (api_key, api_base, model_name) = load_openai_config();
    let base_model = Arc::new(OpenAIModel::new(OpenAIModelConfig {
        model: model_name,
        api_key,
        api_base,
        temperature: Some(0.0),
        ..Default::default()
    }));

    // Build graph
    let channels = GraphState::create_channels();
    let mut graph = StateGraph::new(channels);

    // Node: LLM Call with dynamic tracing wrapper
    let model_arc = base_model.clone();
    let store_clone = store.clone();
    let bus_clone = event_bus.clone();
    let tool_defs = prepared.tool_defs.clone();
    
    graph.add_node("llm_call", move |input: JsonValue, config: RunnableConfig| {
        let model = model_arc.clone();
        let store = store_clone.clone();
        let bus = bus_clone.clone();
        let tool_defs = tool_defs.clone();
        
        async move {
            // Get current trace_id from config
            let trace_id = config.get_configurable()
                .and_then(|c| c.get("trace_id"))
                .and_then(|v| v.as_str())
                .unwrap_or("default")
                .to_string();

            // Wrap model with tracing for THIS call
            let tracing_model = TracingChatModel::new(
                model.bind_tools(tool_defs),
                store,
                bus,
                trace_id
            );

            stream_llm(
                &tracing_model,
                &input,
                "You are a helpful assistant with math and weather tools.",
            )
            .await
        }
    })?;

    // Node: Tools with tracing wrapper
    // Note: To trace tools correctly, we'd need to wrap them too.
    // For simplicity, we'll focus on LLM tracing here.
    let tools_node: Arc<dyn Runnable> = Arc::new(ToolNode::new(prepared.tools.clone()));
    graph.add_node("tool_node", tools_node)?;

    graph.add_edge(START, "llm_call")?;
    conditional_edges!(graph, "llm_call", tools_condition, "tools" => "tool_node", END => END)?;
    graph.add_edge("tool_node", "llm_call")?;

    let checkpointer = Arc::new(InMemorySaver::new());
    let app = graph.compile_builder().checkpointer(checkpointer).build()?;

    // Interactive loop
    let stdin = io::stdin();
    let mut turn = 0u32;
    let mut observer = TracingGraphObserver::new(store.clone(), event_bus.clone());

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input_line = String::new();
        if stdin.read_line(&mut input_line)? == 0 { break; }
        let input_line = input_line.trim();

        if input_line.eq_ignore_ascii_case("quit") || input_line.eq_ignore_ascii_case("exit") {
            println!("Goodbye!");
            break;
        }
        if input_line.is_empty() { continue; }

        turn += 1;
        println!("\n--- Turn {} ---", turn);

        let input = json!({
            "messages": [{"type": "human", "content": input_line}]
        });

        // 1. Start a new trace for this turn
        let trace_id = observer.on_graph_start("interactive_chat_turn", input.clone());

        // 2. Prepare config with trace_id
        let mut config = RunnableConfig::new();
        config.insert("configurable".to_string(), json!({
            "thread_id": "interactive-session",
            "trace_id": trace_id
        }));

        // 3. Execute with streaming
        let mut stream = app.astream(&input, &config, vec![StreamMode::Custom, StreamMode::Updates]);
        
        print!("Assistant: ");
        let collected_text = stream_and_print(&mut stream, false).await;
        println!("\n");

        // 4. End the trace
        let output = json!({
            "messages": [{"type": "ai", "content": collected_text}]
        });
        observer.on_graph_end(&trace_id, output, TraceStatus::Success);
    }

    Ok(())
}

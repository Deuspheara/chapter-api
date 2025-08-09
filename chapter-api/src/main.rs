mod models;
mod redis_client;
mod loader;
mod handlers;
mod immersive;
mod immersive_handlers;
mod story_cache;
mod cached_immersive_analyzer;

use anyhow::Result;
use axum::{
    routing::{get, post},
    Router,
};
use std::{env, sync::Arc};
use tower::ServiceBuilder;
use tower_http::{trace::TraceLayer, cors::{CorsLayer, Any}};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use crate::cached_immersive_analyzer::CachedImmersiveAnalyzer;

use crate::{
    redis_client::RedisClient,
    loader::ChapterLoader,
    handlers::{
        get_chapter, get_chapters_batch, health_check, get_stats,
        admin_reload_chapters, admin_flush_chapters, search_chapters,
        AppState, AppStateInner,
    },
    immersive_handlers::{
        get_immersive_chapter, ask_contextual_question, 
        get_emotional_visualization, get_mystery_analysis,
        refresh_story_cache, get_cache_status, get_chapter_analysis_status,
    },
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Configuration
    let redis_url = env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    let chapters_dir = env::var("CHAPTERS_DIR").unwrap_or_else(|_| "./chapters".to_string());
    let port = env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let auto_load = env::var("AUTO_LOAD").unwrap_or_else(|_| "true".to_string()) == "true";
    let rag_service_url = env::var("RAG_SERVICE_URL").unwrap_or_else(|_| "http://127.0.0.1:8001".to_string());

    // Initialize Redis client
    tracing::info!("Connecting to Redis at {}", redis_url);
    tracing::info!("Using RAG service at {}", rag_service_url);
    let redis_client = RedisClient::new(&redis_url)?;
    
    // Create shared HTTP client with appropriate timeouts and keep-alive
    let http_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .connect_timeout(std::time::Duration::from_secs(10))
        .pool_idle_timeout(std::time::Duration::from_secs(90))
        .pool_max_idle_per_host(10)
        .build()?;
    
    let app_state: AppState = Arc::new(AppStateInner {
        redis_client: redis_client.clone(),
        chapters_dir: chapters_dir.clone(),
        cached_analyzer: CachedImmersiveAnalyzer::new(),
        http_client,
    });

    // Load chapters if requested
    if auto_load {
        tracing::info!("Auto-loading chapters from {}", chapters_dir);
        let loader = ChapterLoader::new(redis_client, chapters_dir);
        
        match loader.load_all_chapters().await {
            Ok(count) => tracing::info!("Successfully loaded {} chapters", count),
            Err(e) => {
                tracing::error!("Failed to load chapters: {}", e);
                std::process::exit(1);
            }
        }
    }

    // Wait for RAG service to be available (optional, with timeout)
    let rag_health_url = format!("{}/health", rag_service_url);
    tracing::info!("Checking RAG service availability at {}", rag_health_url);
    
    let mut rag_attempts = 0;
    let max_rag_attempts = 5;
    
    while rag_attempts < max_rag_attempts {
        match app_state.http_client.get(&rag_health_url).send().await {
            Ok(response) if response.status().is_success() => {
                tracing::info!("RAG service is available");
                break;
            }
            Ok(response) => {
                tracing::warn!("RAG service returned status: {}", response.status());
            }
            Err(e) => {
                tracing::warn!("RAG service not yet available: {}", e);
            }
        }
        
        rag_attempts += 1;
        if rag_attempts < max_rag_attempts {
            tracing::info!("Waiting for RAG service... (attempt {}/{})", rag_attempts, max_rag_attempts);
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        }
    }
    
    if rag_attempts >= max_rag_attempts {
        tracing::warn!("RAG service not available after {} attempts. Starting anyway with limited functionality.", max_rag_attempts);
    }

    // Build the application
    let app = Router::new()
        // Public endpoints
        .route("/health", get(health_check))
        .route("/stats", get(get_stats))
        .route("/chapters/{number}", get(get_chapter))
        .route("/chapters", get(get_chapters_batch))
        .route("/search", get(search_chapters))
        
        // Immersive reading endpoints
        .route("/immersive/chapters/{number}", get(get_immersive_chapter))
        .route("/immersive/question", post(ask_contextual_question))
        .route("/immersive/emotions/{number}", get(get_emotional_visualization))
        .route("/immersive/mystery/{number}", get(get_mystery_analysis))
        .route("/immersive/cache/refresh", post(refresh_story_cache))
        .route("/immersive/cache/status", get(get_cache_status))
        .route("/immersive/analysis/status/{number}", get(get_chapter_analysis_status))
        
        // Admin endpoints
        .route("/admin/reload", post(admin_reload_chapters))
        .route("/admin/flush", post(admin_flush_chapters))
        
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(
                    CorsLayer::new()
                        .allow_origin(Any)
                        .allow_methods(Any)
                        .allow_headers(Any)
                )
        )
        .with_state(app_state);

    // Start the server
    let listener = tokio::net::TcpListener::bind(&format!("0.0.0.0:{}", port)).await?;
    tracing::info!("Server starting on port {}", port);
    tracing::info!("🚀 Chapter API Server Ready!");
    tracing::info!("");
    tracing::info!("📚 Available endpoints:");
    tracing::info!("  GET  /health - Health check");
    tracing::info!("  GET  /stats - API statistics");
    tracing::info!("");
    tracing::info!("📖 Chapter endpoints:");
    tracing::info!("  GET  /chapters/{{number}} - Get single chapter with full content");
    tracing::info!("  GET  /chapters/{{number}}?summary=true - Get chapter summary (no paragraphs)");
    tracing::info!("  GET  /chapters?start=1&end=10&limit=50 - Get chapter batch");
    tracing::info!("  GET  /chapters?start=1&end=10&summary=true - Get lightweight batch");
    tracing::info!("");
    tracing::info!("🔍 Search endpoint:");
    tracing::info!("  GET  /search?q=crimson&limit=10 - Search chapters by title (summaries only)");
    tracing::info!("");
    tracing::info!("🎭 Immersive reading endpoints:");
    tracing::info!("  GET  /immersive/chapters/{{number}} - Get chapter with full immersive features");
    tracing::info!("  POST /immersive/question - Ask contextual questions about the story");
    tracing::info!("  GET  /immersive/emotions/{{number}} - Get emotional visualization for chapter");
    tracing::info!("  GET  /immersive/mystery/{{number}} - Get mystery analysis with clues and theories");
    tracing::info!("  POST /immersive/cache/refresh - Refresh the RAG-based story cache");
    tracing::info!("  GET  /immersive/cache/status - Get cache status and statistics");
    tracing::info!("  GET  /immersive/analysis/status/{{number}} - Check background analysis completion status");
    tracing::info!("");
    tracing::info!("🔧 Admin endpoints:");
    tracing::info!("  POST /admin/reload - Reload all chapters from files");
    tracing::info!("  POST /admin/flush - Delete all chapters from Redis");
    tracing::info!("");
    tracing::info!("💡 Features & Notes:");
    tracing::info!("  - Content field excluded from API responses (redundant with paragraphs)");
    tracing::info!("  - Use ?summary=true for faster responses without paragraphs");
    tracing::info!("  - Search returns summaries only for performance");
    tracing::info!("  - Batch requests support up to 200 chapters per request");
    tracing::info!("  - Immersive mode includes: mood detection, voice narration, emotional journey");
    tracing::info!("  - Mystery solver mode: clue tracking, theories, connections");
    tracing::info!("  - Contextual Q&A integrates with RAG system (configured via RAG_SERVICE_URL)");
    tracing::info!("  - Smart cache: RAG-powered analysis with fast lookup performance");
    
    axum::serve(listener, app).await?;
    
    Ok(())
}
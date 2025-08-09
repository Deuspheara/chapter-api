use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::cached_immersive_analyzer::CachedImmersiveAnalyzer;
use crate::models::{ChapterSummary, ChapterBatch, ChapterBatchSummary};
use crate::{redis_client::RedisClient, loader::ChapterLoader};

pub type AppState = Arc<AppStateInner>;

pub struct AppStateInner {
    pub redis_client: RedisClient,
    pub chapters_dir: String,
    pub cached_analyzer: CachedImmersiveAnalyzer,
    pub http_client: reqwest::Client,
}

#[derive(Deserialize)]
pub struct BatchQuery {
    pub start: Option<u32>,
    pub end: Option<u32>,
    pub limit: Option<u32>,
    pub summary: Option<bool>, // New parameter for summary-only responses
}

#[derive(Deserialize)]
pub struct ChapterQuery {
    pub summary: Option<bool>, // New parameter for summary-only responses
}

#[derive(Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message),
        }
    }
}

// Get single chapter (with optional summary mode)
pub async fn get_chapter(
    Path(chapter_number): Path<u32>,
    Query(params): Query<ChapterQuery>,
    State(app_state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match app_state.redis_client.get_chapter(chapter_number) {
        Ok(Some(chapter)) => {
            if params.summary.unwrap_or(false) {
                let summary: ChapterSummary = chapter.into();
                Ok(Json(serde_json::to_value(ApiResponse::success(summary)).unwrap()))
            } else {
                Ok(Json(serde_json::to_value(ApiResponse::success(chapter)).unwrap()))
            }
        }
        Ok(None) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to get chapter {}: {}", chapter_number, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

// Get batch of chapters (with optional summary mode)
pub async fn get_chapters_batch(
    Query(params): Query<BatchQuery>,
    State(app_state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let total_chapters = match app_state.redis_client.get_total_chapters() {
        Ok(total) => total,
        Err(e) => {
            tracing::error!("Failed to get total chapters: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    let start = params.start.unwrap_or(1);
    let limit = params.limit.unwrap_or(50).min(200); // Max 200 chapters per request
    let end = params.end.unwrap_or(start + limit - 1).min(total_chapters);
    let summary_mode = params.summary.unwrap_or(false);

    match app_state.redis_client.get_chapters_batch(start, end) {
        Ok(chapters) => {
            if summary_mode {
                let summaries: Vec<ChapterSummary> = chapters.into_iter().map(|c| c.into()).collect();
                let batch = ChapterBatchSummary {
                    total: summaries.len(),
                    start,
                    end,
                    chapters: summaries,
                };
                Ok(Json(serde_json::to_value(ApiResponse::success(batch)).unwrap()))
            } else {
                let batch = ChapterBatch {
                    total: chapters.len(),
                    start,
                    end,
                    chapters,
                };
                Ok(Json(serde_json::to_value(ApiResponse::success(batch)).unwrap()))
            }
        }
        Err(e) => {
            tracing::error!("Failed to get chapters batch {}-{}: {}", start, end, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

// Health check
pub async fn health_check() -> Json<ApiResponse<String>> {
    Json(ApiResponse::success("API is healthy".to_string()))
}

// Get API stats
#[derive(Serialize)]
pub struct ApiStats {
    pub total_chapters: u32,
    pub redis_connected: bool,
    pub chapters_dir: String,
    pub optimization_notes: Vec<String>,
}

pub async fn get_stats(
    State(app_state): State<AppState>,
) -> Result<Json<ApiResponse<ApiStats>>, StatusCode> {
    let total_chapters = app_state.redis_client.get_total_chapters().unwrap_or(0);
    let redis_connected = app_state.redis_client.get_connection().is_ok();

    let optimization_notes = vec![
        "Content field excluded from API responses (use paragraphs instead)".to_string(),
        "Add ?summary=true to get lightweight responses without paragraphs".to_string(),
        "Batch requests support up to 200 chapters per request".to_string(),
    ];

    let stats = ApiStats {
        total_chapters,
        redis_connected,
        chapters_dir: app_state.chapters_dir.clone(),
        optimization_notes,
    };

    Ok(Json(ApiResponse::success(stats)))
}

// Admin: Reload all chapters
#[derive(Serialize)]
pub struct ReloadResult {
    pub reloaded_count: usize,
    pub message: String,
}

pub async fn admin_reload_chapters(
    State(app_state): State<AppState>,
) -> Result<Json<ApiResponse<ReloadResult>>, StatusCode> {
    tracing::info!("Admin reload chapters requested");
    
    let loader = ChapterLoader::new(
        app_state.redis_client.clone(),
        app_state.chapters_dir.clone(),
    );
    
    match loader.reload_chapters().await {
        Ok(count) => {
            let result = ReloadResult {
                reloaded_count: count,
                message: format!("Successfully reloaded {} chapters", count),
            };
            tracing::info!("Admin reload completed: {} chapters", count);
            Ok(Json(ApiResponse::success(result)))
        }
        Err(e) => {
            tracing::error!("Admin reload failed: {}", e);
            Ok(Json(ApiResponse::error(format!("Reload failed: {}", e))))
        }
    }
}

// Admin: Flush all chapters
#[derive(Serialize)]
pub struct FlushResult {
    pub message: String,
    pub success: bool,
}

pub async fn admin_flush_chapters(
    State(app_state): State<AppState>,
) -> Result<Json<ApiResponse<FlushResult>>, StatusCode> {
    tracing::info!("Admin flush chapters requested");
    
    match app_state.redis_client.flush_chapters() {
        Ok(_) => {
            let result = FlushResult {
                message: "All chapters flushed successfully".to_string(),
                success: true,
            };
            tracing::info!("Admin flush completed");
            Ok(Json(ApiResponse::success(result)))
        }
        Err(e) => {
            tracing::error!("Admin flush failed: {}", e);
            Ok(Json(ApiResponse::error(format!("Flush failed: {}", e))))
        }
    }
}

// Search chapters by title (summary only for performance)
#[derive(Deserialize)]
pub struct SearchQuery {
    pub q: String,
    pub limit: Option<u32>,
}

#[derive(Serialize)]
pub struct SearchResult {
    pub chapters: Vec<ChapterSummary>,
    pub total_found: usize,
    pub query: String,
}

pub async fn search_chapters(
    Query(params): Query<SearchQuery>,
    State(app_state): State<AppState>,
) -> Result<Json<ApiResponse<SearchResult>>, StatusCode> {
    let limit = params.limit.unwrap_or(20).min(100) as usize;
    let query = params.q.to_lowercase();
    
    let total_chapters = match app_state.redis_client.get_total_chapters() {
        Ok(total) => total,
        Err(e) => {
            tracing::error!("Failed to get total chapters: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    let mut found_chapters = Vec::new();
    let batch_size = 100u32;
    
    for start in (1..=total_chapters).step_by(batch_size as usize) {
        let end = (start + batch_size - 1).min(total_chapters);
        
        if let Ok(chapters) = app_state.redis_client.get_chapters_batch(start, end) {
            for chapter in chapters {
                if chapter.title.to_lowercase().contains(&query) {
                    found_chapters.push(chapter.into()); // Convert to summary
                    if found_chapters.len() >= limit {
                        break;
                    }
                }
            }
        }
        
        if found_chapters.len() >= limit {
            break;
        }
    }

    let result = SearchResult {
        total_found: found_chapters.len(),
        query: params.q,
        chapters: found_chapters,
    };

    Ok(Json(ApiResponse::success(result)))
}
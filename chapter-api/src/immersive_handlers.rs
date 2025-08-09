use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use rand::Rng;
use chrono;
use crate::{
    handlers::{AppState, ApiResponse},
    immersive::*,
    cached_immersive_analyzer::CachedImmersiveAnalyzer,
};

// Build RAG query URL from environment (fallback to localhost for local dev)
fn rag_query_url() -> String {
    let base = std::env::var("RAG_SERVICE_URL").unwrap_or_else(|_| "http://127.0.0.1:8001".to_string());
    format!("{}/query", base)
}

// Retry wrapper with exponential backoff and jitter
async fn retry_with_backoff<F, Fut, T, E>(
    mut operation: F,
    max_retries: u32,
    base_delay_ms: u64,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Debug,
{
    for attempt in 0..=max_retries {
        match operation().await {
            Ok(result) => {
                if attempt > 0 {
                    tracing::info!("RAG operation succeeded after {} retries", attempt);
                }
                return Ok(result);
            }
            Err(e) => {
                if attempt == max_retries {
                    tracing::error!("RAG operation failed after {} attempts: {:?}", max_retries + 1, e);
                    return Err(e);
                }
                
                // Exponential backoff with jitter
                let delay = base_delay_ms * (2_u64.pow(attempt));
                let jitter = rand::thread_rng().gen_range(0..=delay / 4);
                let total_delay = delay + jitter;
                
                tracing::warn!("RAG operation failed (attempt {}/{}), retrying in {}ms: {:?}", 
                    attempt + 1, max_retries + 1, total_delay, e);
                
                sleep(Duration::from_millis(total_delay)).await;
            }
        }
    }
    
    unreachable!()
}

// Health check for RAG service
async fn check_rag_service_health(http_client: &reqwest::Client) -> bool {
    let health_url = std::env::var("RAG_SERVICE_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8001".to_string())
        + "/health";
    
    match http_client.get(&health_url).send().await {
        Ok(response) => {
            let is_healthy = response.status().is_success();
            if !is_healthy {
                tracing::warn!("RAG service health check failed: {}", response.status());
            }
            is_healthy
        }
        Err(e) => {
            tracing::warn!("RAG service health check error: {}", e);
            false
        }
    }
}

pub async fn get_immersive_chapter(
    Path(chapter_number): Path<u32>,
    Query(params): Query<ImmersiveModeQuery>,
    State(app_state): State<AppState>,
) -> Result<Json<ApiResponse<ImmersiveModeResponse>>, StatusCode> {
    let start_time = Instant::now();
    
    // 1. FIRST: Get the chapter immediately
    let chapter = match app_state.redis_client.get_chapter(chapter_number) {
        Ok(Some(chapter)) => chapter,
        Ok(None) => return Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to get chapter {}: {}", chapter_number, e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };
    
    tracing::info!("Chapter {} loaded in {:?}", chapter_number, start_time.elapsed());

    // 2. Check cache for already computed analyses
    let cached_analyzer = &app_state.cached_analyzer;
    
    // 3. Start with fast, non-RAG computations that use cached data
    let mood_analysis = cached_analyzer.analyze_chapter_mood_cached(&chapter).await;
    let emotional_journey = analyze_emotional_journey(&chapter);
    let voice_narration = generate_voice_narration_cached(&chapter, &params, cached_analyzer).await;
    
    tracing::info!("Fast analyses completed in {:?}", start_time.elapsed());
    
    // 4. For slow operations (RAG-based), check cache first, then provide fallbacks
    let mystery_elements = match app_state.redis_client.get_cached_mystery_elements(chapter_number) {
        Ok(Some(cached_data)) => {
            match serde_json::from_str::<MysteryElements>(&cached_data) {
                Ok(elements) => {
                    // Check if cached data is empty (from previous failed cache)
                    if elements.clues_found.is_empty() && elements.theories_suggested.is_empty() && 
                       elements.foreshadowing.is_empty() && elements.mystery_tracker.active_mysteries.is_empty() {
                        tracing::info!("Cached mystery elements are empty for chapter {}, using fallback", chapter_number);
                        create_basic_mystery_elements(&chapter)
                    } else {
                        elements
                    }
                },
                Err(_) => {
                    tracing::warn!("Failed to parse cached mystery elements for chapter {}", chapter_number);
                    create_basic_mystery_elements(&chapter)
                }
            }
        }
        _ => {
            tracing::info!("No cached mystery elements for chapter {}, using fallback", chapter_number);
            create_basic_mystery_elements(&chapter)
        }
    };
    
    // 5. Handle recap with immediate fallback
    let quick_recap = if params.include_recap.unwrap_or(false) && chapter_number > 1 {
        // Check cache first
        match app_state.redis_client.get_cached_recap(chapter_number - 1) {
            Ok(Some(cached_recap)) => {
                tracing::info!("Using cached recap for chapter {}", chapter_number - 1);
                match app_state.redis_client.get_chapter(chapter_number - 1) {
                    Ok(Some(prev_ch)) => Some(parse_structured_recap(&cached_recap, &prev_ch)),
                    _ => None
                }
            }
            _ => {
                // Provide immediate basic recap, trigger background generation
                match app_state.redis_client.get_chapter(chapter_number - 1) {
                    Ok(Some(prev_ch)) => {
                        let fallback_recap = create_fallback_recap(&prev_ch);
                        
                        // Spawn background task to generate proper recap
                        let app_state_clone = app_state.clone();
                        tokio::spawn(async move {
                            if let Err(e) = generate_and_cache_recap(&app_state_clone, chapter_number - 1).await {
                                tracing::warn!("Background recap generation failed: {}", e);
                            }
                        });
                        
                        Some(fallback_recap)
                    }
                    _ => None
                }
            }
        }
    } else {
        None
    };
    
    // 6. Spawn background task for expensive mystery analysis if not cached
    if app_state.redis_client.get_cached_mystery_elements(chapter_number).is_err() {
        let app_state_clone = app_state.clone();
        let chapter_clone = chapter.clone();
        tokio::spawn(async move {
            if let Err(e) = generate_and_cache_mystery_analysis(&app_state_clone, &chapter_clone).await {
                tracing::warn!("Background mystery analysis failed: {}", e);
            }
        });
    }
    
    let response = ImmersiveModeResponse {
        chapter,
        mood_analysis,
        voice_narration,
        emotional_journey,
        mystery_elements,
        quick_recap,
    };
    
    tracing::info!("Immersive chapter {} response ready in {:?}", chapter_number, start_time.elapsed());
    Ok(Json(ApiResponse::success(response)))
}

#[axum::debug_handler]
pub async fn ask_contextual_question(
    State(app_state): State<AppState>,
    Json(question_request): Json<ContextualQuestion>,
) -> Result<Json<ApiResponse<ContextualQuestionResponse>>, StatusCode> {
    let start_time = Instant::now();
    
    // Get the chapter if specified
    let chapter = if question_request.current_chapter > 0 {
        let chapter_num = question_request.current_chapter;
        match app_state.redis_client.get_chapter(chapter_num) {
            Ok(Some(chapter)) => Some(chapter),
            Ok(None) => return Err(StatusCode::NOT_FOUND),
            Err(e) => {
                tracing::error!("Failed to get chapter {}: {}", chapter_num, e);
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }
        }
    } else {
        None
    };

    // Use RAG to get contextual answer
    let answer = match query_rag_for_question(&question_request.question, &app_state.http_client).await {
        Ok(rag_response) => {
            if rag_response.answer.trim().is_empty() || rag_response.answer.len() < 10 {
                tracing::warn!("RAG returned empty or very short answer, using fallback");
                generate_fallback_answer(&question_request.question, chapter.as_ref())
            } else {
                rag_response.answer
            }
        },
        Err(e) => {
            tracing::warn!("RAG query failed for contextual question: {}", e);
            generate_fallback_answer(&question_request.question, chapter.as_ref())
        }
    };

    // Find related paragraphs if we have a chapter and current paragraph
    let mut related_paragraphs = Vec::new();
    if let (Some(chapter), Some(current_para)) = (&chapter, question_request.current_paragraph) {
        // Include current paragraph and surrounding context
        related_paragraphs.push(current_para);
        
        // Add previous paragraph if exists
        if current_para > 0 {
            related_paragraphs.push(current_para - 1);
        }
        
        // Add next paragraph if exists
        if current_para < chapter.paragraphs.len() - 1 {
            related_paragraphs.push(current_para + 1);
        }
        
        // Look for paragraphs containing keywords from the question
        let question_words: Vec<&str> = question_request.question.split_whitespace()
            .filter(|word| word.len() > 3)
            .collect();
        
        for (index, paragraph) in chapter.paragraphs.iter().enumerate() {
            if index != current_para && 
               question_words.iter().any(|word| paragraph.to_lowercase().contains(&word.to_lowercase())) {
                related_paragraphs.push(index);
            }
        }
    }

    // Extract character context from the current chapter or question
    let character_context = if let Some(chapter) = &chapter {
        extract_characters_from_chapter(chapter)
    } else {
        // Extract potential character names from the question itself
        extract_character_names_from_text(&question_request.question)
    };

    // Determine plot relevance based on question content
    let plot_relevance = determine_plot_relevance(&question_request.question);

    // Check for potential spoilers
    let spoiler_warning = check_for_spoilers(&question_request.question, &answer);

    // Generate follow-up suggestions based on the question type
    let follow_up_suggestions = generate_follow_up_suggestions(&question_request.question, chapter.as_ref());

    let contextual_response = ContextualQuestionResponse {
        answer,
        related_paragraphs,
        character_context,
        plot_relevance,
        spoiler_warning,
        follow_up_suggestions,
    };

    tracing::info!("Contextual question processed in {:?}", start_time.elapsed());
    Ok(Json(ApiResponse::success(contextual_response)))
}

pub async fn get_emotional_visualization(
    Path(chapter_number): Path<u32>,
    State(app_state): State<AppState>,
) -> Result<Json<ApiResponse<EmotionalVisualization>>, StatusCode> {
    let chapter = match app_state.redis_client.get_chapter(chapter_number) {
        Ok(Some(chapter)) => chapter,
        Ok(None) => return Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to get chapter {}: {}", chapter_number, e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    let cached_analyzer = &app_state.cached_analyzer;
    let mood_analysis = cached_analyzer.analyze_chapter_mood_cached(&chapter).await;
    let visualization = create_emotional_visualization(&mood_analysis, &chapter);

    Ok(Json(ApiResponse::success(visualization)))
}

#[derive(Deserialize)]
pub struct MysteryQuery {
    pub _depth: Option<String>,
    pub _spoiler_tolerance: Option<String>,
}

pub async fn get_mystery_analysis(
    Path(chapter_number): Path<u32>,
    Query(_params): Query<MysteryQuery>,
    State(app_state): State<AppState>,
) -> Result<Json<ApiResponse<MysteryElements>>, StatusCode> {
    let chapter = match app_state.redis_client.get_chapter(chapter_number) {
        Ok(Some(chapter)) => chapter,
        Ok(None) => return Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to get chapter {}: {}", chapter_number, e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    let cached_analyzer = &app_state.cached_analyzer;
    let mystery_elements = cached_analyzer.analyze_mystery_elements_with_cache(&chapter).await;

    Ok(Json(ApiResponse::success(mystery_elements)))
}

#[derive(Debug)]
struct CharacterVoiceAttributes {
    voice_type: String,
    accent: Option<String>,
    tone: String,
}

async fn determine_character_attributes_batch(character: &str, params: &ImmersiveModeQuery) -> CharacterVoiceAttributes {
    // Use simple heuristics based on character name patterns
    let character_lower = character.to_lowercase();
    
    // Determine voice type based on common name patterns and context
    let voice_type = if character_lower.contains("young") || character_lower.contains("klein") || character_lower.contains("child") {
        if character_lower.contains("mr") || character_lower.contains("sir") || !character_lower.contains("miss") && !character_lower.contains("mrs") {
            "young_male"
        } else {
            "young_female"
        }
    } else if character_lower.contains("old") || character_lower.contains("elder") || character_lower.contains("ancient") {
        if character_lower.contains("mr") || character_lower.contains("sir") {
            "elderly_male"
        } else {
            "elderly_female"
        }
    } else if character_lower.contains("captain") || character_lower.contains("professor") || character_lower.contains("doctor") {
        "mature_male"
    } else if character_lower.contains("miss") || character_lower.contains("mrs") || character_lower.contains("lady") {
        "mature_female"
    } else {
        // Default based on common patterns
        "neutral"
    };
    
    // Determine accent based on name origins or titles
    let accent = if character_lower.contains("sir") || character_lower.contains("lord") || character_lower.contains("duke") {
        Some("aristocratic".to_string())
    } else if character_lower.contains("professor") || character_lower.contains("doctor") || character_lower.contains("scholar") {
        Some("formal_educated".to_string())
    } else if character.chars().any(|c| !c.is_ascii()) {
        Some("foreign".to_string())
    } else {
        None
    };
    
    // Determine tone based on character role hints
    let tone = if character_lower.contains("captain") || character_lower.contains("leader") {
        "confident"
    } else if character_lower.contains("scholar") || character_lower.contains("professor") || character_lower.contains("analyst") {
        "analytical"
    } else if character_lower.contains("cheerful") || character_lower.contains("happy") {
        "cheerful"
    } else if character_lower.contains("serious") || character_lower.contains("stern") {
        "serious"
    } else if character_lower.contains("formal") || character_lower.contains("proper") {
        "formal"
    } else {
        "neutral"
    };
    
    // Consider voice preferences from params if available
    let final_voice_type = if let Some(voice_prefs) = &params.voice_preferences {
        // If user has specific preferences, lean towards those
        if voice_prefs.narrator_voice_type.contains("formal") && accent.is_none() {
            voice_type.to_string()
        } else {
            voice_type.to_string()
        }
    } else {
        voice_type.to_string()
    };
    
    CharacterVoiceAttributes {
        voice_type: final_voice_type,
        accent,
        tone: tone.to_string(),
    }
}

async fn generate_voice_narration_cached(chapter: &crate::models::Chapter, params: &ImmersiveModeQuery, cached_analyzer: &CachedImmersiveAnalyzer) -> VoiceNarration {
    let mut character_voices = HashMap::new();
    let mut emphasis_points = Vec::new();
    let mut audio_cues = Vec::new();

    let characters = cached_analyzer.extract_characters_with_cache(chapter).await;
    
    for character in characters {
        if let Some(voice_profile) = cached_analyzer.get_chapter_adapted_voice(&character, chapter).await {
            character_voices.insert(character.clone(), CharacterVoice {
                voice_type: voice_profile.voice_type,
                accent: voice_profile.accent,
                tone: voice_profile.tone,
                speaking_pace: voice_profile.speaking_pace,
                emotional_range: voice_profile.emotional_range,
            });
        } else if let Some(voice_profile) = cached_analyzer.get_character_voice_with_cache(&character).await {
            character_voices.insert(character.clone(), CharacterVoice {
                voice_type: voice_profile.voice_type,
                accent: voice_profile.accent,
                tone: voice_profile.tone,
                speaking_pace: voice_profile.speaking_pace,
                emotional_range: voice_profile.emotional_range,
            });
        } else {
            // Batch RAG queries for uncached characters to reduce API calls
            let voice_attributes = determine_character_attributes_batch(&character, params).await;
            
            character_voices.insert(character.clone(), CharacterVoice {
                voice_type: voice_attributes.voice_type,
                accent: voice_attributes.accent,
                tone: voice_attributes.tone,
                speaking_pace: "normal".to_string(),
                emotional_range: vec!["calm".to_string(), "excited".to_string(), "worried".to_string()],
            });
        }
    }

    for (index, paragraph) in chapter.paragraphs.iter().enumerate() {
        if is_dialogue(paragraph) {
            emphasis_points.push(EmphasisPoint {
                paragraph_index: index,
                text_segment: paragraph.clone(),
                emphasis_type: "dialogue".to_string(),
                reason: "Character speaking".to_string(),
            });
        }
        
        if contains_action(paragraph) {
            audio_cues.push(AudioCue {
                paragraph_index: index,
                cue_type: determine_audio_cue_type(paragraph),
                description: describe_audio_cue(paragraph),
                timing: "during".to_string(),
            });
        }
    }

    let narrator_voice = params.voice_preferences
        .as_ref()
        .map(|v| v.narrator_voice_type.clone())
        .unwrap_or_else(|| "neutral".to_string());

    VoiceNarration {
        narrator_voice,
        character_voices,
        reading_pace: determine_reading_pace(&chapter.title),
        emphasis_points,
        audio_cues,
    }
}

fn analyze_emotional_journey(chapter: &crate::models::Chapter) -> EmotionalJourney {
    let mood_analysis = ImmersiveModeAnalyzer::analyze_chapter_mood(chapter);
    let character_emotions = HashMap::new();
    let mut tension_graph = Vec::new();
    let mut emotional_moments = Vec::new();

    for paragraph_mood in &mood_analysis.paragraph_moods {
        tension_graph.push(TensionPoint {
            paragraph_index: paragraph_mood.paragraph_index,
            tension_level: paragraph_mood.tension_level,
            tension_type: "general".to_string(),
            description: format!("Tension level: {:.2}", paragraph_mood.tension_level),
        });
        
        if paragraph_mood.intensity > 0.6 {
            emotional_moments.push(EmotionalMoment {
                paragraph_index: paragraph_mood.paragraph_index,
                moment_type: paragraph_mood.mood.clone(),
                intensity: paragraph_mood.intensity,
                description: format!("Intense {} moment", paragraph_mood.mood),
                suggested_pause: paragraph_mood.intensity > 0.8,
            });
        }
    }

    let emotional_waypoints: Vec<EmotionalWaypoint> = mood_analysis.paragraph_moods
        .iter()
        .enumerate()
        .filter(|(_, mood)| mood.intensity > 0.5)
        .map(|(i, mood)| EmotionalWaypoint {
            paragraph_index: i,
            emotion: mood.mood.clone(),
            intensity: mood.intensity,
            significance: if mood.intensity > 0.8 { "high".to_string() } else { "medium".to_string() },
        })
        .collect();

    let chapter_arc = EmotionalArc {
        start_emotion: mood_analysis.paragraph_moods.first()
            .map(|m| m.mood.clone())
            .unwrap_or_else(|| "neutral".to_string()),
        end_emotion: mood_analysis.paragraph_moods.last()
            .map(|m| m.mood.clone())
            .unwrap_or_else(|| "neutral".to_string()),
        arc_type: determine_arc_type(&mood_analysis),
        complexity: calculate_arc_complexity(&mood_analysis),
        emotional_journey_map: emotional_waypoints,
    };

    let reading_recommendations = ReadingRecommendations {
        suggested_breaks: emotional_moments.iter()
            .filter(|m| m.suggested_pause)
            .map(|m| m.paragraph_index)
            .collect(),
        intense_moments_warning: emotional_moments.iter()
            .filter(|m| m.intensity > 0.8)
            .map(|m| m.paragraph_index)
            .collect(),
        reading_pace: if mood_analysis.mood_intensity > 0.7 { 
            "slow".to_string() 
        } else { 
            "normal".to_string() 
        },
        environment_suggestion: suggest_reading_environment(&mood_analysis.overall_mood),
    };

    EmotionalJourney {
        chapter_arc,
        character_emotions,
        tension_graph,
        emotional_moments,
        reading_recommendations,
    }
}

async fn analyze_mystery_elements_from_rag(chapter: &crate::models::Chapter, http_client: &reqwest::Client) -> MysteryElements {
    
    let mystery_query = format!(
        "Analyze Chapter {}: {} for mystery elements. Respond with JSON structure: \
        {{\"clues\": [{{\"type\": \"object_clue|character_clue|dialogue_clue|location_clue\", \"description\": \"text\", \"importance\": 0.0-1.0, \"paragraph_approx\": number}}], \
        \"foreshadowing\": [{{\"element\": \"text\", \"type\": \"ominous|predictive|subtle\", \"confidence\": 0.0-1.0}}], \
        \"theories\": [{{\"title\": \"theory name\", \"description\": \"explanation\", \"confidence\": 0.0-1.0}}], \
        \"complexity\": 0.0-1.0, \
        \"active_mysteries\": [\"identity_mystery\", \"location_mystery\", etc]}}. \
        Base analysis only on what's explicitly present in this chapter.",
        chapter.number, chapter.title
    );

    let response = http_client
        .post(&rag_query_url())
        .json(&serde_json::json!({ "question": mystery_query }))
        .send()
        .await;

    let (clues_found, foreshadowing, theories_suggested, mystery_complexity, active_mysteries) = match response {
        Ok(resp) => {
            match resp.json::<serde_json::Value>().await {
                Ok(rag_response) => {
                    if let Some(answer) = rag_response["answer"].as_str() {
                        parse_structured_mystery_response(answer, chapter)
                    } else {
                        tracing::warn!("RAG response missing 'answer' field");
                        (Vec::new(), Vec::new(), Vec::new(), 0.0, Vec::new())
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to parse RAG response JSON: {}", e);
                    (Vec::new(), Vec::new(), Vec::new(), 0.0, Vec::new())
                }
            }
        }
        Err(e) => {
            tracing::warn!("RAG request failed for mystery analysis: {}", e);
            (Vec::new(), Vec::new(), Vec::new(), 0.0, Vec::new())
        }
    };

    let mystery_tracker = MysteryTracker {
        active_mysteries,
        resolved_mysteries: Vec::new(),
        new_mysteries: clues_found.iter().map(|c| format!("new_{}", c.clue_type)).collect(),
        mystery_complexity,
    };

    MysteryElements {
        clues_found,
        theories_suggested,
        connections_discovered: Vec::new(),
        foreshadowing,
        mystery_tracker,
    }
}

fn create_empty_mystery_elements() -> MysteryElements {
    MysteryElements {
        clues_found: Vec::new(),
        theories_suggested: Vec::new(),
        connections_discovered: Vec::new(),
        foreshadowing: Vec::new(),
        mystery_tracker: MysteryTracker {
            active_mysteries: Vec::new(),
            resolved_mysteries: Vec::new(),
            new_mysteries: Vec::new(),
            mystery_complexity: 0.0,
        },
    }
}

fn parse_structured_mystery_response(response: &str, _chapter: &crate::models::Chapter) -> (Vec<ClueElement>, Vec<ForeshadowingElement>, Vec<Theory>, f32, Vec<String>) {
    // Try to parse structured JSON response
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(response) {
        let mut clues = Vec::new();
        let mut foreshadowing = Vec::new();
        let mut theories = Vec::new();
        
        // Parse clues
        if let Some(clues_array) = parsed["clues"].as_array() {
            for clue_obj in clues_array {
                clues.push(ClueElement {
                    paragraph_index: clue_obj["paragraph_approx"].as_u64().unwrap_or(0) as usize,
                    clue_type: clue_obj["type"].as_str().unwrap_or("general_clue").to_string(),
                    description: clue_obj["description"].as_str().unwrap_or("Mystery element").to_string(),
                    importance: clue_obj["importance"].as_f64().unwrap_or(0.5) as f32,
                    related_mysteries: Vec::new(),
                    spoiler_level: "unknown".to_string(),
                });
            }
        }
        
        // Parse foreshadowing
        if let Some(foreshadow_array) = parsed["foreshadowing"].as_array() {
            for foreshadow_obj in foreshadow_array {
                foreshadowing.push(ForeshadowingElement {
                    paragraph_index: 0, // JSON doesn't specify, use default
                    element: foreshadow_obj["element"].as_str().unwrap_or("Foreshadowing element").to_string(),
                    foreshadow_type: foreshadow_obj["type"].as_str().unwrap_or("subtle").to_string(),
                    potential_future_relevance: "Future significance".to_string(),
                    confidence: foreshadow_obj["confidence"].as_f64().unwrap_or(0.6) as f32,
                });
            }
        }
        
        // Parse theories
        if let Some(theories_array) = parsed["theories"].as_array() {
            for (i, theory_obj) in theories_array.iter().enumerate() {
                theories.push(Theory {
                    theory_id: format!("structured_theory_{}", i + 1),
                    title: theory_obj["title"].as_str().unwrap_or("Mystery theory").to_string(),
                    description: theory_obj["description"].as_str().unwrap_or("Theory description").to_string(),
                    supporting_clues: Vec::new(),
                    confidence: theory_obj["confidence"].as_f64().unwrap_or(0.5) as f32,
                    spoiler_risk: "unknown".to_string(),
                });
            }
        }
        
        let complexity = parsed["complexity"].as_f64().unwrap_or(0.0) as f32;
        let active_mysteries = if let Some(mysteries_array) = parsed["active_mysteries"].as_array() {
            mysteries_array.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        } else {
            Vec::new()
        };
        
        (clues, foreshadowing, theories, complexity, active_mysteries)
    } else {
        // Fallback to empty response if JSON parsing fails
        tracing::warn!("Failed to parse structured mystery JSON, response: {}", response);
        (Vec::new(), Vec::new(), Vec::new(), 0.0, Vec::new())
    }
}

fn parse_structured_recap(response: &str, chapter: &crate::models::Chapter) -> QuickRecap {
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(response) {
        QuickRecap {
            previous_chapter_summary: parsed["summary"].as_str().unwrap_or(&chapter.title).to_string(),
            key_events: if let Some(events) = parsed["key_events"].as_array() {
                events.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()
            } else {
                vec!["Events from previous chapter".to_string()]
            },
            character_status_updates: if let Some(updates) = parsed["character_updates"].as_array() {
                updates.iter().filter_map(|v| {
                    v.as_str().map(|s| {
                        let parts: Vec<&str> = s.splitn(2, ':').collect();
                        CharacterUpdate {
                            character_name: parts.get(0).unwrap_or(&"Unknown").trim().to_string(),
                            last_known_status: parts.get(1).unwrap_or(&"Status unknown").trim().to_string(),
                            important_development: None,
                            emotional_state: "neutral".to_string(),
                        }
                    })
                }).collect()
            } else {
                Vec::new()
            },
            plot_threads_continuation: if let Some(threads) = parsed["plot_threads"].as_array() {
                threads.iter().filter_map(|v| {
                    v.as_str().map(|s| PlotThread {
                        thread_name: s.to_string(),
                        status: "ongoing".to_string(),
                        last_development: "Continuing from previous chapter".to_string(),
                        continuation_in_current_chapter: true,
                    })
                }).collect()
            } else {
                Vec::new()
            },
            emotional_state_from_previous: parsed["emotional_tone"].as_str().unwrap_or("neutral").to_string(),
        }
    } else {
        tracing::warn!("Failed to parse structured recap JSON: {}", response);
        create_fallback_recap(chapter)
    }
}

// Background processing functions
async fn generate_and_cache_recap(app_state: &AppState, chapter_number: u32) -> Result<(), String> {
    tracing::info!("Starting background recap generation for chapter {}", chapter_number);
    
    match generate_quick_recap(app_state, chapter_number).await {
        Ok(_) => {
            tracing::info!("Background recap generation completed for chapter {}", chapter_number);
            Ok(())
        }
        Err(e) => {
            let error_msg = format!("Background recap generation failed for chapter {}: {}", chapter_number, e);
            tracing::error!("{}", error_msg);
            Err(error_msg)
        }
    }
}

async fn generate_and_cache_mystery_analysis(app_state: &AppState, chapter: &crate::models::Chapter) -> Result<(), String> {
    tracing::info!("Starting background mystery analysis for chapter {}", chapter.number);
    
    // Try RAG first if available, otherwise use fallback
    let mystery_elements = if check_rag_service_health(&app_state.http_client).await {
        let rag_result = analyze_mystery_elements_from_rag(chapter, &app_state.http_client).await;
        
        // Check if RAG returned meaningful data
        if rag_result.clues_found.is_empty() && rag_result.theories_suggested.is_empty() && 
           rag_result.foreshadowing.is_empty() && rag_result.mystery_tracker.active_mysteries.is_empty() {
            tracing::info!("RAG returned empty results, using fallback analysis for chapter {}", chapter.number);
            app_state.cached_analyzer.fallback_mystery_analysis(chapter)
        } else {
            rag_result
        }
    } else {
        tracing::info!("RAG service unavailable, using fallback mystery analysis for chapter {}", chapter.number);
        app_state.cached_analyzer.fallback_mystery_analysis(chapter)
    };
    
    // Cache the results
    if let Ok(serialized) = serde_json::to_string(&mystery_elements) {
        if let Err(e) = app_state.redis_client.cache_mystery_elements(chapter.number, &serialized) {
            tracing::warn!("Failed to cache mystery elements: {}", e);
        } else {
            tracing::info!("Background mystery analysis cached for chapter {}", chapter.number);
        }
    }
    
    Ok(())
}

fn create_basic_mystery_elements(chapter: &crate::models::Chapter) -> MysteryElements {
    // Create basic mystery elements using simple heuristics
    let mut clues = Vec::new();
    let mut theories = Vec::new();
    
    // Simple keyword-based clue detection
    let mystery_keywords = ["strange", "mysterious", "secret", "hidden", "suspicious", "unusual", "weird"];
    let content_lower = chapter.paragraphs.join(" ").to_lowercase();
    
    for (i, keyword) in mystery_keywords.iter().enumerate() {
        if content_lower.contains(keyword) {
            clues.push(ClueElement {
                paragraph_index: 0, // Would need actual paragraph analysis
                clue_type: "observation_clue".to_string(),
                description: format!("Something {} mentioned in this chapter", keyword),
                importance: 0.3,
                related_mysteries: Vec::new(),
                spoiler_level: "low".to_string(),
            });
            
            if i < 2 { // Only add theories for first few clues
                theories.push(Theory {
                    theory_id: format!("basic_theory_{}", i),
                    title: format!("Theory about {}", keyword),
                    description: "Further analysis needed when RAG service is available".to_string(),
                    supporting_clues: Vec::new(),
                    confidence: 0.2,
                    spoiler_risk: "minimal".to_string(),
                });
            }
        }
    }
    
    MysteryElements {
        clues_found: clues,
        theories_suggested: theories,
        connections_discovered: Vec::new(),
        foreshadowing: Vec::new(),
        mystery_tracker: MysteryTracker {
            active_mysteries: vec!["general_plot".to_string()],
            resolved_mysteries: Vec::new(),
            new_mysteries: Vec::new(),
            mystery_complexity: 0.1,
        },
    }
}

fn create_fallback_recap(chapter: &crate::models::Chapter) -> QuickRecap {
    // Create a more informative fallback recap using chapter metadata
    let word_count_desc = if chapter.word_count > 0 {
        format!(" ({} words)", chapter.word_count)
    } else {
        String::new()
    };
    
    let summary = format!(
        "Chapter {}: {}{} - A continuation of the story. Full details available when RAG service is connected.", 
        chapter.number, chapter.title, word_count_desc
    );
    
    QuickRecap {
        previous_chapter_summary: summary,
        key_events: vec![
            format!("Chapter {} events", chapter.number),
            "Story development continues".to_string()
        ],
        character_status_updates: Vec::new(),
        plot_threads_continuation: vec![
            PlotThread {
                thread_name: "Main storyline".to_string(),
                status: "continuing".to_string(),
                last_development: format!("From Chapter {}: {}", chapter.number, chapter.title),
                continuation_in_current_chapter: true,
            }
        ],
        emotional_state_from_previous: "continuing".to_string(),
    }
}

// Add endpoint to check if background processing is complete
pub async fn get_chapter_analysis_status(
    Path(chapter_number): Path<u32>,
    State(app_state): State<AppState>,
) -> Result<Json<ApiResponse<ChapterAnalysisStatus>>, StatusCode> {
    let _cached_analyzer = &app_state.cached_analyzer;
    
    let mystery_cached = app_state.redis_client.get_cached_mystery_elements(chapter_number).is_ok();
    let recap_cached = if chapter_number > 1 {
        app_state.redis_client.get_cached_recap(chapter_number - 1).is_ok()
    } else {
        true // No recap needed for first chapter
    };
    
    let status = ChapterAnalysisStatus {
        chapter_number,
        mystery_analysis_ready: mystery_cached,
        recap_ready: recap_cached,
        last_updated: chrono::Utc::now().to_rfc3339(),
        processing_complete: mystery_cached && recap_cached,
    };
    
    Ok(Json(ApiResponse::success(status)))
}





async fn generate_quick_recap(app_state: &AppState, previous_chapter: u32) -> Result<QuickRecap, Box<dyn std::error::Error + Send + Sync>> {
    let prev_chapter = app_state.redis_client.get_chapter(previous_chapter)?
        .ok_or("Previous chapter not found")?;

    // Check Redis cache first
    if let Ok(Some(cached_recap)) = app_state.redis_client.get_cached_recap(previous_chapter) {
        tracing::info!("Using cached recap for chapter {}", previous_chapter);
        return Ok(parse_structured_recap(&cached_recap, &prev_chapter));
    }

    // Check if RAG service is available before attempting request
    if !check_rag_service_health(&app_state.http_client).await {
        tracing::warn!("RAG service unavailable, returning fallback recap for chapter {}", previous_chapter);
        return Ok(create_fallback_recap(&prev_chapter));
    }

    let recap_query = format!(
        "Generate a brief recap of Chapter {}: {}. Respond with JSON: \
        {{\"summary\": \"brief description\", \"key_events\": [\"event1\", \"event2\"], \
        \"character_updates\": [\"character: status\"], \"plot_threads\": [\"thread1\"], \
        \"emotional_tone\": \"mood\"}}",
        prev_chapter.number, prev_chapter.title
    );

    let operation = || async {
        app_state.http_client
            .post(&rag_query_url())
            .json(&serde_json::json!({
                "question": recap_query,
                "fast_mode": true  // Skip reranking for recaps
            }))
            .send()
            .await
    };

    let response = retry_with_backoff(operation, 3, 1000).await;  // 3 retries, 1 second base

    let recap_text = match response {
        Ok(resp) => {
            match resp.json::<serde_json::Value>().await {
                Ok(json) => {
                    if let Some(answer) = json["answer"].as_str() {
                        tracing::info!("Successfully received RAG recap for chapter {}", previous_chapter);
                        answer.to_string()
                    } else {
                        tracing::warn!("RAG response missing 'answer' field, using fallback");
                        serde_json::to_string(&create_fallback_recap(&prev_chapter))?
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to parse RAG recap JSON: {}, using fallback", e);
                    serde_json::to_string(&create_fallback_recap(&prev_chapter))?
                }
            }
        }
        Err(e) => {
            tracing::warn!("RAG request failed for recap after retries: {}, using fallback", e);
            serde_json::to_string(&create_fallback_recap(&prev_chapter))?
        }
    };

    // Cache the recap permanently (chapters don't change)
    if let Err(e) = app_state.redis_client.cache_chapter_recap(previous_chapter, &recap_text) {
        tracing::warn!("Failed to cache recap: {}", e);
    }

    let recap = parse_structured_recap(&recap_text, &prev_chapter);
    Ok(recap)
}

#[derive(Debug, Serialize, Deserialize)]
struct RagResponse {
    answer: String,
    sources: Vec<RagSource>,
    confidence: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct RagSource {
    text: String,
    chapter_number: u32,
    chapter_title: String,
}

async fn query_rag_system(question: &str, http_client: &reqwest::Client) -> Result<RagResponse, Box<dyn std::error::Error + Send + Sync>> {
    let start_time = Instant::now();
    
    let operation = || async {
        let request_start = Instant::now();
        let response = http_client
            .post(&rag_query_url())
            .json(&serde_json::json!({
                "question": question
            }))
            .send()
            .await?;
        
        let request_time = request_start.elapsed();
        
        let parse_start = Instant::now();
        let rag_response: RagResponse = response.json().await?;
        let parse_time = parse_start.elapsed();
        
        tracing::debug!("RAG request completed - HTTP: {:?}, Parse: {:?}", request_time, parse_time);
        Ok(rag_response)
    };
    
    let result = retry_with_backoff(operation, 2, 500).await; // 2 retries, 500ms base delay
    
    let total_time = start_time.elapsed();
    match &result {
        Ok(response) => tracing::info!(
            "RAG query completed - Total: {:?}, Confidence: {:.1}", 
            total_time, response.confidence
        ),
        Err(e) => tracing::error!("RAG query failed after retries - Total: {:?}, Error: {:?}", total_time, e),
    }
    
    result
}

async fn enhance_with_chapter_context(
    rag_response: RagResponse,
    current_chapter: &crate::models::Chapter,
    question_request: &ContextualQuestion,
) -> ContextualQuestionResponse {
    let mut related_paragraphs = Vec::new();
    let mut character_context = Vec::new();
    
    if let Some(paragraph_index) = question_request.current_paragraph {
        related_paragraphs.push(paragraph_index);
        
        if paragraph_index > 0 {
            related_paragraphs.push(paragraph_index - 1);
        }
        if paragraph_index < current_chapter.paragraphs.len() - 1 {
            related_paragraphs.push(paragraph_index + 1);
        }
    }

    let characters = extract_characters_from_chapter(current_chapter);
    character_context.extend(characters);

    let follow_up_suggestions = vec![
        "What happens next in this chapter?".to_string(),
        "Who are the main characters involved?".to_string(),
        "What is the significance of this event?".to_string(),
    ];

    ContextualQuestionResponse {
        answer: rag_response.answer,
        related_paragraphs,
        character_context,
        plot_relevance: "medium".to_string(),
        spoiler_warning: None,
        follow_up_suggestions,
    }
}

fn create_emotional_visualization(
    mood_analysis: &MoodAnalysis,
    _chapter: &crate::models::Chapter,
) -> EmotionalVisualization {
    let emotion_timeline: Vec<EmotionTimelinePoint> = mood_analysis.paragraph_moods
        .iter()
        .map(|mood| EmotionTimelinePoint {
            paragraph_index: mood.paragraph_index,
            primary_emotion: mood.mood.clone(),
            intensity: mood.intensity,
            secondary_emotions: mood.emotions.clone(),
        })
        .collect();

    let tension_curve: Vec<f32> = mood_analysis.paragraph_moods
        .iter()
        .map(|mood| mood.tension_level)
        .collect();

    let mood_color_map: Vec<MoodColorPoint> = mood_analysis.paragraph_moods
        .iter()
        .map(|mood| {
            let color = match mood.mood.as_str() {
                "fear" => "#FF4444".to_string(),
                "pain" => "#FF8800".to_string(),
                "peaceful" => "#88FF88".to_string(),
                "mysterious" => "#8844FF".to_string(),
                _ => "#CCCCCC".to_string(),
            };
            MoodColorPoint {
                paragraph_index: mood.paragraph_index,
                color,
                intensity: mood.intensity,
            }
        })
        .collect();

    EmotionalVisualization {
        emotion_timeline,
        tension_curve,
        character_emotional_arcs: HashMap::new(),
        mood_color_map,
    }
}

pub async fn refresh_story_cache(
    State(app_state): State<AppState>,
) -> Result<Json<ApiResponse<String>>, StatusCode> {
    let cached_analyzer = &app_state.cached_analyzer;
    cached_analyzer.invalidate_cache().await;
    
    match cached_analyzer.get_or_build_cache().await {
        Ok(_) => Ok(Json(ApiResponse::success("Story cache refreshed successfully".to_string()))),
        Err(e) => {
            tracing::error!("Failed to refresh story cache: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

pub async fn get_cache_status(
    State(app_state): State<AppState>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let cached_analyzer = &app_state.cached_analyzer;
    
    match cached_analyzer.get_or_build_cache().await {
        Ok(cache_data) => {
            let status = serde_json::json!({
                "cache_version": cache_data.version,
                "last_updated": cache_data.last_updated,
                "characters_count": cache_data.characters.character_profiles.len(),
                "mystery_keywords_count": cache_data.story_elements.mystery_keywords.len(),
                "foreshadowing_patterns_count": cache_data.foreshadowing.foreshadowing_patterns.len(),
                "mood_triggers_count": cache_data.mood_patterns.mood_triggers.len()
            });
            Ok(Json(ApiResponse::success(status)))
        }
        Err(e) => {
            tracing::error!("Failed to get cache status: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

fn extract_characters_from_chapter(chapter: &crate::models::Chapter) -> Vec<String> {
    // Generic character detection - no hardcoded names
    let mut characters = Vec::new();
    let content = chapter.paragraphs.join(" ");
    let words: Vec<&str> = content.split_whitespace().collect();
    
    // Look for capitalized name patterns
    for window in words.windows(2) {
        if window.len() == 2 {
            let potential_name = format!("{} {}", window[0], window[1]);
            if looks_like_character_name(&potential_name) {
                characters.push(potential_name);
            }
        }
    }
    
    // Remove duplicates
    characters.sort();
    characters.dedup();
    
    characters
}

fn looks_like_character_name(text: &str) -> bool {
    text.chars().all(|c| c.is_alphabetic() || c.is_whitespace()) && 
    text.chars().any(|c| c.is_uppercase()) &&
    !text.to_lowercase().contains("the") &&
    !text.to_lowercase().contains("and") &&
    text.len() > 3 && text.len() < 30
}

async fn determine_voice_type_from_cache(character: &str, cached_analyzer: &CachedImmersiveAnalyzer, http_client: &reqwest::Client) -> String {
    if let Some(voice) = cached_analyzer.get_character_voice_with_cache(character).await {
        voice.voice_type
    } else {
        // Fallback: use RAG to determine voice type
        determine_voice_type_from_rag(character, http_client).await.unwrap_or_else(|| "neutral".to_string())
    }
}

async fn determine_accent_from_cache(character: &str, cached_analyzer: &CachedImmersiveAnalyzer, http_client: &reqwest::Client) -> Option<String> {
    if let Some(voice) = cached_analyzer.get_character_voice_with_cache(character).await {
        voice.accent
    } else {
        // Fallback: use RAG to determine accent
        determine_accent_from_rag(character, http_client).await
    }
}

async fn determine_character_tone_from_cache(character: &str, cached_analyzer: &CachedImmersiveAnalyzer, http_client: &reqwest::Client) -> String {
    if let Some(voice) = cached_analyzer.get_character_voice_with_cache(character).await {
        voice.tone
    } else {
        // Fallback: use RAG to determine tone
        determine_tone_from_rag(character, http_client).await.unwrap_or_else(|| "neutral".to_string())
    }
}

async fn determine_voice_type_from_rag(character: &str, http_client: &reqwest::Client) -> Option<String> {
    let query = format!(
        "Based on the available text, analyze the voice type for character '{}'. \
         Consider age, gender, and voice characteristics from dialogue/descriptions. \
         Respond with JSON: {{\"voice_type\": \"young_male|young_female|mature_male|mature_female|elderly_male|elderly_female|child|neutral\", \"reasoning\": \"brief explanation\"}}",
        character
    );

    let response = http_client
        .post(&rag_query_url())
        .json(&serde_json::json!({ "question": query }))
        .send()
        .await
        .ok()?;

    let rag_response: serde_json::Value = response.json().await.ok()?;
    let answer = rag_response["answer"].as_str()?;
    
    // Try to parse JSON response
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(answer) {
        parsed["voice_type"].as_str().map(|s| s.to_string())
    } else {
        // Fallback to keyword matching if JSON parsing fails
        let answer_lower = answer.to_lowercase();
        if answer_lower.contains("young") && answer_lower.contains("male") { Some("young_male".to_string()) }
        else if answer_lower.contains("young") && answer_lower.contains("female") { Some("young_female".to_string()) }
        else { Some("neutral".to_string()) }
    }
}

async fn determine_accent_from_rag(character: &str, http_client: &reqwest::Client) -> Option<String> {
    let query = format!(
        "Based on the available text, analyze the accent/speech pattern for character '{}'. \
         Look for evidence in their dialogue and descriptions. \
         Respond with JSON: {{\"accent\": \"formal_educated|casual_street|aristocratic|foreign|regional|none\", \"evidence\": \"quote or description from text\"}}",
        character
    );

    let response = http_client
        .post(&rag_query_url())
        .json(&serde_json::json!({ "question": query }))
        .send()
        .await
        .ok()?;

    let rag_response: serde_json::Value = response.json().await.ok()?;
    let answer = rag_response["answer"].as_str()?;
    
    // Try to parse JSON response
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(answer) {
        let accent = parsed["accent"].as_str()?;
        if accent == "none" { None } else { Some(accent.to_string()) }
    } else {
        // Fallback to keyword matching
        let answer_lower = answer.to_lowercase();
        if answer_lower.contains("formal") || answer_lower.contains("educated") { Some("formal_educated".to_string()) }
        else if answer_lower.contains("aristocratic") { Some("aristocratic".to_string()) }
        else { None }
    }
}

async fn determine_tone_from_rag(character: &str, http_client: &reqwest::Client) -> Option<String> {
    let query = format!(
        "Based on the available text, analyze the typical tone/emotional baseline of character '{}'. \
         Look at their dialogue and behavior patterns. \
         Respond with JSON: {{\"tone\": \"confident|uncertain|analytical|emotional|cheerful|serious|sarcastic|formal|casual|neutral\", \"examples\": [\"quote1\", \"quote2\"]}}",
        character
    );

    let response = http_client
        .post(&rag_query_url())
        .json(&serde_json::json!({ "question": query }))
        .send()
        .await
        .ok()?;

    let rag_response: serde_json::Value = response.json().await.ok()?;
    let answer = rag_response["answer"].as_str()?;
    
    // Try to parse JSON response
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(answer) {
        parsed["tone"].as_str().map(|s| s.to_string())
    } else {
        // Fallback to keyword matching
        let answer_lower = answer.to_lowercase();
        if answer_lower.contains("confident") { Some("confident".to_string()) }
        else if answer_lower.contains("analytical") { Some("analytical".to_string()) }
        else { Some("neutral".to_string()) }
    }
}

fn is_dialogue(paragraph: &str) -> bool {
    // Single check with more robust dialogue detection
    paragraph.contains('"') || paragraph.contains('"') || paragraph.contains('"') || 
    paragraph.contains('—') || paragraph.contains('–') // em dash, en dash
}

fn contains_action(paragraph: &str) -> bool {
    let action_words = ["suddenly", "quickly", "slammed", "rushed", "jumped"];
    let para_lower = paragraph.to_lowercase();
    action_words.iter().any(|word| para_lower.contains(word))
}

fn determine_audio_cue_type(paragraph: &str) -> String {
    let para_lower = paragraph.to_lowercase();
    if para_lower.contains("pain") || para_lower.contains("hurt") {
        "pain_sound".to_string()
    } else if para_lower.contains("slam") {
        "impact_sound".to_string()
    } else {
        "ambient".to_string()
    }
}

fn describe_audio_cue(paragraph: &str) -> String {
    let para_lower = paragraph.to_lowercase();
    if para_lower.contains("pain") {
        "Sound of discomfort or pain".to_string()
    } else if para_lower.contains("slam") {
        "Heavy impact sound".to_string()
    } else {
        "Ambient background sound".to_string()
    }
}

fn determine_reading_pace(title: &str) -> String {
    if title.to_lowercase().contains("crimson") {
        "dramatic".to_string()
    } else {
        "normal".to_string()
    }
}

fn determine_arc_type(mood_analysis: &MoodAnalysis) -> String {
    if mood_analysis.mood_transitions.len() > 3 {
        "complex".to_string()
    } else if mood_analysis.mood_transitions.len() > 1 {
        "developing".to_string()
    } else {
        "steady".to_string()
    }
}

fn calculate_arc_complexity(mood_analysis: &MoodAnalysis) -> f32 {
    (mood_analysis.mood_transitions.len() as f32 * 0.2 + 
     mood_analysis.emotional_peaks.len() as f32 * 0.3).min(1.0)
}

fn suggest_reading_environment(mood: &str) -> String {
    match mood {
        "dark" | "horror" => "Dimly lit room, quiet environment".to_string(),
        "mysterious" => "Comfortable chair with good lighting".to_string(),
        "peaceful" => "Natural lighting, peaceful surroundings".to_string(),
        _ => "Well-lit, comfortable reading space".to_string(),
    }
}

async fn query_rag_for_question(question: &str, http_client: &reqwest::Client) -> Result<RagResponse, Box<dyn std::error::Error + Send + Sync>> {
    let operation = || async {
        let response = http_client
            .post(&rag_query_url())
            .json(&serde_json::json!({
                "question": question,
                "fast_mode": false
            }))
            .send()
            .await?;
        
        let rag_response: RagResponse = response.json().await?;
        Ok(rag_response)
    };
    
    retry_with_backoff(operation, 2, 500).await
}

fn extract_character_names_from_text(text: &str) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut characters = Vec::new();
    
    // Look for capitalized words that might be names
    for window in words.windows(2) {
        if window.len() == 2 {
            let potential_name = format!("{} {}", window[0], window[1]);
            if looks_like_character_name(&potential_name) {
                characters.push(potential_name);
            }
        }
    }
    
    // Also check single capitalized words
    for word in words {
        if word.len() > 3 && word.chars().next().unwrap().is_uppercase() && 
           word.chars().all(|c| c.is_alphabetic()) &&
           !["The", "This", "That", "What", "Where", "When", "How", "Why", "Who"].contains(&word) {
            characters.push(word.to_string());
        }
    }
    
    characters.sort();
    characters.dedup();
    characters.into_iter().take(5).collect() // Limit to 5 most likely characters
}

fn determine_plot_relevance(question: &str) -> String {
    let question_lower = question.to_lowercase();
    
    // High relevance indicators
    let high_relevance_words = ["plot", "story", "what happens", "why", "mystery", "secret", "reveal"];
    let medium_relevance_words = ["character", "who", "relationship", "feeling", "emotion"];
    let low_relevance_words = ["describe", "what", "where", "when", "how"];
    
    if high_relevance_words.iter().any(|word| question_lower.contains(word)) {
        "high".to_string()
    } else if medium_relevance_words.iter().any(|word| question_lower.contains(word)) {
        "medium".to_string()
    } else if low_relevance_words.iter().any(|word| question_lower.contains(word)) {
        "low".to_string()
    } else {
        "medium".to_string()
    }
}

fn check_for_spoilers(question: &str, answer: &str) -> Option<String> {
    let spoiler_indicators = ["future", "later", "next chapter", "end", "conclusion", "outcome", "result"];
    let combined_text = format!("{} {}", question, answer).to_lowercase();
    
    if spoiler_indicators.iter().any(|indicator| combined_text.contains(indicator)) {
        Some("This answer may contain spoilers for future chapters.".to_string())
    } else {
        None
    }
}

fn generate_follow_up_suggestions(question: &str, chapter: Option<&crate::models::Chapter>) -> Vec<String> {
    let question_lower = question.to_lowercase();
    let mut suggestions = Vec::new();
    
    // Question type based suggestions
    if question_lower.contains("who") {
        suggestions.push("What is this character's role in the story?".to_string());
        suggestions.push("How does this character relate to others?".to_string());
    } else if question_lower.contains("what") {
        suggestions.push("Why did this happen?".to_string());
        suggestions.push("What are the consequences of this event?".to_string());
    } else if question_lower.contains("why") {
        suggestions.push("What happens next?".to_string());
        suggestions.push("How does this affect the story?".to_string());
    } else if question_lower.contains("how") {
        suggestions.push("What does this mean for the characters?".to_string());
        suggestions.push("What other similar events occur?".to_string());
    }
    
    // Chapter-specific suggestions
    if let Some(chapter) = chapter {
        suggestions.push(format!("What other important events happen in Chapter {}?", chapter.number));
        suggestions.push("How does this chapter connect to the overall story?".to_string());
        
        if chapter.number > 1 {
            suggestions.push(format!("How does this relate to Chapter {}?", chapter.number - 1));
        }
    }
    
    // Generic fallbacks
    if suggestions.is_empty() {
        suggestions.extend(vec![
            "What happens next in the story?".to_string(),
            "How do the characters develop?".to_string(),
            "What themes are present in this section?".to_string(),
        ]);
    }
    
    suggestions.into_iter().take(3).collect()
}

fn generate_fallback_answer(question: &str, chapter: Option<&crate::models::Chapter>) -> String {
    let question_lower = question.to_lowercase();
    
    // If no chapter is provided, give a generic response
    let Some(chapter) = chapter else {
        return "I need chapter context to answer your question. Please make sure you're asking about a specific chapter.".to_string();
    };
    
    let content = chapter.paragraphs.join(" ");
    let content_lower = content.to_lowercase();
    
    // Character-related questions
    if question_lower.contains("who") && (question_lower.contains("main character") || question_lower.contains("protagonist")) {
        let main_chars = find_main_characters(&content);
        if !main_chars.is_empty() {
            return format!("Based on Chapter {}, the main character appears to be {}. This character is prominently featured throughout the chapter.", 
                chapter.number, main_chars[0]);
        }
    }
    
    if question_lower.contains("who") && question_lower.contains("character") {
        let characters = extract_characters_from_text(&content);
        if !characters.is_empty() {
            return format!("In Chapter {}, the key characters mentioned include: {}. These characters play important roles in the events described.", 
                chapter.number, characters.join(", "));
        }
    }
    
    // Plot and event questions
    if question_lower.contains("what happens") || question_lower.contains("what is happening") {
        let events = find_key_events(&content);
        if !events.is_empty() {
            return format!("Chapter {} covers several key events: {}. These events drive the story forward and develop the plot.", 
                chapter.number, events.join("; "));
        } else {
            return format!("Chapter {} continues the story with character development and plot progression. The chapter contains {} paragraphs of narrative content.", 
                chapter.number, chapter.paragraphs.len());
        }
    }
    
    // "What" questions in general
    if question_lower.starts_with("what") {
        if question_lower.contains("about") {
            return format!("Chapter {} '{}' focuses on character interactions and story development. The chapter contains detailed narrative exploring the ongoing plot and character relationships.", 
                chapter.number, chapter.title);
        }
    }
    
    // "Why" questions
    if question_lower.starts_with("why") {
        if content_lower.contains("because") || content_lower.contains("reason") {
            return "The motivations and reasons behind the events in this chapter are explored through character dialogue and narrative exposition. The story provides context for the characters' actions and decisions.".to_string();
        }
        return "This chapter explores the underlying motivations and causes behind the story events. Character development and plot progression help explain the 'why' behind key story moments.".to_string();
    }
    
    // "How" questions
    if question_lower.starts_with("how") {
        return "The chapter shows character actions and story progression through detailed narrative. The events unfold through dialogue, character interactions, and narrative description.".to_string();
    }
    
    // Location questions
    if question_lower.contains("where") {
        let locations = find_locations(&content);
        if !locations.is_empty() {
            return format!("Chapter {} takes place in several locations: {}. These settings provide the backdrop for the story events.", 
                chapter.number, locations.join(", "));
        }
        return format!("Chapter {} is set in various locations that are important to the story development and character interactions.", chapter.number);
    }
    
    // Emotional or relationship questions
    if question_lower.contains("feel") || question_lower.contains("emotion") || question_lower.contains("relationship") {
        return "This chapter explores character emotions and relationships through dialogue and internal monologue. The emotional development adds depth to the story and character connections.".to_string();
    }
    
    // Plot-related questions
    if question_lower.contains("plot") || question_lower.contains("story") || question_lower.contains("narrative") {
        let events = find_key_events(&content);
        if !events.is_empty() {
            return format!("The plot of Chapter {} develops through several key elements: {}. These narrative threads contribute to the overall story progression and character development.", 
                chapter.number, events.join("; "));
        } else {
            return format!("Chapter {} '{}' advances the plot through character development, dialogue, and narrative progression. The story builds on previous events while introducing new elements that drive the narrative forward.", 
                chapter.number, chapter.title);
        }
    }
    
    // Character development and insight questions
    if question_lower.contains("character") && (question_lower.contains("develop") || question_lower.contains("insight") || 
        question_lower.contains("growth") || question_lower.contains("change") || question_lower.contains("personality")) {
        let characters = extract_characters_from_text(&content);
        if !characters.is_empty() {
            return format!("Chapter {} provides character insights for: {}. The chapter reveals character motivations, relationships, and development through their actions, dialogue, and interactions with other characters.", 
                chapter.number, characters.join(", "));
        } else {
            return format!("Chapter {} offers deep character insights through internal monologue, dialogue, and character actions. The narrative explores character motivations and development as they navigate the story's challenges.", chapter.number);
        }
    }
    
    // Mystery or secret questions
    if question_lower.contains("mystery") || question_lower.contains("secret") {
        if content_lower.contains("mystery") || content_lower.contains("secret") || content_lower.contains("hidden") {
            return "This chapter contains mysterious elements and secrets that are gradually revealed through the narrative. These plot devices add intrigue and drive reader engagement with hidden clues and foreshadowing.".to_string();
        }
    }
    
    // Detailed questions - increased threshold and more specific response
    if question.len() > 150 {
        return format!("Based on the detailed nature of your question about Chapter {} '{}', the story provides rich content exploring character relationships, plot development, and thematic elements. The chapter's {} paragraphs offer comprehensive narrative details that address various aspects of the story.", 
            chapter.number, chapter.title, chapter.paragraphs.len());
    }
    
    // Final fallback
    format!("Chapter {} '{}' contains {} paragraphs of story content. The chapter develops the ongoing narrative through character interactions, dialogue, and plot progression. For more detailed analysis, you might want to focus on specific aspects of the chapter.", 
        chapter.number, chapter.title, chapter.paragraphs.len())
}

fn find_main_characters(content: &str) -> Vec<String> {
    let mut character_mentions: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let words: Vec<&str> = content.split_whitespace().collect();
    
    // Look for potential character names (capitalized words)
    for window in words.windows(2) {
        if window.len() == 2 {
            let potential_name = format!("{} {}", window[0], window[1]);
            if looks_like_character_name(&potential_name) {
                *character_mentions.entry(potential_name).or_insert(0) += 1;
            }
        }
    }
    
    // Also check single names
    for word in &words {
        if looks_like_single_character_name(word) {
            *character_mentions.entry(word.to_string()).or_insert(0) += 1;
        }
    }
    
    // Sort by frequency and return top characters
    let mut characters: Vec<(String, usize)> = character_mentions.into_iter().collect();
    characters.sort_by(|a, b| b.1.cmp(&a.1));
    
    characters.into_iter()
        .take(3)
        .map(|(name, _)| name)
        .collect()
}

fn looks_like_single_character_name(word: &str) -> bool {
    word.len() > 2 && 
    word.len() < 20 &&
    word.chars().next().unwrap_or('a').is_uppercase() &&
    word.chars().all(|c| c.is_alphabetic()) &&
    !["The", "And", "But", "For", "With", "This", "That", "Chapter", "He", "She", "It", "They", "We", "I", "You"].contains(&word)
}

fn find_key_events(content: &str) -> Vec<String> {
    let content_lower = content.to_lowercase();
    let mut events = Vec::new();
    
    // Look for action words and event indicators
    let event_patterns = [
        ("arrived", "arrival"),
        ("left", "departure"),
        ("died", "death"),
        ("killed", "killing"),
        ("discovered", "discovery"),
        ("found", "finding"),
        ("revealed", "revelation"),
        ("decided", "decision"),
        ("attacked", "attack"),
        ("escaped", "escape"),
        ("met", "meeting"),
        ("married", "marriage"),
        ("born", "birth"),
        ("disappeared", "disappearance"),
        ("returned", "return"),
    ];
    
    for (trigger, event_name) in event_patterns {
        if content_lower.contains(trigger) {
            events.push(format!("A {} occurs in this chapter", event_name));
        }
    }
    
    // Look for dialogue as key interactions
    if content.contains('"') {
        events.push("Important dialogue and character interactions take place".to_string());
    }
    
    // Look for time transitions
    if content_lower.contains("later") || content_lower.contains("then") || content_lower.contains("after") {
        events.push("Time progression and sequential events unfold".to_string());
    }
    
    if events.is_empty() {
        events.push("Character development and story progression continue".to_string());
    }
    
    events.into_iter().take(3).collect()
}

fn find_locations(content: &str) -> Vec<String> {
    let content_lower = content.to_lowercase();
    let mut locations = Vec::new();
    
    // Common location words
    let location_words = [
        "room", "house", "city", "town", "street", "road", "building", "castle", "palace",
        "forest", "mountain", "river", "ocean", "beach", "garden", "park", "school", "office",
        "church", "temple", "market", "shop", "restaurant", "hotel", "inn", "tavern"
    ];
    
    for location in location_words {
        if content_lower.contains(location) {
            locations.push(format!("a {}", location));
        }
    }
    
    // Look for "in the" or "at the" patterns for specific locations
    let words: Vec<&str> = content.split_whitespace().collect();
    for i in 0..words.len().saturating_sub(2) {
        if (words[i].to_lowercase() == "in" || words[i].to_lowercase() == "at") && 
           words[i+1].to_lowercase() == "the" {
            if let Some(location) = words.get(i+2) {
                if location.len() > 2 && location.chars().all(|c| c.is_alphabetic()) {
                    locations.push(format!("the {}", location));
                }
            }
        }
    }
    
    locations.into_iter().take(4).collect()
}

fn extract_characters_from_text(content: &str) -> Vec<String> {
    let words: Vec<&str> = content.split_whitespace().collect();
    let mut characters = std::collections::HashSet::new();
    
    // Look for capitalized name patterns
    for window in words.windows(2) {
        if window.len() == 2 {
            let potential_name = format!("{} {}", window[0], window[1]);
            if looks_like_character_name(&potential_name) {
                characters.insert(potential_name);
            }
        }
    }
    
    // Also check single names
    for word in words {
        if looks_like_single_character_name(word) {
            characters.insert(word.to_string());
        }
    }
    
    let mut chars: Vec<String> = characters.into_iter().collect();
    chars.sort();
    chars.into_iter().take(5).collect()
}








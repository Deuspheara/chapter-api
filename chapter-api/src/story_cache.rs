use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryCacheData {
    pub characters: CharacterCache,
    pub story_elements: StoryElementCache,
    pub foreshadowing: ForeshadowingCache,
    pub mood_patterns: MoodPatternCache,
    pub last_updated: u64,
    pub version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterCache {
    pub character_profiles: HashMap<String, CharacterProfile>,
    pub character_aliases: HashMap<String, String>,
    pub voice_mappings: HashMap<String, VoiceProfile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterProfile {
    pub canonical_name: String,
    pub aliases: Vec<String>,
    pub description: String,
    pub personality_traits: Vec<String>,
    pub emotional_baseline: String,
    pub speaking_style: String,
    pub importance_level: f32,
    pub first_appearance: u32,
    pub last_appearance: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceProfile {
    pub voice_type: String,
    pub accent: Option<String>,
    pub tone: String,
    pub speaking_pace: String,
    pub emotional_range: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryElementCache {
    pub important_objects: HashMap<String, ObjectProfile>,
    pub locations: HashMap<String, LocationProfile>,
    pub concepts: HashMap<String, ConceptProfile>,
    pub mystery_keywords: Vec<MysteryKeyword>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectProfile {
    pub name: String,
    pub description: String,
    pub significance: f32,
    pub mystery_level: f32,
    pub first_mentioned: u32,
    pub related_mysteries: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocationProfile {
    pub name: String,
    pub description: String,
    pub mood_influence: String,
    pub significance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptProfile {
    pub name: String,
    pub description: String,
    pub story_impact: f32,
    pub related_themes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MysteryKeyword {
    pub keyword: String,
    pub importance: f32,
    pub context: String,
    pub related_clues: Vec<String>,
    pub spoiler_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeshadowingCache {
    pub foreshadowing_patterns: Vec<ForeshadowingPattern>,
    pub symbolic_elements: HashMap<String, SymbolicElement>,
    pub narrative_threads: Vec<NarrativeThread>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeshadowingPattern {
    pub pattern: String,
    pub significance: f32,
    pub predicted_relevance: String,
    pub confidence: f32,
    pub chapters_mentioned: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicElement {
    pub symbol: String,
    pub meaning: String,
    pub emotional_weight: f32,
    pub story_significance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeThread {
    pub thread_id: String,
    pub title: String,
    pub status: String,
    pub key_chapters: Vec<u32>,
    pub resolution_prediction: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodPatternCache {
    pub mood_triggers: HashMap<String, MoodTrigger>,
    pub emotional_patterns: Vec<EmotionalPattern>,
    pub tension_indicators: Vec<TensionIndicator>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodTrigger {
    pub trigger_words: Vec<String>,
    pub resulting_mood: String,
    pub intensity_modifier: f32,
    pub context_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalPattern {
    pub pattern_name: String,
    pub trigger_sequence: Vec<String>,
    pub emotional_arc: Vec<String>,
    pub intensity_curve: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensionIndicator {
    pub indicator: String,
    pub tension_level: f32,
    pub context: String,
    pub reliability: f32,
}

pub struct StoryCache {
    cache: Arc<RwLock<Option<StoryCacheData>>>,
    rag_client: reqwest::Client,
    build_lock: Arc<Mutex<()>>,
    // Backoff tracking for fallback (version 0) caches to avoid thrashing RAG
    last_build_attempt: Arc<RwLock<Option<u64>>>,
    retry_backoff_secs: u64,
}

impl StoryCache {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
            
        Self {
            cache: Arc::new(RwLock::new(None)),
            rag_client: client,
            build_lock: Arc::new(Mutex::new(())),
            last_build_attempt: Arc::new(RwLock::new(None)),
            retry_backoff_secs: 120, // 2 minutes
        }
    }

    pub async fn get_or_build_cache(&self) -> Result<StoryCacheData, Box<dyn std::error::Error + Send + Sync>> {
        {
            let cache_read = self.cache.read().await;
            if let Some(cache_data) = cache_read.as_ref() {
                if self.is_cache_fresh(cache_data) {
                    return Ok(cache_data.clone());
                }
                // If cache is a fallback (version 0), apply short backoff before rebuilding to avoid thrashing
                if cache_data.version == 0 {
                    let now = chrono::Utc::now().timestamp() as u64;
                    let last_attempt = *self.last_build_attempt.read().await;
                    if let Some(ts) = last_attempt {
                        if now.saturating_sub(ts) < self.retry_backoff_secs {
                            tracing::warn!("Using fallback cache during backoff window; will retry RAG build later");
                            return Ok(cache_data.clone());
                        }
                    }
                }
            }
        }

        // Acquire build lock to prevent duplicate concurrent rebuilds
        let _build_guard = self.build_lock.lock().await;

        // Double-check after acquiring the lock in case another task built it
        {
            let cache_read = self.cache.read().await;
            if let Some(cache_data) = cache_read.as_ref() {
                if self.is_cache_fresh(cache_data) {
                    return Ok(cache_data.clone());
                }
                if cache_data.version == 0 {
                    let now = chrono::Utc::now().timestamp() as u64;
                    let last_attempt = *self.last_build_attempt.read().await;
                    if let Some(ts) = last_attempt {
                        if now.saturating_sub(ts) < self.retry_backoff_secs {
                            tracing::warn!("Using fallback cache during backoff window (post-lock); will retry RAG build later");
                            return Ok(cache_data.clone());
                        }
                    }
                }
            }
        }

        // Record attempt time for backoff
        {
            let mut last = self.last_build_attempt.write().await;
            *last = Some(chrono::Utc::now().timestamp() as u64);
        }

        let new_cache = self.build_cache_from_rag().await?;
        {
            let mut cache_write = self.cache.write().await;
            *cache_write = Some(new_cache.clone());
        }

        Ok(new_cache)
    }

    async fn build_cache_from_rag(&self) -> Result<StoryCacheData, Box<dyn std::error::Error + Send + Sync>> {
        tracing::info!("Building story cache from RAG...");

        // Build cache components individually with fallbacks
        let characters = match self.build_character_cache().await {
            Ok(cache) => cache,
            Err(e) => {
                tracing::error!("Failed to build character cache, using fallback: {}", e);
                self.create_fallback_character_cache()
            }
        };

        let story_elements = match self.build_story_element_cache().await {
            Ok(cache) => cache,
            Err(e) => {
                tracing::error!("Failed to build story element cache, using fallback: {}", e);
                StoryElementCache {
                    important_objects: HashMap::new(),
                    locations: HashMap::new(),
                    concepts: HashMap::new(),
                    mystery_keywords: Vec::new(),
                }
            }
        };

        let foreshadowing = match self.build_foreshadowing_cache().await {
            Ok(cache) => cache,
            Err(e) => {
                tracing::error!("Failed to build foreshadowing cache, using fallback: {}", e);
                ForeshadowingCache {
                    foreshadowing_patterns: Vec::new(),
                    symbolic_elements: HashMap::new(),
                    narrative_threads: Vec::new(),
                }
            }
        };

        let mood_patterns = match self.build_mood_pattern_cache().await {
            Ok(cache) => cache,
            Err(e) => {
                tracing::error!("Failed to build mood pattern cache, using fallback: {}", e);
                self.create_fallback_mood_cache()
            }
        };

        tracing::info!("Story cache built successfully with {} characters, {} objects, {} patterns, {} mood triggers",
            characters.character_profiles.len(),
            story_elements.important_objects.len(),
            foreshadowing.foreshadowing_patterns.len(),
            mood_patterns.mood_triggers.len()
        );

        // Determine if the cache has meaningful mystery signal.
        // If it does not, mark version as 0 so it is considered stale and retried soon.
        let has_mystery_signal =
            !story_elements.mystery_keywords.is_empty()
            || !foreshadowing.foreshadowing_patterns.is_empty()
            || !foreshadowing.narrative_threads.is_empty();

        if !has_mystery_signal {
            tracing::warn!(
                "Story cache contains no mystery signals (keywords/patterns/threads empty). Marking as fallback (version=0)."
            );
        }

        Ok(StoryCacheData {
            characters,
            story_elements,
            foreshadowing,
            mood_patterns,
            last_updated: chrono::Utc::now().timestamp() as u64,
            version: if has_mystery_signal { 1 } else { 0 },
        })
    }

    async fn build_character_cache(&self) -> Result<CharacterCache, Box<dyn std::error::Error + Send + Sync>> {
        let character_query = "Based on the available chapters, identify the main characters. Respond with JSON: \
        {\"characters\": [{\"name\": \"character name\", \"aliases\": [\"alt names\"], \
        \"description\": \"brief description\", \"traits\": [\"trait1\"], \
        \"speaking_style\": \"formal/casual/etc\", \"voice_type\": \"young_male/mature_female/etc\", \
        \"accent\": \"formal_educated/none/etc\", \"tone\": \"confident/neutral/etc\"}]}. \
        Base analysis only on what appears in the text.";
        
        let response = match self.query_rag(character_query).await {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("Failed to get character data from RAG: {}", e);
                return Ok(self.create_fallback_character_cache());
            }
        };
        
        let mut character_profiles = HashMap::new();
        let mut character_aliases = HashMap::new();
        let mut voice_mappings = HashMap::new();

        let extracted_chars = self.extract_characters_from_structured_response(&response.answer);
        
        for character in extracted_chars {
            character_profiles.insert(character.canonical_name.clone(), character.clone());
            
            for alias in &character.aliases {
                character_aliases.insert(alias.clone(), character.canonical_name.clone());
            }
            
            let voice_profile = VoiceProfile {
                voice_type: self.determine_voice_type_from_description(&character.description),
                accent: self.determine_accent_from_description(&character.description),
                tone: character.speaking_style.clone(),
                speaking_pace: self.determine_pace_from_personality(&character.personality_traits),
                emotional_range: character.personality_traits.clone(),
            };
            
            voice_mappings.insert(character.canonical_name.clone(), voice_profile);
        }

        if character_profiles.is_empty() {
            tracing::warn!("No characters extracted from RAG response, using fallback");
            let fallback_chars = self.create_fallback_character_cache();
            return Ok(fallback_chars);
        }

        Ok(CharacterCache {
            character_profiles,
            character_aliases,
            voice_mappings,
        })
    }

    async fn build_story_element_cache(&self) -> Result<StoryElementCache, Box<dyn std::error::Error + Send + Sync>> {
        let elements_query = "Based on the available chapters, identify:\n\
            1. Important objects, items, or artifacts that appear frequently or seem significant\n\
            2. Key locations or settings mentioned\n\
            3. Recurring concepts, themes, or mysterious elements\n\
            4. Words or phrases that seem to carry special meaning or create intrigue\n\n\
            Focus on elements that appear multiple times or seem to drive the narrative forward, based only on what's present in the text.";
        let response = self.query_rag(elements_query).await?;
        
        let mut important_objects = HashMap::new();
        let locations = HashMap::new();
        let concepts = HashMap::new();
        let mut mystery_keywords = Vec::new();

        // Parse important objects from RAG response
        self.extract_objects_from_rag(&response.answer, &mut important_objects);
        
        // Parse mystery keywords from RAG response  
        self.extract_mystery_keywords_from_rag(&response.answer, &mut mystery_keywords);

        Ok(StoryElementCache {
            important_objects,
            locations,
            concepts,
            mystery_keywords,
        })
    }

    async fn build_foreshadowing_cache(&self) -> Result<ForeshadowingCache, Box<dyn std::error::Error + Send + Sync>> {
        let foreshadowing_query = "Analyze the available chapters for:\n\
            1. Passages that seem to hint at future events or developments\n\
            2. Symbolic elements or metaphors that might have deeper meaning\n\
            3. Recurring motifs or patterns in the narrative\n\
            4. Unresolved questions or mysteries set up in the text\n\n\
            Base your analysis only on what's explicitly present in the chapters, identifying patterns and hints without speculating beyond the text.";
        let response = self.query_rag(foreshadowing_query).await?;
        
        // Extract all foreshadowing elements from RAG response
        let foreshadowing_patterns = self.extract_foreshadowing_patterns_from_rag(&response.answer);
        let symbolic_elements = self.extract_symbolic_elements_from_rag(&response.answer);  
        let narrative_threads = self.extract_narrative_threads_from_rag(&response.answer);

        Ok(ForeshadowingCache {
            foreshadowing_patterns,
            symbolic_elements,
            narrative_threads,
        })
    }

    async fn build_mood_pattern_cache(&self) -> Result<MoodPatternCache, Box<dyn std::error::Error + Send + Sync>> {
        let mut mood_triggers = HashMap::new();
        
        mood_triggers.insert("pain_cluster".to_string(), MoodTrigger {
            trigger_words: vec!["pain".to_string(), "hurt".to_string(), "agony".to_string(), "suffering".to_string()],
            resulting_mood: "pain".to_string(),
            intensity_modifier: 1.2,
            context_required: false,
        });

        mood_triggers.insert("mystery_cluster".to_string(), MoodTrigger {
            trigger_words: vec!["mysterious".to_string(), "strange".to_string(), "peculiar".to_string(), "unknown".to_string()],
            resulting_mood: "mysterious".to_string(),
            intensity_modifier: 1.0,
            context_required: false,
        });

        mood_triggers.insert("horror_cluster".to_string(), MoodTrigger {
            trigger_words: vec!["blood".to_string(), "death".to_string(), "corpse".to_string(), "horror".to_string()],
            resulting_mood: "horror".to_string(),
            intensity_modifier: 1.3,
            context_required: false,
        });

        let emotional_patterns = vec![
            EmotionalPattern {
                pattern_name: "discovery_shock".to_string(),
                trigger_sequence: vec!["confusion".to_string(), "realization".to_string(), "shock".to_string()],
                emotional_arc: vec!["confused".to_string(), "surprised".to_string(), "overwhelmed".to_string()],
                intensity_curve: vec![0.3, 0.7, 0.9],
            },
        ];

        let tension_indicators = vec![
            TensionIndicator {
                indicator: "suddenly".to_string(),
                tension_level: 0.7,
                context: "indicates abrupt change or surprise".to_string(),
                reliability: 0.8,
            },
            TensionIndicator {
                indicator: "blood".to_string(),
                tension_level: 0.9,
                context: "indicates violence or danger".to_string(),
                reliability: 0.9,
            },
        ];

        Ok(MoodPatternCache {
            mood_triggers,
            emotional_patterns,
            tension_indicators,
        })
    }

    async fn query_rag(&self, question: &str) -> Result<RagResponse, Box<dyn std::error::Error + Send + Sync>> {
        let rag_base = std::env::var("RAG_SERVICE_URL").unwrap_or_else(|_| "http://127.0.0.1:8001".to_string());
        let rag_url = format!("{}/query", rag_base);
        let response = self.rag_client
            .post(&rag_url)
            .json(&serde_json::json!({
                "question": question,
                "fast_mode": true  // Skip reranking for cache building
            }))
            .send()
            .await
            .map_err(|e| {
                tracing::error!("Failed to connect to RAG service: {}", e);
                e
            })?;

        // Check if response is successful before attempting to parse
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            tracing::error!("RAG service returned error {}: {}", status, error_text);
            return Err(format!("RAG service error: {}", status).into());
        }

        let rag_response: RagResponse = response.json().await.map_err(|e| {
            tracing::error!("Failed to parse RAG response: {}", e);
            e
        })?;
        
        Ok(rag_response)
    }

    fn is_cache_fresh(&self, cache_data: &StoryCacheData) -> bool {
        // If the cache is a fallback (version 0), treat it as stale so we keep trying to rebuild
        if cache_data.version == 0 {
            return false;
        }

        let now = chrono::Utc::now().timestamp() as u64;
        let age_hours = (now - cache_data.last_updated) / 3600;
        age_hours < 24
    }

    pub async fn invalidate_cache(&self) {
        let mut cache_write = self.cache.write().await;
        *cache_write = None;
        tracing::info!("Story cache invalidated");
    }

    fn extract_characters_from_structured_response(&self, response: &str) -> Vec<CharacterProfile> {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(response) {
            if let Some(characters_array) = parsed["characters"].as_array() {
                let mut characters = Vec::new();
                
                for char_obj in characters_array {
                    let name = char_obj["name"].as_str().unwrap_or("Unknown").to_string();
                    let mut aliases = vec![name.clone()];
                    
                    if let Some(alias_array) = char_obj["aliases"].as_array() {
                        for alias in alias_array {
                            if let Some(alias_str) = alias.as_str() {
                                aliases.push(alias_str.to_string());
                            }
                        }
                    }
                    
                    let traits = if let Some(traits_array) = char_obj["traits"].as_array() {
                        traits_array.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    } else {
                        vec!["unknown".to_string()]
                    };
                    
                    characters.push(CharacterProfile {
                        canonical_name: name,
                        aliases,
                        description: char_obj["description"].as_str().unwrap_or("Character from story").to_string(),
                        personality_traits: traits,
                        emotional_baseline: char_obj["tone"].as_str().unwrap_or("neutral").to_string(),
                        speaking_style: char_obj["voice_type"].as_str().unwrap_or("neutral").to_string(),
                        importance_level: 0.7,
                        first_appearance: 1,
                        last_appearance: 9999,
                    });
                }
                
                if !characters.is_empty() {
                    tracing::info!("Successfully parsed {} characters from structured JSON", characters.len());
                    return characters;
                }
            }
        }
        
        tracing::warn!("Failed to parse structured character JSON, attempting fallback extraction");
        tracing::debug!("Response was: {}", response);
        self.fallback_character_extraction(response)
    }
    
    fn fallback_character_extraction(&self, text: &str) -> Vec<CharacterProfile> {
        let mut characters = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        
        for window in words.windows(2) {
            if window.len() == 2 {
                let potential_name = format!("{} {}", window[0], window[1]);
                if self.looks_like_character_name(&potential_name) {
                    characters.push(CharacterProfile {
                        canonical_name: potential_name.clone(),
                        aliases: vec![potential_name],
                        description: "Character detected from text analysis".to_string(),
                        personality_traits: vec!["unknown".to_string()],
                        emotional_baseline: "neutral".to_string(),
                        speaking_style: "normal".to_string(),
                        importance_level: 0.3,
                        first_appearance: 1,
                        last_appearance: 9999,
                    });
                }
            }
        }
        
        characters
    }
    
    fn looks_like_character_name(&self, text: &str) -> bool {
        text.chars().all(|c| c.is_alphabetic() || c.is_whitespace()) && 
        text.chars().any(|c| c.is_uppercase()) &&
        !text.to_lowercase().contains("the") &&
        !text.to_lowercase().contains("and") &&
        text.len() > 3 && text.len() < 30
    }
    
    fn determine_voice_type_from_description(&self, description: &str) -> String {
        let desc_lower = description.to_lowercase();
        
        if desc_lower.contains("young") && desc_lower.contains("male") {
            "young_adult_male".to_string()
        } else if desc_lower.contains("young") && desc_lower.contains("female") {
            "young_adult_female".to_string()
        } else if desc_lower.contains("old") && desc_lower.contains("male") {
            "elderly_male".to_string()
        } else if desc_lower.contains("old") && desc_lower.contains("female") {
            "elderly_female".to_string()
        } else if desc_lower.contains("child") {
            "child".to_string()
        } else if desc_lower.contains("deep") || desc_lower.contains("masculine") {
            "mature_male".to_string()
        } else if desc_lower.contains("soft") || desc_lower.contains("feminine") {
            "mature_female".to_string()
        } else {
            "neutral".to_string()
        }
    }
    
    fn determine_accent_from_description(&self, description: &str) -> Option<String> {
        let desc_lower = description.to_lowercase();
        
        if desc_lower.contains("formal") || desc_lower.contains("educated") {
            Some("formal_educated".to_string())
        } else if desc_lower.contains("noble") || desc_lower.contains("aristocratic") {
            Some("aristocratic".to_string())
        } else if desc_lower.contains("street") || desc_lower.contains("casual") {
            Some("casual_street".to_string())
        } else if desc_lower.contains("foreign") || desc_lower.contains("accent") {
            Some("foreign".to_string())
        } else {
            None
        }
    }
    
    fn determine_pace_from_personality(&self, traits: &[String]) -> String {
        let traits_str = traits.join(" ").to_lowercase();
        
        if traits_str.contains("quick") || traits_str.contains("energetic") || traits_str.contains("excited") {
            "quick".to_string()
        } else if traits_str.contains("slow") || traits_str.contains("deliberate") || traits_str.contains("careful") {
            "slow".to_string()
        } else if traits_str.contains("analytical") || traits_str.contains("thoughtful") {
            "measured".to_string()
        } else {
            "normal".to_string()
        }
    }
    
    fn create_fallback_character_cache(&self) -> CharacterCache {
        // Return empty cache - let the system work without any characters
        // rather than making assumptions about the story
        CharacterCache {
            character_profiles: HashMap::new(),
            character_aliases: HashMap::new(), 
            voice_mappings: HashMap::new(),
        }
    }

    fn create_fallback_mood_cache(&self) -> MoodPatternCache {
        let mut mood_triggers = HashMap::new();
        
        // Add some basic fallback mood triggers
        mood_triggers.insert("pain_cluster".to_string(), MoodTrigger {
            trigger_words: vec!["pain".to_string(), "hurt".to_string(), "agony".to_string(), "suffering".to_string()],
            resulting_mood: "pain".to_string(),
            intensity_modifier: 1.2,
            context_required: false,
        });

        mood_triggers.insert("mystery_cluster".to_string(), MoodTrigger {
            trigger_words: vec!["mysterious".to_string(), "strange".to_string(), "peculiar".to_string(), "unknown".to_string()],
            resulting_mood: "mysterious".to_string(),
            intensity_modifier: 1.0,
            context_required: false,
        });

        MoodPatternCache {
            mood_triggers,
            emotional_patterns: Vec::new(),
            tension_indicators: Vec::new(),
        }
    }
    
    fn extract_objects_from_rag(&self, response: &str, objects: &mut HashMap<String, ObjectProfile>) {
        // Parse objects dynamically from RAG response
        for line in response.lines() {
            let line = line.trim();
            if line.is_empty() { continue; }
            
            // Look for object patterns in RAG response
            if line.to_lowercase().contains("object:") || line.to_lowercase().contains("item:") {
                if let Some(object_name) = line.split(':').nth(1) {
                    let name = object_name.trim().to_string();
                    objects.insert(name.clone(), ObjectProfile {
                        name: name.clone(),
                        description: format!("Important object mentioned in story: {}", name),
                        significance: 0.5,
                        mystery_level: 0.3,
                        first_mentioned: 1,
                        related_mysteries: Vec::new(),
                    });
                }
            }
        }
    }
    
    fn extract_mystery_keywords_from_rag(&self, response: &str, keywords: &mut Vec<MysteryKeyword>) {
        // Extract keywords dynamically from RAG response
        let words: Vec<&str> = response.split_whitespace().collect();
        let mystery_indicators = ["mysterious", "strange", "unknown", "secret", "hidden", "clue", "evidence"];
        
        for window in words.windows(2) {
            if mystery_indicators.iter().any(|&indicator| 
                window.iter().any(|&word| word.to_lowercase().contains(indicator))
            ) {
                let keyword = window.join(" ").to_lowercase();
                if keyword.len() > 3 && keyword.len() < 50 {
                    keywords.push(MysteryKeyword {
                        keyword: keyword.clone(),
                        importance: 0.5,
                        context: "Mystery element detected from story analysis".to_string(),
                        related_clues: Vec::new(),
                        spoiler_level: "unknown".to_string(),
                    });
                }
            }
        }
    }
    
    fn extract_foreshadowing_patterns_from_rag(&self, response: &str) -> Vec<ForeshadowingPattern> {
        let mut patterns = Vec::new();
        
        // Look for foreshadowing patterns in RAG response
        for line in response.lines() {
            let line = line.trim().to_lowercase();
            if line.contains("foreshadow") || line.contains("hint") || line.contains("suggest") {
                patterns.push(ForeshadowingPattern {
                    pattern: line.to_string(),
                    significance: 0.5,
                    predicted_relevance: "Future story development".to_string(),
                    confidence: 0.6,
                    chapters_mentioned: Vec::new(),
                });
            }
        }
        
        patterns
    }
    
    fn extract_symbolic_elements_from_rag(&self, response: &str) -> HashMap<String, SymbolicElement> {
        let mut elements = HashMap::new();
        
        // Look for symbolic elements in RAG response
        for line in response.lines() {
            let line = line.trim().to_lowercase();
            if line.contains("symbol") || line.contains("represent") || line.contains("meaning") {
                let words: Vec<&str> = line.split_whitespace().collect();
                for word in words {
                    if word.len() > 3 && word.len() < 20 && word.chars().all(|c| c.is_alphabetic()) {
                        elements.insert(word.to_string(), SymbolicElement {
                            symbol: word.to_string(),
                            meaning: "Symbolic element from story analysis".to_string(),
                            emotional_weight: 0.5,
                            story_significance: 0.5,
                        });
                    }
                }
            }
        }
        
        elements
    }
    
    fn extract_narrative_threads_from_rag(&self, response: &str) -> Vec<NarrativeThread> {
        let mut threads = Vec::new();
        
        // Look for narrative threads in RAG response
        for (i, line) in response.lines().enumerate() {
            let line = line.trim().to_lowercase();
            if line.contains("plot") || line.contains("story") || line.contains("thread") {
                threads.push(NarrativeThread {
                    thread_id: format!("thread_{}", i),
                    title: "Narrative thread from analysis".to_string(),
                    status: "active".to_string(),
                    key_chapters: Vec::new(),
                    resolution_prediction: None,
                });
            }
        }
        
        threads
    }
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


impl Default for StoryCache {
    fn default() -> Self {
        Self::new()
    }
}
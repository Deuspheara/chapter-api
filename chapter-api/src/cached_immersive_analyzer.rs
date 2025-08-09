use crate::{
    immersive::*,
    story_cache::*,
    models::Chapter,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

pub struct CachedImmersiveAnalyzer {
    story_cache: StoryCache,
}

impl CachedImmersiveAnalyzer {
    pub fn new() -> Self {
        Self {
            story_cache: StoryCache::new(),
        }
    }

    pub async fn get_or_build_cache(&self) -> Result<StoryCacheData, Box<dyn std::error::Error + Send + Sync>> {
        match self.story_cache.get_or_build_cache().await {
            Ok(cache) => Ok(cache),
            Err(e) => {
                tracing::error!("Failed to get or build cache: {}", e);
                // Return a fallback empty cache instead of propagating the error
                Ok(StoryCacheData {
                    characters: CharacterCache {
                        character_profiles: HashMap::new(),
                        character_aliases: HashMap::new(),
                        voice_mappings: HashMap::new(),
                    },
                    story_elements: StoryElementCache {
                        important_objects: HashMap::new(),
                        locations: HashMap::new(),
                        concepts: HashMap::new(),
                        mystery_keywords: Vec::new(),
                    },
                    foreshadowing: ForeshadowingCache {
                        foreshadowing_patterns: Vec::new(),
                        symbolic_elements: HashMap::new(),
                        narrative_threads: Vec::new(),
                    },
                    mood_patterns: MoodPatternCache {
                        mood_triggers: HashMap::new(),
                        emotional_patterns: Vec::new(),
                        tension_indicators: Vec::new(),
                    },
                    last_updated: chrono::Utc::now().timestamp() as u64,
                    version: 0, // Version 0 indicates fallback cache
                })
            }
        }
    }

    pub async fn invalidate_cache(&self) {
        self.story_cache.invalidate_cache().await
    }

    pub async fn analyze_chapter_mood_cached(&self, chapter: &Chapter) -> MoodAnalysis {
        let cache_data = match self.story_cache.get_or_build_cache().await {
            Ok(cache) => cache,
            Err(e) => {
                tracing::warn!("Failed to get story cache, falling back to basic analysis: {}", e);
                return ImmersiveModeAnalyzer::analyze_chapter_mood(chapter);
            }
        };

        let mut paragraph_moods = Vec::new();
        let mut emotional_peaks = Vec::new();
        
        for (index, paragraph) in chapter.paragraphs.iter().enumerate() {
            let mood = self.analyze_paragraph_mood_with_cache(paragraph, &cache_data.mood_patterns);
            let intensity = self.calculate_paragraph_intensity_with_cache(paragraph, &cache_data.mood_patterns);
            let emotions = self.extract_emotions_with_cache(paragraph, &cache_data.mood_patterns);
            let tension = self.calculate_tension_level_with_cache(paragraph, &cache_data.mood_patterns);
            
            paragraph_moods.push(ParagraphMood {
                paragraph_index: index,
                mood: mood.clone(),
                intensity,
                emotions: emotions.clone(),
                tension_level: tension,
            });
            
            if intensity > 0.8 {
                emotional_peaks.push(EmotionalPeak {
                    paragraph_index: index,
                    peak_type: Self::classify_peak_type(&mood, intensity),
                    intensity,
                    description: Self::describe_emotional_peak(&mood, intensity),
                });
            }
        }
        
        let overall_mood = self.determine_overall_mood_with_cache(chapter, &cache_data);
        let mood_intensity = self.calculate_mood_intensity_with_cache(chapter, &cache_data.mood_patterns);
        let mood_transitions = Self::detect_mood_transitions(&paragraph_moods);
        
        let background_theme = MoodAnalysis {
            overall_mood: overall_mood.clone(),
            mood_intensity,
            background_theme: BackgroundTheme {
                primary_color: "".to_string(),
                secondary_color: "".to_string(),
                gradient_type: "".to_string(),
                opacity: 0.0,
                suggested_font_color: "".to_string(),
                theme_name: "".to_string(),
            },
            paragraph_moods: paragraph_moods.clone(),
            mood_transitions: mood_transitions.clone(),
            emotional_peaks: emotional_peaks.clone(),
        }.get_background_colors();
        
        MoodAnalysis {
            overall_mood,
            mood_intensity,
            background_theme,
            paragraph_moods,
            mood_transitions,
            emotional_peaks,
        }
    }

    pub async fn extract_characters_with_cache(&self, chapter: &Chapter) -> Vec<String> {
        let cache_data = match self.story_cache.get_or_build_cache().await {
            Ok(cache) => cache,
            Err(_) => return self.fallback_character_extraction(chapter),
        };

        let mut characters = Vec::new();
        let content = chapter.paragraphs.join(" ").to_lowercase();
        
        for (canonical_name, profile) in &cache_data.characters.character_profiles {
            for alias in &profile.aliases {
                if content.contains(&alias.to_lowercase()) {
                    characters.push(canonical_name.clone());
                    break;
                }
            }
            if content.contains(&canonical_name.to_lowercase()) {
                characters.push(canonical_name.clone());
            }
        }

        for (alias, canonical) in &cache_data.characters.character_aliases {
            if content.contains(&alias.to_lowercase()) && !characters.contains(canonical) {
                characters.push(canonical.clone());
            }
        }
        
        characters
    }

    pub async fn get_character_voice_with_cache(&self, character: &str) -> Option<VoiceProfile> {
        let cache_data = self.story_cache.get_or_build_cache().await.ok()?;
        
        if let Some(voice) = cache_data.characters.voice_mappings.get(character) {
            return Some(voice.clone());
        }
        
        if let Some(canonical) = cache_data.characters.character_aliases.get(character) {
            return cache_data.characters.voice_mappings.get(canonical).cloned();
        }
        
        None
    }

    pub async fn get_chapter_adapted_voice(&self, character: &str, chapter: &crate::models::Chapter) -> Option<VoiceProfile> {
        let base_voice = self.get_character_voice_with_cache(character).await?;
        let cache_data = self.story_cache.get_or_build_cache().await.ok()?;
        
        let character_profile = cache_data.characters.character_profiles.get(character)?;
        
        let chapter_context = format!("Chapter {}: {}\n\nCharacter: {} - {}\n\nHow might this character's voice, tone, and speaking style change in this specific chapter based on the events and their emotional state?", 
            chapter.number, 
            chapter.title,
            character_profile.canonical_name,
            character_profile.description
        );

        match self.get_chapter_voice_adaptation(&chapter_context).await {
            Ok(adapted_voice) => Some(self.merge_voice_profiles(base_voice, adapted_voice)),
            Err(_) => Some(base_voice)
        }
    }

    async fn get_chapter_voice_adaptation(&self, context: &str) -> Result<VoiceProfile, Box<dyn std::error::Error + Send + Sync>> {
        let rag_client = reqwest::Client::new();
        let voice_query = format!("{}\n\nProvide specific voice adaptations:\n\
            - How does their tone change? (more confident, uncertain, angry, sad, etc.)\n\
            - Does their speaking pace change? (faster when excited, slower when thoughtful)\n\
            - Any temporary accent changes? (stress, emotion, situation)\n\
            - Updated emotional range for this chapter", context);

        let rag_base = std::env::var("RAG_SERVICE_URL").unwrap_or_else(|_| "http://127.0.0.1:8001".to_string());
        let rag_url = format!("{}/query", rag_base);
        let response = rag_client
            .post(&rag_url)
            .json(&serde_json::json!({
                "question": voice_query
            }))
            .send()
            .await?;

        let rag_response: RagResponse = response.json().await?;
        
        let tone = self.extract_tone_from_response(&rag_response.answer);
        let pace = self.extract_pace_from_response(&rag_response.answer);
        let accent = self.extract_accent_from_response(&rag_response.answer);
        let emotional_range = self.extract_emotions_from_response(&rag_response.answer);
        
        Ok(VoiceProfile {
            voice_type: "adapted".to_string(), // Will be merged with base
            accent,
            tone,
            speaking_pace: pace,
            emotional_range,
        })
    }

    fn merge_voice_profiles(&self, base: VoiceProfile, adaptation: VoiceProfile) -> VoiceProfile {
        VoiceProfile {
            voice_type: base.voice_type, // Keep original voice type
            accent: adaptation.accent.or(base.accent), // Use adapted accent if available
            tone: if adaptation.tone != "adapted" { adaptation.tone } else { base.tone },
            speaking_pace: if adaptation.speaking_pace != "normal" { adaptation.speaking_pace } else { base.speaking_pace },
            emotional_range: if !adaptation.emotional_range.is_empty() { 
                adaptation.emotional_range 
            } else { 
                base.emotional_range 
            },
        }
    }

    fn extract_tone_from_response(&self, response: &str) -> String {
        let response_lower = response.to_lowercase();
        
        if response_lower.contains("confident") { "confident".to_string() }
        else if response_lower.contains("uncertain") { "uncertain".to_string() }
        else if response_lower.contains("angry") { "angry".to_string() }
        else if response_lower.contains("sad") { "melancholic".to_string() }
        else if response_lower.contains("excited") { "excited".to_string() }
        else if response_lower.contains("fearful") { "fearful".to_string() }
        else if response_lower.contains("determined") { "determined".to_string() }
        else { "neutral".to_string() }
    }

    fn extract_pace_from_response(&self, response: &str) -> String {
        let response_lower = response.to_lowercase();
        
        if response_lower.contains("faster") || response_lower.contains("quick") { "quick".to_string() }
        else if response_lower.contains("slower") || response_lower.contains("deliberate") { "slow".to_string() }
        else if response_lower.contains("measured") || response_lower.contains("careful") { "measured".to_string() }
        else { "normal".to_string() }
    }

    fn extract_accent_from_response(&self, response: &str) -> Option<String> {
        let response_lower = response.to_lowercase();
        
        if response_lower.contains("stress") && response_lower.contains("accent") {
            Some("stressed".to_string())
        } else if response_lower.contains("formal") {
            Some("formal_stressed".to_string())
        } else {
            None
        }
    }

    fn extract_emotions_from_response(&self, response: &str) -> Vec<String> {
        let response_lower = response.to_lowercase();
        let mut emotions = Vec::new();
        
        let emotion_keywords = [
            "angry", "sad", "happy", "excited", "fearful", "confident", 
            "uncertain", "determined", "confused", "hopeful", "desperate"
        ];
        
        for emotion in emotion_keywords {
            if response_lower.contains(emotion) {
                emotions.push(emotion.to_string());
            }
        }
        
        if emotions.is_empty() {
            emotions.push("neutral".to_string());
        }
        
        emotions
    }

    pub async fn analyze_mystery_elements_with_cache(&self, chapter: &Chapter) -> MysteryElements {
        let cache_data = match self.story_cache.get_or_build_cache().await {
            Ok(cache) => cache,
            Err(_) => return self.fallback_mystery_analysis(chapter),
        };

        // If cache is empty (version 0), use fallback analysis
        if cache_data.version == 0 || 
           (cache_data.story_elements.mystery_keywords.is_empty() && 
            cache_data.foreshadowing.foreshadowing_patterns.is_empty()) {
            tracing::info!("Cache is empty or not built, using fallback mystery analysis for chapter {}", chapter.number);
            return self.fallback_mystery_analysis(chapter);
        }

        let mut clues_found = Vec::new();
        let mut theories_suggested = Vec::new();
        let connections_discovered = Vec::new();
        let mut foreshadowing = Vec::new();

        for (index, paragraph) in chapter.paragraphs.iter().enumerate() {
            if let Some(clue) = self.find_mystery_clue_with_cache(paragraph, index, &cache_data.story_elements) {
                clues_found.push(clue);
            }

            if let Some(foreshadow) = self.find_foreshadowing_with_cache(paragraph, index, &cache_data.foreshadowing) {
                foreshadowing.push(foreshadow);
            }
        }

        let generated_theories = self.generate_theories_from_cache(&clues_found, &cache_data);
        theories_suggested.extend(generated_theories);

        let mystery_tracker = MysteryTracker {
            active_mysteries: cache_data.foreshadowing.narrative_threads
                .iter()
                .filter(|thread| thread.status == "active")
                .map(|thread| thread.thread_id.clone())
                .collect(),
            resolved_mysteries: cache_data.foreshadowing.narrative_threads
                .iter()
                .filter(|thread| thread.status == "resolved")
                .map(|thread| thread.thread_id.clone())
                .collect(),
            new_mysteries: clues_found.iter()
                .filter(|clue| clue.importance > 0.7)
                .map(|clue| clue.clue_type.clone())
                .collect(),
            mystery_complexity: (clues_found.len() as f32 * 0.1 + foreshadowing.len() as f32 * 0.2).min(1.0),
        };

        MysteryElements {
            clues_found,
            theories_suggested,
            connections_discovered,
            foreshadowing,
            mystery_tracker,
        }
    }

    fn analyze_paragraph_mood_with_cache(&self, paragraph: &str, mood_cache: &MoodPatternCache) -> String {
        let para_lower = paragraph.to_lowercase();
        
        for (_, trigger) in &mood_cache.mood_triggers {
            for trigger_word in &trigger.trigger_words {
                if para_lower.contains(trigger_word) {
                    return trigger.resulting_mood.clone();
                }
            }
        }
        
        "neutral".to_string()
    }

    fn calculate_paragraph_intensity_with_cache(&self, paragraph: &str, mood_cache: &MoodPatternCache) -> f32 {
        let para_lower = paragraph.to_lowercase();
        let mut intensity = 0.0;
        
        for (_, trigger) in &mood_cache.mood_triggers {
            for trigger_word in &trigger.trigger_words {
                if para_lower.contains(trigger_word) {
                    intensity += 0.3 * trigger.intensity_modifier;
                }
            }
        }
        
        let punctuation_intensity = paragraph.matches('!').count() as f32 * 0.3 +
                                   paragraph.matches('?').count() as f32 * 0.2 +
                                   paragraph.matches("...").count() as f32 * 0.4;
        
        (intensity + punctuation_intensity).min(1.0)
    }

    fn extract_emotions_with_cache(&self, paragraph: &str, mood_cache: &MoodPatternCache) -> Vec<String> {
        let para_lower = paragraph.to_lowercase();
        let mut emotions = Vec::new();
        
        for (_, trigger) in &mood_cache.mood_triggers {
            for trigger_word in &trigger.trigger_words {
                if para_lower.contains(trigger_word) {
                    emotions.push(trigger.resulting_mood.clone());
                    break;
                }
            }
        }
        
        emotions
    }

    fn calculate_tension_level_with_cache(&self, paragraph: &str, mood_cache: &MoodPatternCache) -> f32 {
        let para_lower = paragraph.to_lowercase();
        let mut tension: f32 = 0.0;
        
        for indicator in &mood_cache.tension_indicators {
            if para_lower.contains(&indicator.indicator) {
                tension += indicator.tension_level * indicator.reliability;
            }
        }
        
        tension.min(1.0)
    }

    fn determine_overall_mood_with_cache(&self, chapter: &Chapter, cache_data: &StoryCacheData) -> String {
        let content = chapter.paragraphs.join(" ").to_lowercase();
        let title_lower = chapter.title.to_lowercase();
        
        for (symbol, element) in &cache_data.foreshadowing.symbolic_elements {
            if title_lower.contains(symbol) || content.contains(symbol) {
                if element.emotional_weight > 0.7 {
                    return symbol.clone();
                }
            }
        }
        
        for (_, trigger) in &cache_data.mood_patterns.mood_triggers {
            for trigger_word in &trigger.trigger_words {
                if content.contains(trigger_word) {
                    return trigger.resulting_mood.clone();
                }
            }
        }
        
        "neutral".to_string()
    }

    fn calculate_mood_intensity_with_cache(&self, chapter: &Chapter, mood_cache: &MoodPatternCache) -> f32 {
        let content = chapter.paragraphs.join(" ").to_lowercase();
        let mut intensity = 0.0;
        
        for (_, trigger) in &mood_cache.mood_triggers {
            for trigger_word in &trigger.trigger_words {
                if content.contains(trigger_word) {
                    intensity += 0.2 * trigger.intensity_modifier;
                }
            }
        }
        
        for indicator in &mood_cache.tension_indicators {
            if content.contains(&indicator.indicator) {
                intensity += indicator.tension_level * 0.1;
            }
        }
        
        intensity.min(1.0)
    }

    fn find_mystery_clue_with_cache(&self, paragraph: &str, index: usize, story_elements: &StoryElementCache) -> Option<ClueElement> {
        let para_lower = paragraph.to_lowercase();
        
        for keyword in &story_elements.mystery_keywords {
            if para_lower.contains(&keyword.keyword) {
                return Some(ClueElement {
                    paragraph_index: index,
                    clue_type: keyword.keyword.clone(),
                    description: if paragraph.len() > 100 {
                        format!("{}...", &paragraph[..97])
                    } else {
                        paragraph.to_string()
                    },
                    importance: keyword.importance,
                    related_mysteries: keyword.related_clues.clone(),
                    spoiler_level: keyword.spoiler_level.clone(),
                });
            }
        }
        
        for (obj_name, obj_profile) in &story_elements.important_objects {
            if para_lower.contains(obj_name) {
                return Some(ClueElement {
                    paragraph_index: index,
                    clue_type: "object_clue".to_string(),
                    description: obj_profile.description.clone(),
                    importance: obj_profile.significance,
                    related_mysteries: obj_profile.related_mysteries.clone(),
                    spoiler_level: if obj_profile.mystery_level > 0.7 { "medium".to_string() } else { "low".to_string() },
                });
            }
        }
        
        None
    }

    fn find_foreshadowing_with_cache(&self, paragraph: &str, index: usize, foreshadowing_cache: &ForeshadowingCache) -> Option<ForeshadowingElement> {
        let para_lower = paragraph.to_lowercase();
        
        for pattern in &foreshadowing_cache.foreshadowing_patterns {
            if para_lower.contains(&pattern.pattern) {
                return Some(ForeshadowingElement {
                    paragraph_index: index,
                    element: if paragraph.len() > 50 {
                        format!("{}...", &paragraph[..47])
                    } else {
                        paragraph.to_string()
                    },
                    foreshadow_type: "pattern_based".to_string(),
                    potential_future_relevance: pattern.predicted_relevance.clone(),
                    confidence: pattern.confidence,
                });
            }
        }
        
        for (symbol, element) in &foreshadowing_cache.symbolic_elements {
            if para_lower.contains(symbol) {
                return Some(ForeshadowingElement {
                    paragraph_index: index,
                    element: element.meaning.clone(),
                    foreshadow_type: "symbolic".to_string(),
                    potential_future_relevance: format!("Symbol significance: {}", element.story_significance),
                    confidence: element.emotional_weight,
                });
            }
        }
        
        None
    }

    fn generate_theories_from_cache(&self, clues: &[ClueElement], cache_data: &StoryCacheData) -> Vec<Theory> {
        let mut theories = Vec::new();
        
        if !clues.is_empty() {
            for thread in &cache_data.foreshadowing.narrative_threads {
                if thread.status == "active" {
                    theories.push(Theory {
                        theory_id: thread.thread_id.clone(),
                        title: thread.title.clone(),
                        description: thread.resolution_prediction.clone().unwrap_or_else(|| "Theory in development".to_string()),
                        supporting_clues: clues.iter().map(|c| c.clue_type.clone()).collect(),
                        confidence: 0.7,
                        spoiler_risk: "low".to_string(),
                    });
                }
            }
        }
        
        theories
    }

    fn fallback_character_extraction(&self, chapter: &Chapter) -> Vec<String> {
        // Pure pattern-based character detection - no story assumptions
        let mut characters = Vec::new();
        let content = chapter.paragraphs.join(" ");
        
        // Look for character-like patterns (capitalized names)
        let words: Vec<&str> = content.split_whitespace().collect();
        for window in words.windows(2) {
            if window.len() == 2 {
                let potential_name = format!("{} {}", window[0], window[1]);
                if self.looks_like_character_name(&potential_name) {
                    characters.push(potential_name);
                }
            }
        }
        
        // Also check single words that look like names
        for word in words {
            if self.looks_like_single_name(word) {
                characters.push(word.to_string());
            }
        }
        
        // Remove duplicates
        characters.sort();
        characters.dedup();
        
        characters
    }
    
    fn looks_like_single_name(&self, word: &str) -> bool {
        word.len() > 2 && 
        word.len() < 20 &&
        word.chars().next().unwrap_or('a').is_uppercase() &&
        word.chars().all(|c| c.is_alphabetic()) &&
        !["The", "And", "But", "For", "With", "This", "That", "Chapter"].contains(&word)
    }
    
    fn looks_like_character_name(&self, text: &str) -> bool {
        text.chars().all(|c| c.is_alphabetic() || c.is_whitespace()) && 
        text.chars().any(|c| c.is_uppercase()) &&
        !text.to_lowercase().contains("the") &&
        !text.to_lowercase().contains("and") &&
        text.len() > 3 && text.len() < 30
    }

    pub fn fallback_mystery_analysis(&self, chapter: &Chapter) -> MysteryElements {
        // Provide actual mystery analysis using basic pattern matching
        let mut clues_found = Vec::new();
        let mut theories_suggested = Vec::new();
        let mut foreshadowing = Vec::new();
        
        let content_lower = chapter.paragraphs.join(" ").to_lowercase();
        
        // Mystery keywords to look for
        let mystery_keywords = [
            ("strange", "observation", 0.4),
            ("mysterious", "observation", 0.5),
            ("secret", "information", 0.6),
            ("hidden", "location", 0.5),
            ("suspicious", "behavior", 0.4),
            ("unusual", "observation", 0.3),
            ("whisper", "dialogue", 0.4),
            ("shadow", "visual", 0.3),
            ("disappeared", "event", 0.7),
            ("vanished", "event", 0.7),
            ("appeared", "event", 0.4),
            ("footsteps", "sound", 0.5),
            ("locked", "object", 0.5),
            ("key", "object", 0.6),
            ("door", "location", 0.3),
            ("room", "location", 0.2),
            ("letter", "object", 0.5),
            ("note", "object", 0.4),
            ("blood", "evidence", 0.8),
            ("scream", "sound", 0.7),
            ("weapon", "object", 0.9),
            ("knife", "object", 0.8),
            ("gun", "object", 0.9),
            ("poison", "object", 0.9),
            ("murder", "event", 1.0),
            ("kill", "event", 1.0),
            ("death", "event", 0.8),
            ("body", "evidence", 0.9),
            ("corpse", "evidence", 0.9),
        ];
        
        // Analyze each paragraph for clues
        for (paragraph_index, paragraph) in chapter.paragraphs.iter().enumerate() {
            let para_lower = paragraph.to_lowercase();
            
            // Check for mystery keywords
            for (keyword, clue_type, importance) in &mystery_keywords {
                if para_lower.contains(keyword) {
                    clues_found.push(ClueElement {
                        paragraph_index,
                        clue_type: format!("{}_clue", clue_type),
                        description: if paragraph.len() > 150 {
                            format!("{}...", &paragraph[..147])
                        } else {
                            paragraph.clone()
                        },
                        importance: *importance,
                        related_mysteries: vec![format!("mystery_involving_{}", keyword)],
                        spoiler_level: if *importance > 0.7 { "medium".to_string() } else { "low".to_string() },
                    });
                }
            }
            
            // Look for dialogue that might contain clues
            if paragraph.contains('"') || paragraph.contains('"') || paragraph.contains('"') {
                let suspicious_dialogue_words = ["don't tell", "secret", "nobody knows", "between us", "promise me"];
                if suspicious_dialogue_words.iter().any(|word| para_lower.contains(word)) {
                    clues_found.push(ClueElement {
                        paragraph_index,
                        clue_type: "dialogue_clue".to_string(),
                        description: if paragraph.len() > 150 {
                            format!("{}...", &paragraph[..147])
                        } else {
                            paragraph.clone()
                        },
                        importance: 0.6,
                        related_mysteries: vec!["secretive_dialogue".to_string()],
                        spoiler_level: "low".to_string(),
                    });
                }
            }
            
            // Look for foreshadowing patterns
            let foreshadowing_words = ["ominous", "foreboding", "sense of", "feeling that", "little did", "unaware", "would later", "destiny"];
            for foreshadow_word in &foreshadowing_words {
                if para_lower.contains(foreshadow_word) {
                    foreshadowing.push(ForeshadowingElement {
                        paragraph_index,
                        element: if paragraph.len() > 100 {
                            format!("{}...", &paragraph[..97])
                        } else {
                            paragraph.clone()
                        },
                        foreshadow_type: "narrative".to_string(),
                        potential_future_relevance: "May hint at future plot developments".to_string(),
                        confidence: 0.5,
                    });
                    break; // Only add one foreshadowing element per paragraph
                }
            }
        }
        
        // Generate theories based on found clues
        if !clues_found.is_empty() {
            // Group clues by type for theory generation
            let mut clue_types: std::collections::HashMap<String, Vec<&ClueElement>> = std::collections::HashMap::new();
            for clue in &clues_found {
                clue_types.entry(clue.clue_type.clone()).or_insert_with(Vec::new).push(clue);
            }
            
            // Generate theories based on clue patterns
            for (clue_type, clues) in &clue_types {
                if clues.len() >= 2 {
                    theories_suggested.push(Theory {
                        theory_id: format!("theory_{}", clue_type),
                        title: format!("The {} Pattern", clue_type.replace("_", " ").to_uppercase()),
                        description: format!("Multiple {} found throughout the chapter suggest a developing pattern or mystery that may be significant to the overall story.", clue_type.replace("_", " ")),
                        supporting_clues: clues.iter().map(|c| c.description.clone()).collect(),
                        confidence: (clues.len() as f32 * 0.2).min(0.8),
                        spoiler_risk: "minimal".to_string(),
                    });
                }
            }
            
            // Add a general mystery theory if we have high-importance clues
            let high_importance_clues: Vec<&ClueElement> = clues_found.iter().filter(|c| c.importance > 0.6).collect();
            if !high_importance_clues.is_empty() {
                theories_suggested.push(Theory {
                    theory_id: "general_mystery".to_string(),
                    title: "Chapter Mystery Elements".to_string(),
                    description: format!("This chapter contains {} significant clues that may be important to understanding the broader mystery or plot.", high_importance_clues.len()),
                    supporting_clues: high_importance_clues.iter().map(|c| c.clue_type.clone()).collect(),
                    confidence: 0.6,
                    spoiler_risk: "low".to_string(),
                });
            }
        }
        
        // Determine active mysteries based on content
        let mut active_mysteries = Vec::new();
        if content_lower.contains("murder") || content_lower.contains("kill") || content_lower.contains("death") {
            active_mysteries.push("death_mystery".to_string());
        }
        if content_lower.contains("disappeared") || content_lower.contains("vanished") || content_lower.contains("missing") {
            active_mysteries.push("disappearance_mystery".to_string());
        }
        if content_lower.contains("secret") || content_lower.contains("hidden") {
            active_mysteries.push("secret_mystery".to_string());
        }
        if content_lower.contains("identity") || content_lower.contains("who") {
            active_mysteries.push("identity_mystery".to_string());
        }
        
        let mystery_complexity = (clues_found.len() as f32 * 0.1 + theories_suggested.len() as f32 * 0.2 + foreshadowing.len() as f32 * 0.15).min(1.0);
        
        let new_mysteries = clues_found.iter()
            .filter(|c| c.importance > 0.5)
            .map(|c| format!("new_{}", c.clue_type))
            .collect();
            
        MysteryElements {
            clues_found,
            theories_suggested,
            connections_discovered: Vec::new(), // Would need cross-chapter analysis
            foreshadowing,
            mystery_tracker: MysteryTracker {
                active_mysteries,
                resolved_mysteries: Vec::new(), // Would need story progression tracking
                new_mysteries,
                mystery_complexity,
            },
        }
    }

    fn detect_mood_transitions(paragraph_moods: &[ParagraphMood]) -> Vec<MoodTransition> {
        let mut transitions = Vec::new();
        
        for i in 1..paragraph_moods.len() {
            let prev_mood = &paragraph_moods[i-1];
            let curr_mood = &paragraph_moods[i];
            
            if prev_mood.mood != curr_mood.mood {
                let significance = (prev_mood.intensity - curr_mood.intensity).abs();
                transitions.push(MoodTransition {
                    from_paragraph: i - 1,
                    to_paragraph: i,
                    transition_type: format!("{} to {}", prev_mood.mood, curr_mood.mood),
                    significance,
                });
            }
        }
        
        transitions
    }

    fn classify_peak_type(mood: &str, intensity: f32) -> String {
        match (mood, intensity) {
            (_, i) if i > 0.9 => "climactic".to_string(),
            (_, i) if i > 0.8 => "intense".to_string(),
            _ => "moderate".to_string(),
        }
    }
    
    fn describe_emotional_peak(mood: &str, intensity: f32) -> String {
        format!("A {} emotional peak with intensity {:.1}", mood, intensity)
    }
}

impl Default for CachedImmersiveAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
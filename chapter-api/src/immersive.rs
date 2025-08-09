use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize)]
pub struct ImmersiveModeResponse {
    pub chapter: crate::models::Chapter,
    pub mood_analysis: MoodAnalysis,
    pub voice_narration: VoiceNarration,
    pub emotional_journey: EmotionalJourney,
    pub mystery_elements: MysteryElements,
    pub quick_recap: Option<QuickRecap>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MoodAnalysis {
    pub overall_mood: String,
    pub mood_intensity: f32,
    pub background_theme: BackgroundTheme,
    pub paragraph_moods: Vec<ParagraphMood>,
    pub mood_transitions: Vec<MoodTransition>,
    pub emotional_peaks: Vec<EmotionalPeak>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackgroundTheme {
    pub primary_color: String,
    pub secondary_color: String,
    pub gradient_type: String,
    pub opacity: f32,
    pub suggested_font_color: String,
    pub theme_name: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ParagraphMood {
    pub paragraph_index: usize,
    pub mood: String,
    pub intensity: f32,
    pub emotions: Vec<String>,
    pub tension_level: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct MoodTransition {
    pub from_paragraph: usize,
    pub to_paragraph: usize,
    pub transition_type: String,
    pub significance: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmotionalPeak {
    pub paragraph_index: usize,
    pub peak_type: String,
    pub intensity: f32,
    pub description: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct VoiceNarration {
    pub narrator_voice: String,
    pub character_voices: HashMap<String, CharacterVoice>,
    pub reading_pace: String,
    pub emphasis_points: Vec<EmphasisPoint>,
    pub audio_cues: Vec<AudioCue>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CharacterVoice {
    pub voice_type: String,
    pub accent: Option<String>,
    pub tone: String,
    pub speaking_pace: String,
    pub emotional_range: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmphasisPoint {
    pub paragraph_index: usize,
    pub text_segment: String,
    pub emphasis_type: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AudioCue {
    pub paragraph_index: usize,
    pub cue_type: String,
    pub description: String,
    pub timing: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmotionalJourney {
    pub chapter_arc: EmotionalArc,
    pub character_emotions: HashMap<String, Vec<CharacterEmotion>>,
    pub tension_graph: Vec<TensionPoint>,
    pub emotional_moments: Vec<EmotionalMoment>,
    pub reading_recommendations: ReadingRecommendations,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmotionalArc {
    pub start_emotion: String,
    pub end_emotion: String,
    pub arc_type: String,
    pub complexity: f32,
    pub emotional_journey_map: Vec<EmotionalWaypoint>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmotionalWaypoint {
    pub paragraph_index: usize,
    pub emotion: String,
    pub intensity: f32,
    pub significance: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CharacterEmotion {
    pub paragraph_index: usize,
    pub character_name: String,
    pub emotion: String,
    pub intensity: f32,
    pub context: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TensionPoint {
    pub paragraph_index: usize,
    pub tension_level: f32,
    pub tension_type: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmotionalMoment {
    pub paragraph_index: usize,
    pub moment_type: String,
    pub intensity: f32,
    pub description: String,
    pub suggested_pause: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReadingRecommendations {
    pub suggested_breaks: Vec<usize>,
    pub intense_moments_warning: Vec<usize>,
    pub reading_pace: String,
    pub environment_suggestion: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MysteryElements {
    pub clues_found: Vec<ClueElement>,
    pub theories_suggested: Vec<Theory>,
    pub connections_discovered: Vec<Connection>,
    pub foreshadowing: Vec<ForeshadowingElement>,
    pub mystery_tracker: MysteryTracker,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClueElement {
    pub paragraph_index: usize,
    pub clue_type: String,
    pub description: String,
    pub importance: f32,
    pub related_mysteries: Vec<String>,
    pub spoiler_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Theory {
    pub theory_id: String,
    pub title: String,
    pub description: String,
    pub supporting_clues: Vec<String>,
    pub confidence: f32,
    pub spoiler_risk: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub connection_type: String,
    pub current_element: String,
    pub connected_chapter: u32,
    pub connected_element: String,
    pub significance: f32,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeshadowingElement {
    pub paragraph_index: usize,
    pub element: String,
    pub foreshadow_type: String,
    pub potential_future_relevance: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MysteryTracker {
    pub active_mysteries: Vec<String>,
    pub resolved_mysteries: Vec<String>,
    pub new_mysteries: Vec<String>,
    pub mystery_complexity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuickRecap {
    pub previous_chapter_summary: String,
    pub key_events: Vec<String>,
    pub character_status_updates: Vec<CharacterUpdate>,
    pub plot_threads_continuation: Vec<PlotThread>,
    pub emotional_state_from_previous: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterUpdate {
    pub character_name: String,
    pub last_known_status: String,
    pub important_development: Option<String>,
    pub emotional_state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotThread {
    pub thread_name: String,
    pub status: String,
    pub last_development: String,
    pub continuation_in_current_chapter: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImmersiveModeQuery {
    pub include_recap: Option<bool>,
    pub mood_sensitivity: Option<f32>,
    pub spoiler_tolerance: Option<String>,
    pub voice_preferences: Option<VoicePreferences>,
    pub emotional_tracking: Option<bool>,
    pub mystery_analysis_depth: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoicePreferences {
    pub narrator_voice_type: String,
    pub reading_speed: String,
    pub character_differentiation: bool,
    pub emotional_expression: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualQuestionResponse {
    pub answer: String,
    pub related_paragraphs: Vec<usize>,
    pub character_context: Vec<String>,
    pub plot_relevance: String,
    pub spoiler_warning: Option<String>,
    pub follow_up_suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualQuestion {
    pub question: String,
    pub current_chapter: u32,
    pub current_paragraph: Option<usize>,
    pub reading_context: Vec<u32>,
    pub user_reading_history: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterAnalysisStatus {
    pub chapter_number: u32,
    pub mystery_analysis_ready: bool,
    pub recap_ready: bool,
    pub last_updated: String,
    pub processing_complete: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmotionalVisualization {
    pub emotion_timeline: Vec<EmotionTimelinePoint>,
    pub tension_curve: Vec<f32>,
    pub character_emotional_arcs: HashMap<String, Vec<EmotionPoint>>,
    pub mood_color_map: Vec<MoodColorPoint>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmotionTimelinePoint {
    pub paragraph_index: usize,
    pub primary_emotion: String,
    pub intensity: f32,
    pub secondary_emotions: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmotionPoint {
    pub paragraph_index: usize,
    pub emotion: String,
    pub intensity: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct MoodColorPoint {
    pub paragraph_index: usize,
    pub color: String,
    pub intensity: f32,
}

impl MoodAnalysis {
    pub fn get_background_colors(&self) -> BackgroundTheme {
        match self.overall_mood.as_str() {
            "joyful" | "happy" | "excited" => BackgroundTheme {
                primary_color: "#FFE066".to_string(),
                secondary_color: "#FFD700".to_string(),
                gradient_type: "radial".to_string(),
                opacity: 0.3,
                suggested_font_color: "#2C3E50".to_string(),
                theme_name: "Sunburst Joy".to_string(),
            },
            "angry" | "rage" | "furious" => BackgroundTheme {
                primary_color: "#FF4444".to_string(),
                secondary_color: "#CC0000".to_string(),
                gradient_type: "linear".to_string(),
                opacity: 0.4,
                suggested_font_color: "#FFFFFF".to_string(),
                theme_name: "Crimson Fury".to_string(),
            },
            "sad" | "melancholy" | "grief" => BackgroundTheme {
                primary_color: "#4A90E2".to_string(),
                secondary_color: "#2C5F87".to_string(),
                gradient_type: "linear".to_string(),
                opacity: 0.3,
                suggested_font_color: "#FFFFFF".to_string(),
                theme_name: "Melancholic Blue".to_string(),
            },
            "mysterious" | "suspenseful" | "dark" => BackgroundTheme {
                primary_color: "#2C2C2C".to_string(),
                secondary_color: "#8B4513".to_string(),
                gradient_type: "radial".to_string(),
                opacity: 0.5,
                suggested_font_color: "#D3D3D3".to_string(),
                theme_name: "Shadow Mystery".to_string(),
            },
            "romantic" | "love" | "tender" => BackgroundTheme {
                primary_color: "#FFB6C1".to_string(),
                secondary_color: "#FF69B4".to_string(),
                gradient_type: "linear".to_string(),
                opacity: 0.25,
                suggested_font_color: "#8B0000".to_string(),
                theme_name: "Rose Romance".to_string(),
            },
            "tense" | "anxious" | "nervous" => BackgroundTheme {
                primary_color: "#FF8C00".to_string(),
                secondary_color: "#FF4500".to_string(),
                gradient_type: "linear".to_string(),
                opacity: 0.35,
                suggested_font_color: "#FFFFFF".to_string(),
                theme_name: "Tension Orange".to_string(),
            },
            "peaceful" | "calm" | "serene" => BackgroundTheme {
                primary_color: "#98FB98".to_string(),
                secondary_color: "#90EE90".to_string(),
                gradient_type: "radial".to_string(),
                opacity: 0.2,
                suggested_font_color: "#2F4F4F".to_string(),
                theme_name: "Tranquil Green".to_string(),
            },
            "horror" | "fear" | "terror" => BackgroundTheme {
                primary_color: "#800080".to_string(),
                secondary_color: "#4B0082".to_string(),
                gradient_type: "radial".to_string(),
                opacity: 0.4,
                suggested_font_color: "#FFFFFF".to_string(),
                theme_name: "Nightmare Purple".to_string(),
            },
            _ => BackgroundTheme {
                primary_color: "#F5F5F5".to_string(),
                secondary_color: "#E0E0E0".to_string(),
                gradient_type: "linear".to_string(),
                opacity: 0.1,
                suggested_font_color: "#333333".to_string(),
                theme_name: "Neutral Reading".to_string(),
            }
        }
    }
}

pub struct ImmersiveModeAnalyzer;

impl ImmersiveModeAnalyzer {
    pub fn analyze_chapter_mood(chapter: &crate::models::Chapter) -> MoodAnalysis {
        let mut paragraph_moods = Vec::new();
        let mut emotional_peaks = Vec::new();
        
        let overall_mood = Self::determine_overall_mood(chapter);
        let mood_intensity = Self::calculate_mood_intensity(chapter);
        
        for (index, paragraph) in chapter.paragraphs.iter().enumerate() {
            let mood = Self::analyze_paragraph_mood(paragraph);
            let intensity = Self::calculate_paragraph_intensity(paragraph);
            let emotions = Self::extract_emotions(paragraph);
            let tension = Self::calculate_tension_level(paragraph);
            
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
    
    fn determine_overall_mood(chapter: &crate::models::Chapter) -> String {
        let title_lower = chapter.title.to_lowercase();
        let content = chapter.paragraphs.join(" ").to_lowercase();
        
        if title_lower.contains("crimson") || content.contains("blood") || content.contains("death") {
            "dark".to_string()
        } else if content.contains("horror") || content.contains("terror") || content.contains("fear") {
            "horror".to_string()
        } else if content.contains("mysterious") || content.contains("mystery") {
            "mysterious".to_string()
        } else if content.contains("joy") || content.contains("happy") || content.contains("laugh") {
            "joyful".to_string()
        } else if content.contains("sad") || content.contains("grief") || content.contains("mourn") {
            "sad".to_string()
        } else if content.contains("angry") || content.contains("rage") || content.contains("fury") {
            "angry".to_string()
        } else if content.contains("love") || content.contains("romantic") {
            "romantic".to_string()
        } else if content.contains("tense") || content.contains("anxious") {
            "tense".to_string()
        } else if content.contains("peaceful") || content.contains("calm") {
            "peaceful".to_string()
        } else {
            "neutral".to_string()
        }
    }
    
    fn calculate_mood_intensity(chapter: &crate::models::Chapter) -> f32 {
        let content = chapter.paragraphs.join(" ").to_lowercase();
        let mut intensity = 0.0;
        
        let intense_words = [
            "excruciating", "painful", "horror", "terror", "blood", "death",
            "screaming", "agony", "magnificent", "brilliant", "stunning",
            "devastating", "overwhelming", "incredible", "amazing"
        ];
        
        for word in intense_words.iter() {
            if content.contains(word) {
                intensity += 0.2;
            }
        }
        
        let punctuation_intensity = content.matches('!').count() as f32 * 0.1 +
                                   content.matches('?').count() as f32 * 0.05 +
                                   content.matches("...").count() as f32 * 0.15;
        
        (intensity + punctuation_intensity).min(1.0)
    }
    
    fn analyze_paragraph_mood(paragraph: &str) -> String {
        let para_lower = paragraph.to_lowercase();
        
        if para_lower.contains("pain") || para_lower.contains("hurt") || para_lower.contains("blood") {
            "pain".to_string()
        } else if para_lower.contains("fear") || para_lower.contains("terror") || para_lower.contains("horror") {
            "fear".to_string()
        } else if para_lower.contains("mysterious") || para_lower.contains("strange") {
            "mysterious".to_string()
        } else if para_lower.contains("calm") || para_lower.contains("peaceful") {
            "peaceful".to_string()
        } else if para_lower.contains("confus") || para_lower.contains("puzzle") {
            "confused".to_string()
        } else {
            "neutral".to_string()
        }
    }
    
    fn calculate_paragraph_intensity(paragraph: &str) -> f32 {
        let exclamations = paragraph.matches('!').count() as f32 * 0.3;
        let questions = paragraph.matches('?').count() as f32 * 0.2;
        let ellipses = paragraph.matches("...").count() as f32 * 0.4;
        let caps_words = paragraph.split_whitespace()
            .filter(|word| word.chars().any(|c| c.is_uppercase()))
            .count() as f32 * 0.1;
        
        (exclamations + questions + ellipses + caps_words / 10.0).min(1.0)
    }
    
    fn extract_emotions(paragraph: &str) -> Vec<String> {
        let para_lower = paragraph.to_lowercase();
        let mut emotions = Vec::new();
        
        let emotion_keywords = [
            ("fear", vec!["afraid", "scared", "terror", "horror", "frightened"]),
            ("pain", vec!["hurt", "painful", "agony", "suffering", "ache"]),
            ("confusion", vec!["confused", "puzzled", "bewildered", "perplexed"]),
            ("curiosity", vec!["curious", "wonder", "interested", "intrigued"]),
            ("determination", vec!["determined", "resolved", "focused", "committed"]),
            ("shock", vec!["shocked", "stunned", "amazed", "astounded", "surprised"]),
        ];
        
        for (emotion, keywords) in emotion_keywords.iter() {
            for keyword in keywords {
                if para_lower.contains(keyword) {
                    emotions.push(emotion.to_string());
                    break;
                }
            }
        }
        
        emotions
    }
    
    fn calculate_tension_level(paragraph: &str) -> f32 {
        let para_lower = paragraph.to_lowercase();
        let mut tension: f32 = 0.0;
        
        let tension_words = [
            "suddenly", "abruptly", "immediately", "instantly", "quickly",
            "danger", "threat", "warning", "urgent", "critical",
            "blood", "death", "kill", "murder", "attack"
        ];
        
        for word in tension_words.iter() {
            if para_lower.contains(word) {
                tension += 0.15;
            }
        }
        
        tension.min(1.0)
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